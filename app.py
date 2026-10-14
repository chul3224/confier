import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import ollama
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from allergens import ALLERGEN_NAMES, detect_allergens_in_text
from rag import RecipeRAG


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
DOCUMENTS_DIR = Path("documents")
MENU_JSON = Path("data/menu_allergens.json")

rag: RecipeRAG | None = None
menu_db: dict = {}


def load_menu_db():
    global menu_db
    if MENU_JSON.exists():
        with open(MENU_JSON, encoding="utf-8") as f:
            menu_db = json.load(f)


import re as _re

_SKIP_TOKENS = {"코스", "단품", "잠실", "3가지"}

def find_matching_menus(query: str) -> list[tuple[str, dict]]:
    """쿼리에서 메뉴명 매칭 — _와 공백 모두 분리, 양방향 포함 검사."""
    matches = []
    q = query.lower().replace(" ", "")
    for name, info in menu_db.items():
        # 메뉴명을 _ 와 공백으로 분리해 의미 있는 토큰 추출
        tokens = [t for t in _re.split(r"[_\s]+", name.lower())
                  if len(t) > 1 and t not in _SKIP_TOKENS]
        # 메뉴 토큰이 쿼리에 포함되거나, 쿼리 단어가 메뉴명에 포함되면 매칭
        name_flat = name.lower().replace(" ", "")
        query_words = [w for w in _re.split(r"\s+", query.lower()) if len(w) > 1]
        if any(t in q for t in tokens) or any(w in name_flat for w in query_words):
            matches.append((name, info))
    return matches


def build_menu_context(matches: list[tuple[str, dict]]) -> str:
    if not matches:
        return ""
    lines = ["[메뉴 알러지 데이터베이스 조회 결과]"]
    for name, info in matches:
        allergen_str = ", ".join(info["allergens"]) if info["allergens"] else "없음"
        lines.append(f"\n● {name} ({info['category']})")
        lines.append(f"  알러지: {allergen_str}")
        for a, srcs in info.get("allergen_sources", {}).items():
            lines.append(f"  - {a} 유발 재료: {', '.join(srcs)}")
    return "\n".join(lines)


def build_full_menu_summary() -> str:
    """전체 메뉴 알러지 요약 (추천/회피 질문용)."""
    if not menu_db:
        return ""
    lines = ["[전체 메뉴 알러지 현황]"]
    for name, info in menu_db.items():
        allergen_str = ", ".join(info["allergens"]) if info["allergens"] else "없음"
        lines.append(f"  {name}: {allergen_str}")
    return "\n".join(lines)


RECOMMEND_KEYWORDS = ["추천", "먹을 수 있", "괜찮은", "안전한", "피해야", "어떤 메뉴", "뭐 먹", "전체", "모든 메뉴"]

def is_recommendation_query(query: str) -> bool:
    return any(kw in query for kw in RECOMMEND_KEYWORDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    load_menu_db()
    rag = RecipeRAG()
    rag.index_documents()
    yield


app = FastAPI(title="레시피 알러지 챗봇", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


SYSTEM_PROMPT = """당신은 레시피 알러지 전문가 챗봇입니다.
한국 식품의약품안전처(식약처) 기준 22가지 알러지 유발 식품을 기반으로 답변합니다.

식약처 알러지 유발 22종: {allergens}

{menu_context}

{rag_context}

답변 규칙:
1. 메뉴 데이터베이스 조회 결과가 있으면 그것을 우선으로 답변하세요.
2. 알러지 유발 성분과 그 유발 재료를 명확히 알려주세요.
3. 알러지가 없으면 안전하다고 알려주세요.
4. 레시피 세부 내용(배합비율, 조리법)은 답변에 포함하지 마세요.
5. 불확실한 경우 솔직히 말씀해주세요.
6. 한국어로 답변하세요."""


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="메시지를 입력해주세요.")

    # 1. 메뉴 DB 직접 조회
    matched_menus = find_matching_menus(query)
    if matched_menus:
        menu_context = build_menu_context(matched_menus)
    elif is_recommendation_query(query):
        menu_context = build_full_menu_summary()
    else:
        menu_context = ""

    # 2. RAG 보조 검색
    chunks = rag.query(query, n_results=2) if not matched_menus else []
    if chunks:
        rag_context = "[레시피 문서 검색 결과]\n" + "\n\n---\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in chunks
        )
    else:
        rag_context = ""

    if not menu_context and not rag_context:
        rag_context = "현재 참조할 레시피 데이터가 없습니다."

    system_msg = SYSTEM_PROMPT.format(
        allergens=", ".join(ALLERGEN_NAMES),
        menu_context=menu_context,
        rag_context=rag_context,
    )

    messages = [{"role": "system", "content": system_msg}]
    for h in req.history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": query})

    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        answer = response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama 오류: {e}")

    detected = list({a for _, info in matched_menus for a in info.get("allergens", [])}) or detect_allergens_in_text(query + " " + answer)

    return JSONResponse({
        "answer": answer,
        "detected_allergens": detected,
        "sources": [name for name, _ in matched_menus] or list({c["source"] for c in chunks}),
    })


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    dest = DOCUMENTS_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    result = rag.index_documents()
    return JSONResponse({"message": f"{file.filename} 업로드 완료", "indexed": result["indexed"]})


@app.get("/status")
async def status():
    return JSONResponse({
        "indexed_files": rag.get_indexed_files(),
        "menu_db_count": len(menu_db),
        "model": OLLAMA_MODEL,
        "allergens": ALLERGEN_NAMES,
    })


@app.get("/menus")
async def list_menus():
    return JSONResponse({
        name: {"category": info["category"], "allergens": info["allergens"]}
        for name, info in menu_db.items()
    })


@app.post("/reindex")
async def reindex():
    rag.reset()
    return JSONResponse(rag.index_documents())
