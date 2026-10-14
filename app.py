import json
import os
import re
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


# ─────────────────────────────────────────────
# DB 로드
# ─────────────────────────────────────────────

def load_menu_db():
    global menu_db
    if MENU_JSON.exists():
        with open(MENU_JSON, encoding="utf-8") as f:
            menu_db = json.load(f)


# ─────────────────────────────────────────────
# 쿼리 유형 판별 & 컨텍스트 빌더
# ─────────────────────────────────────────────

_SKIP_TOKENS = {"코스", "단품", "잠실", "3가지"}
_RECOMMEND_KW = ["추천", "먹을 수 있", "괜찮은", "안전한", "피해야", "어떤 메뉴", "뭐 먹", "전체", "모든 메뉴"]


def find_matching_menus(query: str) -> list[tuple[str, dict]]:
    """메뉴명 키워드가 쿼리에 있으면 반환."""
    q = query.lower().replace(" ", "")
    matches = []
    for name, info in menu_db.items():
        tokens = [t for t in re.split(r"[_\s]+", name.lower())
                  if len(t) > 1 and t not in _SKIP_TOKENS]
        name_flat = name.lower().replace(" ", "")
        query_words = [w for w in re.split(r"\s+", query.lower()) if len(w) > 1]
        if any(t in q for t in tokens) or any(w in name_flat for w in query_words):
            matches.append((name, info))
    return matches


def find_menus_by_allergen(allergens: list[str]) -> dict[str, list[tuple]]:
    """알러지명 → 해당 알러지 포함 메뉴 역방향 조회."""
    result = {}
    for allergen in allergens:
        affected = [
            (name, info) for name, info in menu_db.items()
            if allergen in info.get("allergens", [])
        ]
        result[allergen] = affected
    return result


def build_menu_context(matches: list[tuple[str, dict]]) -> str:
    lines = ["[메뉴 알러지 DB 조회 결과]"]
    for name, info in matches:
        allergen_str = ", ".join(info["allergens"]) if info["allergens"] else "없음"
        lines.append(f"\n● {name} ({info['category']})")
        lines.append(f"  알러지: {allergen_str}")
        for a, srcs in info.get("allergen_sources", {}).items():
            lines.append(f"  - {a} 유발 재료: {', '.join(srcs)}")
    return "\n".join(lines)


_SAFE_MENU_KW = ["주문 가능", "먹을 수 있", "괜찮", "안전", "추천", "피해서", "제외하고", "뭐 먹"]


def build_allergen_context(allergen_results: dict[str, list], show_safe: bool = False) -> str:
    """알러지 역방향 조회 결과 컨텍스트. show_safe=True면 안전 메뉴 목록도 포함."""
    lines = ["[알러지 기반 메뉴 조회 결과]"]
    dangerous_names: set[str] = set()

    for allergen, menus in allergen_results.items():
        if not menus:
            lines.append(f"\n● {allergen}: 현재 메뉴 중 해당 알러지 포함 메뉴 없음 → 모든 메뉴 주문 가능")
        else:
            lines.append(f"\n● {allergen} 포함 메뉴 ({len(menus)}개) — 주문 불가:")
            for name, info in menus:
                srcs = info["allergen_sources"].get(allergen, [])
                lines.append(f"  ✗ {name} ({info['category']}) / 유발 재료: {', '.join(srcs)}")
                dangerous_names.add(name)

    if show_safe and menu_db:
        safe = [(n, i) for n, i in menu_db.items() if n not in dangerous_names]
        if safe:
            lines.append(f"\n● 주문 가능 메뉴 ({len(safe)}개):")
            for name, info in safe:
                other_allergens = ", ".join(info["allergens"]) if info["allergens"] else "알러지 없음"
                lines.append(f"  ✓ {name} ({info['category']}) / 기타 알러지: {other_allergens}")
        else:
            lines.append("\n● 주문 가능 메뉴: 없음 (모든 메뉴에 해당 알러지 포함)")

    return "\n".join(lines)


def build_full_menu_summary() -> str:
    if not menu_db:
        return ""
    lines = ["[전체 메뉴 알러지 현황]"]
    for name, info in menu_db.items():
        allergen_str = ", ".join(info["allergens"]) if info["allergens"] else "없음"
        lines.append(f"  {name}: {allergen_str}")
    return "\n".join(lines)


def classify_query(query: str) -> tuple[str, list]:
    """
    쿼리 유형 분류.
    반환: (유형, 추가 데이터)
      'menu'      → 매칭된 메뉴 리스트
      'allergen'  → 감지된 알러지 리스트
      'recommend' → []
      'general'   → []
    """
    # 1순위: 메뉴명 직접 매칭
    matched_menus = find_matching_menus(query)
    if matched_menus:
        return "menu", matched_menus

    # 2순위: 알러지명 감지 (역방향 조회)
    detected = detect_allergens_in_text(query)
    if detected:
        return "allergen", detected

    # 3순위: 추천/회피 질문
    if any(kw in query for kw in _RECOMMEND_KW):
        return "recommend", []

    return "general", []


# ─────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 홀 직원을 위한 레시피 알러지 안내 챗봇입니다.
한국 식약처 기준 22가지 알러지 유발 식품을 기반으로 답변합니다.

식약처 알러지 유발 22종: {allergens}

{context}

[답변 규칙]
1. 조회 결과가 있으면 반드시 그 데이터를 근거로 답변하세요.
2. 알러지 포함 여부와 유발 재료를 명확히 안내하세요.
3. 레시피 세부 내용(배합비율, 조리법)은 절대 포함하지 마세요.
4. 대체 재료 제안은 현재 지원되지 않습니다. 요청 시 "추후 업데이트 예정"이라고만 답하세요.
5. 짧고 명확하게 한국어로 답변하세요."""


# ─────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────

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

    query_type, data = classify_query(query)

    if query_type == "menu":
        context = build_menu_context(data)
        response_allergens = list({a for _, info in data for a in info.get("allergens", [])})
        sources = [name for name, _ in data]

    elif query_type == "allergen":
        allergen_results = find_menus_by_allergen(data)
        show_safe = any(kw in query for kw in _SAFE_MENU_KW)
        context = build_allergen_context(allergen_results, show_safe=show_safe)
        response_allergens = data
        sources = [name for menus in allergen_results.values() for name, _ in menus]

    elif query_type == "recommend":
        context = build_full_menu_summary()
        response_allergens = []
        sources = []

    else:
        chunks = rag.query(query, n_results=3)
        context = ("[레시피 문서 검색 결과]\n" + "\n---\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
                   if chunks else "현재 참조할 데이터가 없습니다.")
        response_allergens = detect_allergens_in_text(query)
        sources = list({c["source"] for c in chunks}) if chunks else []

    system_msg = SYSTEM_PROMPT.format(
        allergens=", ".join(ALLERGEN_NAMES),
        context=context,
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

    return JSONResponse({
        "answer": answer,
        "detected_allergens": response_allergens,
        "sources": list(dict.fromkeys(sources)),  # 중복 제거, 순서 유지
        "query_type": query_type,
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
