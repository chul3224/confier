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
rag: RecipeRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RecipeRAG()
    rag.index_documents()
    yield


app = FastAPI(title="레시피 알러지 챗봇", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


SYSTEM_PROMPT = """당신은 레시피 알러지 전문가 챗봇입니다.
한국 식품의약품안전처(식약처) 기준 22가지 알러지 유발 식품을 기반으로 답변합니다.

식약처 알러지 유발 22종: {allergens}

아래는 관련 레시피 정보입니다:
{recipe_context}

답변 규칙:
1. 레시피에 포함된 알러지 유발 성분을 정확히 나열하세요.
2. 알러지가 없으면 안전하다고 알려주세요.
3. 가능하면 대체 재료를 제안하세요.
4. 불확실한 경우 솔직히 말씀해주세요.
5. 한국어로 답변하세요."""


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

    chunks = rag.query(query)
    if chunks:
        recipe_context = "\n\n---\n\n".join(
            f"[{c['source']}]\n{c['text']}" for c in chunks
        )
    else:
        recipe_context = "현재 인덱싱된 레시피 문서가 없습니다. PDF를 업로드해주세요."

    system_msg = SYSTEM_PROMPT.format(
        allergens=", ".join(ALLERGEN_NAMES),
        recipe_context=recipe_context,
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

    detected = detect_allergens_in_text(query + " " + answer)

    return JSONResponse({
        "answer": answer,
        "detected_allergens": detected,
        "sources": list({c["source"] for c in chunks}),
    })


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    dest = DOCUMENTS_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = rag.index_documents()
    return JSONResponse({
        "message": f"{file.filename} 업로드 완료",
        "indexed": result["indexed"],
    })


@app.get("/status")
async def status():
    indexed_files = rag.get_indexed_files()
    return JSONResponse({
        "indexed_files": indexed_files,
        "model": OLLAMA_MODEL,
        "allergens": ALLERGEN_NAMES,
    })


@app.post("/reindex")
async def reindex():
    rag.reset()
    result = rag.index_documents()
    return JSONResponse(result)
