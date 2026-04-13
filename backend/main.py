"""
HandwritingAI — FastAPI backend
Serves the local web UI and exposes endpoints for:
  - uploading handwriting samples
  - transcribing handwriting images
  - generating a custom .ttf font
"""

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recognizer.model import HandwritingRecognizer
from recognizer.preprocess import preprocess_image
from font_generator.segmenter import segment_characters
from font_generator.vectorizer import vectorize_glyphs
from font_generator.builder import build_font

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
FONTS_DIR = STORAGE_DIR / "fonts"
MODELS_DIR = STORAGE_DIR / "models"

for d in [UPLOADS_DIR, FONTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="HandwritingAI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
FRONTEND_DIR = BASE_DIR.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# In-memory job tracker (for a local app, this is fine)
jobs: dict[str, dict] = {}

# ── The prompt paragraph ───────────────────────────────────────────────────────
PROMPT_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "0 1 2 3 4 5 6 7 8 9  ! ? . , ; : ' \" ( ) - /"
)


# ── Models ─────────────────────────────────────────────────────────────────────
class TranscribeRequest(BaseModel):
    image_path: str


class StatusResponse(BaseModel):
    job_id: str
    status: str          # "pending" | "processing" | "done" | "error"
    result: str | None = None
    error: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/prompt")
async def get_prompt():
    """Return the handwriting prompt the user should write out."""
    return {"paragraph": PROMPT_PARAGRAPH}


@app.post("/upload")
async def upload_samples(files: list[UploadFile] = File(...)):
    """
    Accept one or more handwriting sample images.
    Returns a list of saved file paths (used by subsequent endpoints).
    """
    saved = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
            raise HTTPException(400, f"Unsupported file type: {ext}")
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{ext}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(str(dest))
    return {"uploaded": saved, "count": len(saved)}


@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """
    Upload image(s) and transcribe the handwriting in them.
    Returns a job_id to poll for results.
    """
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "pending", "result": None, "error": None}

    # Save uploads first
    paths = []
    for file in files:
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{Path(file.filename).suffix}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        paths.append(str(dest))

    background_tasks.add_task(_run_transcription, job_id, paths)
    return {"job_id": job_id}


@app.post("/generate-font")
async def generate_font(
    background_tasks: BackgroundTasks,
    font_name: str = "MyHandwriting",
    files: list[UploadFile] = File(...),
):
    """
    Upload handwriting sample image(s) and generate a .ttf font.
    Returns a job_id to poll for results.
    """
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "pending", "result": None, "error": None}

    paths = []
    for file in files:
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{Path(file.filename).suffix}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        paths.append(str(dest))

    background_tasks.add_task(_run_font_generation, job_id, paths, font_name)
    return {"job_id": job_id}


@app.post("/fine-tune")
async def fine_tune(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """
    Fine-tune the HTR model on the user's handwriting samples.
    The model will use the known prompt paragraph as ground-truth labels.
    Returns a job_id to poll for results.
    """
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "pending", "result": None, "error": None}

    paths = []
    for file in files:
        dest = UPLOADS_DIR / f"{uuid.uuid4().hex}{Path(file.filename).suffix}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        paths.append(str(dest))

    background_tasks.add_task(_run_fine_tune, job_id, paths)
    return {"job_id": job_id}


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return StatusResponse(job_id=job_id, **j)


@app.get("/download/font/{font_name}")
async def download_font(font_name: str):
    """Download the generated .ttf font file."""
    font_path = FONTS_DIR / f"{font_name}.ttf"
    if not font_path.exists():
        raise HTTPException(404, "Font not found — has it been generated yet?")
    return FileResponse(
        str(font_path),
        media_type="font/ttf",
        filename=f"{font_name}.ttf",
    )


# ── Background task runners ────────────────────────────────────────────────────

def _run_transcription(job_id: str, image_paths: list[str]):
    try:
        jobs[job_id]["status"] = "processing"
        recognizer = HandwritingRecognizer(model_dir=str(MODELS_DIR))
        results = []
        for path in image_paths:
            img = preprocess_image(path)
            text = recognizer.transcribe(img)
            results.append(text)
        jobs[job_id]["result"] = "\n\n---\n\n".join(results)
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


def _run_font_generation(job_id: str, image_paths: list[str], font_name: str):
    try:
        jobs[job_id]["status"] = "processing"
        glyphs = {}
        for path in image_paths:
            img = preprocess_image(path)
            chars = segment_characters(img)
            glyphs.update(chars)  # {char: binary_image}

        if not glyphs:
            raise ValueError("No characters could be segmented from the images.")

        svg_glyphs = vectorize_glyphs(glyphs)  # {char: svg_path_data}
        font_path = FONTS_DIR / f"{font_name}.ttf"
        build_font(svg_glyphs, str(font_path), family_name=font_name)

        jobs[job_id]["result"] = f"/download/font/{font_name}"
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


def _run_fine_tune(job_id: str, image_paths: list[str]):
    try:
        jobs[job_id]["status"] = "processing"
        recognizer = HandwritingRecognizer(model_dir=str(MODELS_DIR))
        recognizer.fine_tune(image_paths, ground_truth=PROMPT_PARAGRAPH)
        jobs[job_id]["result"] = "Fine-tuning complete. Model saved."
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🖊️  HandwritingAI is running at http://localhost:8000\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
