# main.py
# local FastAPI server — serves the frontend and runs HTR + font generation jobs

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recognizer.model import HandwritingRecognizer
from recognizer.preprocess import preprocess_image, check_quality
from font_generator.segmenter import segment_characters
from font_generator.vectorizer import vectorize_glyphs
from font_generator.builder import build_font

# paths
BASE = Path(__file__).parent
STORAGE = BASE / "storage"
UPLOADS = STORAGE / "uploads"
FONTS = STORAGE / "fonts"
MODELS = STORAGE / "models"
FRONTEND = BASE.parent / "frontend"

for d in [UPLOADS, FONTS, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="HandwritingAI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

# in-memory job tracker (fine for a local single-user app)
jobs: dict[str, dict] = {}

PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "0 1 2 3 4 5 6 7 8 9  ! ? . , ; : ' \" ( ) - /"
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# models

class StatusResponse(BaseModel):
    job_id: str
    status: str              # pending | processing | done | error
    result: str | None = None
    error: str | None = None
    progress: str | None = None


# routes

@app.get("/")
async def root():
    return FileResponse(str(FRONTEND / "index.html"))


@app.get("/prompt")
async def get_prompt():
    return {"paragraph": PROMPT}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Save uploaded images and run quality checks. Returns per-file feedback."""
    results = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            results.append({"filename": f.filename, "ok": False, "path": None,
                            "errors": [f"Unsupported type: {ext}"], "warnings": []})
            continue

        dest = UPLOADS / f"{uuid.uuid4().hex}{ext}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)

        report = check_quality(str(dest))
        results.append({
            "filename": f.filename,
            "ok": report.ok,
            "path": str(dest),
            "errors": report.errors,
            "warnings": report.warnings,
        })

    return {"results": results, "all_ok": all(r["ok"] for r in results), "count": len(results)}


@app.post("/transcribe")
async def transcribe(bg: BackgroundTasks, files: list[UploadFile] = File(...)):
    """Upload images and transcribe the handwriting in them. Returns a job_id."""
    job_id, paths = _new_job(), await _save_uploads(files)
    bg.add_task(_run_transcribe, job_id, paths)
    return {"job_id": job_id}


@app.post("/generate-font")
async def generate_font(bg: BackgroundTasks, files: list[UploadFile] = File(...), font_name: str = "MyHandwriting"):
    """Upload images and generate a .otf font. Returns a job_id."""
    job_id, paths = _new_job(), await _save_uploads(files)
    bg.add_task(_run_font, job_id, paths, font_name)
    return {"job_id": job_id}


@app.post("/fine-tune")
async def fine_tune(bg: BackgroundTasks, files: list[UploadFile] = File(...)):
    """Upload images and fine-tune the HTR model on them. Returns a job_id."""
    job_id, paths = _new_job(), await _save_uploads(files)
    bg.add_task(_run_finetune, job_id, paths)
    return {"job_id": job_id}


@app.get("/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return StatusResponse(job_id=job_id, **jobs[job_id])


@app.get("/download/font/{font_name}")
async def download_font(font_name: str):
    p = FONTS / f"{font_name}.otf"
    if not p.exists():
        raise HTTPException(404, "Font not found — has it been generated yet?")
    return FileResponse(str(p), media_type="font/otf", filename=f"{font_name}.otf")


# background tasks

def _run_transcribe(job_id: str, paths: list[str]):
    try:
        _progress(job_id, "Loading model...")
        recognizer = HandwritingRecognizer(model_dir=str(MODELS))
        results = []

        for i, path in enumerate(paths):
            _progress(job_id, f"Checking image {i + 1}/{len(paths)}...")
            report = check_quality(path)
            if not report.ok:
                raise ValueError(f"Image {i + 1} failed: {'; '.join(report.errors)}")

            _progress(job_id, f"Preprocessing image {i + 1}/{len(paths)}...")
            img = preprocess_image(path)

            def cb(cur: int, tot: int):
                _progress(job_id, f"Image {i + 1}/{len(paths)} — line {cur}/{tot}")

            results.append(recognizer.transcribe(img, progress_cb=cb))

        jobs[job_id].update(status="done", result="\n\n---\n\n".join(results), progress=None)
    except Exception as e:
        jobs[job_id].update(status="error", error=str(e))


def _run_font(job_id: str, paths: list[str], font_name: str):
    try:
        _progress(job_id, "Segmenting characters...")
        glyphs: dict = {}
        for i, path in enumerate(paths):
            _progress(job_id, f"Segmenting image {i + 1}/{len(paths)}...")
            img = preprocess_image(path)
            glyphs.update(segment_characters(img))

        if not glyphs:
            raise ValueError("No characters could be segmented. Try a clearer photo.")

        _progress(job_id, f"Vectorizing {len(glyphs)} glyphs...")
        svg_glyphs = vectorize_glyphs(glyphs)

        if not svg_glyphs:
            raise ValueError("Vectorization produced no usable glyphs.")

        _progress(job_id, "Building font file...")
        out = str(FONTS / f"{font_name}.otf")
        build_font(svg_glyphs, out, family_name=font_name)

        jobs[job_id].update(status="done", result=f"/download/font/{font_name}", progress=None)
    except Exception as e:
        jobs[job_id].update(status="error", error=str(e))


def _run_finetune(job_id: str, paths: list[str]):
    try:
        _progress(job_id, "Loading model...")
        recognizer = HandwritingRecognizer(model_dir=str(MODELS))
        _progress(job_id, "Fine-tuning (this takes a few minutes)...")
        recognizer.fine_tune(paths, ground_truth=PROMPT)
        jobs[job_id].update(status="done", result="Fine-tuning complete. Model saved.", progress=None)
    except Exception as e:
        jobs[job_id].update(status="error", error=str(e))


# helpers

def _new_job() -> str:
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "pending", "result": None, "error": None, "progress": None}
    return job_id


async def _save_uploads(files: list[UploadFile]) -> list[str]:
    paths = []
    for f in files:
        dest = UPLOADS / f"{uuid.uuid4().hex}{Path(f.filename).suffix}"
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(str(dest))
    return paths


def _progress(job_id: str, msg: str):
    jobs[job_id].update(status="processing", progress=msg)


if __name__ == "__main__":
    import uvicorn
    print("\n🖊️  HandwritingAI running at http://localhost:8000\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
