# HandwritingAI

> Turn your handwriting into text — and into a font. Runs 100% locally on your computer.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Offline](https://img.shields.io/badge/runs-offline-purple)

---

## What it does

HandwritingAI is a local desktop app that analyses photos of your handwriting and does two things:

1. **Transcribe** — reads your handwriting and converts it to editable text
2. **Font generation** — extracts each character and packages them into a `.ttf` font you can install and use anywhere

Everything runs on your machine. No data is sent anywhere.

---

## How it works

```
Your photo  →  Preprocessing  →  Line segmentation
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
           TrOCR model                           Character segmentation
           (transcription)                        (OpenCV connected components)
                    │                                       │
                    ▼                                       ▼
           Editable text                          Vectorize with potrace
                                                           │
                                                           ▼
                                                  Build .ttf with fonttools
```

---

## Quick start

### 1. Install

**macOS / Linux**
```bash
git clone https://github.com/yourusername/handwriting-ai.git
cd handwriting-ai
chmod +x install.sh
./install.sh
```

**Windows**
```
1. Clone or download this repo
2. Double-click install.bat
3. Follow the prompts
```

The installer handles everything automatically:

- Detects Python 3.10+ (tries `py`, `python`, `python3` — handles the Windows Store stub issue)
- Installs `potrace` automatically via winget, chocolatey, or scoop (Windows) / brew, apt, dnf, or pacman (macOS/Linux). Falls back to manual instructions only if none of those are available.
- Creates a Python virtual environment
- Installs all Python dependencies
- Pre-downloads the TrOCR model (~300 MB, one time only)

> **Windows note:** The installer uses plain ASCII output (`[OK]`, `[ERROR]`) for compatibility with all terminal types. If you see garbled characters, make sure you're running it by double-clicking rather than from within an existing terminal session.

### 2. Run

```bash
./run.sh        # macOS / Linux
# or double-click run.bat on Windows
```

Open your browser at **http://localhost:8000**

---

## Usage guide

### Step 1 — Write the prompt paragraph

The app shows you a specific paragraph to write out. It covers all 26 letters (upper and lower case), digits, and common punctuation. Write it on **white unlined paper** in **natural light**.

> **For best font results:** write in print (not cursive), clearly and consistently.
> **For best transcription:** any style works — the fine-tuning step adapts the model to your style.

### Step 2 — Upload your photos

Take 2–3 photos of your writing. The app checks each photo for quality (blur, exposure, resolution) and flags any issues before you start processing.

### Step 3 — Choose what to do

| Mode | What it does | Time |
|------|-------------|------|
| **Transcribe** | Reads handwriting → text using TrOCR | ~10–30 sec |
| **Fine-tune** | Trains the model on your handwriting style | ~5–10 min |
| **Generate font** | Segments characters → builds a `.ttf` file | ~2–5 min |

**Recommended order:** Fine-tune first → then Transcribe → then Generate Font.

---

## Project structure

```
handwriting-ai/
├── backend/
│   ├── main.py                  # FastAPI server + job queue
│   ├── recognizer/
│   │   ├── model.py             # TrOCR wrapper + fine-tuning
│   │   └── preprocess.py        # Quality checks, deskewing, binarization
│   └── font_generator/
│       ├── segmenter.py         # Line + character segmentation (OpenCV)
│       ├── vectorizer.py        # Bitmap → em-normalized SVG path (potrace)
│       └── builder.py           # SVG glyphs → .ttf (fonttools + cu2qu)
├── frontend/
│   └── index.html               # Local web UI
├── scripts/
│   └── test_htr.py              # CLI test script for the HTR pipeline
├── requirements.txt
├── install.sh                   # macOS / Linux installer
├── install.bat                  # Windows installer
└── .gitignore
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Local web server |
| `opencv-python` | Image processing, segmentation |
| `transformers` + `torch` | TrOCR model (HTR) |
| `cu2qu` | Cubic → quadratic bezier conversion for TTF fonts |
| `potrace` (system) | Bitmap → vector tracing |
| `fonttools` | Building the `.ttf` font file |
| `Pillow` | Image loading and manipulation |

---

## ML approach

This project deliberately avoids training from scratch. Instead it uses **Microsoft TrOCR** (`trocr-base-handwritten`), a pre-trained Transformer model that already understands handwriting. The fine-tuning step freezes the encoder and only updates the decoder layers using the user's own samples — this takes minutes on a CPU and produces a personalized model saved entirely locally.

The font pipeline doesn't use ML — it's pure computer vision (OpenCV segmentation + potrace vectorization + fonttools assembly).

---

## Testing

To test the HTR pipeline on a photo without running the full server:

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python scripts/test_htr.py path/to/your/photo.jpg

# save the preprocessed image to inspect it
python scripts/test_htr.py photo.jpg --save-preprocessed
```

---

## Known limitations

- **Cursive writing** is harder to segment for font generation — print works best for fonts
- **Lighting matters** — avoid shadows across the paper
- **Fine-tuning needs ~50+ line pairs** — writing the full prompt paragraph 2–3 times gives the best results
- Tested on Python 3.10–3.12, macOS 13+, Ubuntu 22.04, Windows 10/11

---

## Roadmap

- [ ] Cursive character segmentation (v2)
- [ ] Variable font support (weight/size variants)
- [ ] Batch transcription of multiple pages
- [ ] Export transcription as `.docx` / `.pdf`
- [ ] Drag-to-reorder glyph editor before font export

---

## License

MIT — do whatever you want with it.

---

*Built with TrOCR, OpenCV, potrace, fonttools, and cu2qu.*
