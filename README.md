# VoiceForge AI TTS Server

Combined FastAPI TTS service for Qwen3-TTS and XTTSv2.

## What This App Does

- Serves `POST /v1/audio/speech`
- Serves `GET /health`
- Serves `POST /admin/unload-models`
- Uses a VRAM-aware model manager to hot-swap between Qwen and XTTS on limited GPU memory

## First-Run Model Download Behavior

- Qwen base downloads into `apps/fastapi/Qwen3-TTS-12Hz-0.6B-Base`
- Qwen custom voice downloads into `apps/fastapi/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- Hugging Face cache is redirected into `apps/fastapi/.cache`
- Coqui XTTS cache is redirected into `apps/fastapi`

This keeps model storage inside the app folder instead of scattering it across user-profile cache locations.

## Prerequisites

- Python 3.10+
- CUDA-enabled PyTorch recommended

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Repo Notes

- Do not commit downloaded model weights.
- Keep only code, requirements, and documentation tracked.
- `.gitkeep` files preserve the local model folders in a clean clone.