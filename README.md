# VoiceForge AI TTS Server

A specialized, local-only Python FastAPI AI Voice Engine managing dual state-of-the-art TTS models: Qwen3-TTS and Coqui XTTSv2.

## What This App Does

- Serves `POST /v1/audio/speech` for AI voice syntheses.
- Serves `POST /admin/unload-models` for memory reclamation.
- Features a sophisticated `VRAM-aware ModelManager` ensuring stable execution of concurrent AI structures on limited (e.g., 8 GB RTX 3070) GPU memory topologies by dynamically unmounting and bootstrapping checkpoints based on incoming queue demands.
- Processes audio extensively via pre-conditioning flows: decoding, resampling strictly at 24kHz mono space, spectral denoising (`noisereduce`), VAD trimming, normalization, and enforcing strict 5-second embedding caps.
- Auto-handles token limit calculations, auto-wrapping, and multi-chunk concatenation without crashing the GPU inference loop natively.

## Core Dependencies

- `fastapi`, `uvicorn`, `pydantic`
- `torch`, `torchaudio`
- `transformers==4.57.3`, `accelerate`
- `TTS>=0.22.0` (Coqui XTTSv2 engine)
- `librosa`, `scipy`, `soundfile`, `imageio-ffmpeg`
- `noisereduce` (Spectral denoise optimization)

## First-Run Model Download Behavior

- Qwen base model downloads efficiently to `apps/fastapi/Qwen3-TTS-12Hz-0.6B-Base`.
- Qwen custom-voice checkpoint downloads to `apps/fastapi/Qwen3-TTS-12Hz-0.6B-CustomVoice`.
- Hugging Face cache maps intelligently to `apps/fastapi/.cache`.
- Coqui XTTS cache points dynamically fully localized bypassing standard persistent profile paths.

This guarantees models remain fully tethered to the app directory.

## Prerequisites

- Python 3.10+
- CUDA-enabled PyTorch strongly recommended for functional operation speed.

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

- Keep the environment container (`venv/`) securely out of git scope.
- Model directories dynamically bypass source tracks.
- Adhere strictly to `.gitignore` principles locally keeping only code tracked.