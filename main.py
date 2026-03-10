import os
import io
import gc
import copy
import asyncio
import re
import torch
import traceback
import threading
import imageio_ffmpeg
import shutil

_ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
_ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
_ffmpeg_alias = os.path.join(_ffmpeg_dir, "ffmpeg.exe")
if not os.path.exists(_ffmpeg_alias):
    try:
        shutil.copy(_ffmpeg_exe, _ffmpeg_alias)
    except:
        pass
os.environ["PATH"] += os.pathsep + _ffmpeg_dir

import librosa
import audioread
import numpy as np
import soundfile as sf
import torchaudio

try:
    import noisereduce as nr
except ImportError:
    nr = None
    print("[Startup] noisereduce not found — XTTS reference denoising disabled.")

# torchaudio >= 2.6 defaults to torchcodec which requires a separate install.
# Fall back to soundfile (already installed) when torchcodec is absent.
try:
    import torchcodec  # noqa: F401  — if present, torchaudio.load works natively
except ImportError:

    def _torchaudio_load_sf(path, *args, **kwargs):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(data.T), sr  # (channels, samples), sr

    torchaudio.load = _torchaudio_load_sf
    print("[Startup] torchcodec not found — torchaudio.load patched to use soundfile.")
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from qwen_tts import Qwen3TTSModel

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────
# Device & model path resolution
# ─────────────────────────────────────────────
device = "cuda:0" if torch.cuda.is_available() else "cpu"
_FASTAPI_DIR = os.path.dirname(os.path.abspath(__file__))
_FASTAPI_CACHE_DIR = os.path.join(_FASTAPI_DIR, ".cache")

os.makedirs(_FASTAPI_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(_FASTAPI_CACHE_DIR, "huggingface"))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE", os.path.join(_FASTAPI_CACHE_DIR, "huggingface", "hub")
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    os.path.join(_FASTAPI_CACHE_DIR, "huggingface", "transformers"),
)
os.environ.setdefault("XDG_CACHE_HOME", _FASTAPI_CACHE_DIR)
os.environ.setdefault("TTS_HOME", _FASTAPI_DIR)


def _patch_xtts_transformers_compatibility() -> None:
    """
    Patch Coqui XTTS's custom generation mixin for newer transformers builds.

    Coqui TTS 0.22 ships a custom `NewGenerationMixin.generate()` copied from an
    older transformers release. On transformers 5.x, generation now expects
    prepared private token tensors such as `_eos_token_tensor`. XTTS skips that
    preparation step, which causes:

        AttributeError: 'StreamGenerationConfig' object has no attribute '_eos_token_tensor'

    This wrapper pre-populates the special-token tensors before delegating to the
    original XTTS generate implementation.
    """
    try:
        from TTS.tts.layers.xtts.stream_generator import (
            NewGenerationMixin,
            StreamGenerationConfig,
        )
    except Exception as exc:
        print(f"[Startup] XTTS compatibility patch skipped: {exc}")
        return

    original_generate = NewGenerationMixin.generate
    if getattr(original_generate, "_voiceforge_xtts_special_tokens_patch", False):
        return

    from transformers import LogitsProcessorList, StoppingCriteriaList

    if not hasattr(NewGenerationMixin, "sample") and hasattr(
        NewGenerationMixin, "_sample"
    ):

        def _compat_sample(
            self,
            input_ids,
            logits_processor=None,
            logits_warper=None,
            stopping_criteria=None,
            pad_token_id=None,
            eos_token_id=None,
            output_scores=None,
            return_dict_in_generate=None,
            synced_gpus=False,
            **model_kwargs,
        ):
            generation_config = getattr(self, "generation_config", None)
            if generation_config is None:
                generation_config = StreamGenerationConfig.from_model_config(
                    self.config
                )
            elif not isinstance(generation_config, StreamGenerationConfig):
                generation_config = StreamGenerationConfig(
                    **generation_config.to_dict()
                )
            else:
                generation_config = copy.deepcopy(generation_config)

            generation_config.do_sample = True
            if pad_token_id is not None:
                generation_config.pad_token_id = pad_token_id
            if eos_token_id is not None:
                generation_config.eos_token_id = eos_token_id
            if output_scores is not None:
                generation_config.output_scores = output_scores
            if return_dict_in_generate is not None:
                generation_config.return_dict_in_generate = return_dict_in_generate

            device_for_tokens = getattr(
                input_ids, "device", getattr(self, "device", None)
            )
            if hasattr(self, "_prepare_special_tokens"):
                self._prepare_special_tokens(
                    generation_config,
                    kwargs_has_attention_mask=model_kwargs.get("attention_mask")
                    is not None,
                    device=device_for_tokens,
                )

            merged_processors = LogitsProcessorList()
            if logits_processor is not None:
                merged_processors.extend(logits_processor)
            if logits_warper is not None:
                merged_processors.extend(logits_warper)

            return self._sample(
                input_ids,
                logits_processor=merged_processors,
                stopping_criteria=stopping_criteria or StoppingCriteriaList(),
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        NewGenerationMixin.sample = _compat_sample

    if not hasattr(NewGenerationMixin, "_get_logits_warper"):

        try:
            from transformers import LogitNormalization
        except ImportError:
            LogitNormalization = None

        _warper_classes = {}
        for _name in (
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "MinPLogitsWarper",
            "TypicalLogitsWarper",
            "EpsilonLogitsWarper",
            "EtaLogitsWarper",
        ):
            try:
                _warper_classes[_name] = getattr(
                    __import__("transformers", fromlist=[_name]), _name
                )
            except (ImportError, AttributeError):
                _warper_classes[_name] = None

        def _compat_get_logits_warper(self, generation_config):
            warpers = LogitsProcessorList()
            eos_token_tensor = getattr(generation_config, "_eos_token_tensor", None)

            if generation_config.num_beams > 1:
                if isinstance(eos_token_tensor, list):
                    min_tokens_to_keep = len(eos_token_tensor) + 1
                elif isinstance(eos_token_tensor, torch.Tensor):
                    min_tokens_to_keep = eos_token_tensor.shape[0] + 1
                else:
                    min_tokens_to_keep = 2
            else:
                min_tokens_to_keep = 1

            temperature_cls = _warper_classes["TemperatureLogitsWarper"]
            if (
                temperature_cls is not None
                and generation_config.temperature is not None
                and generation_config.temperature != 1.0
            ):
                warpers.append(temperature_cls(generation_config.temperature))

            top_k_cls = _warper_classes["TopKLogitsWarper"]
            if (
                top_k_cls is not None
                and generation_config.top_k is not None
                and generation_config.top_k != 0
            ):
                warpers.append(
                    top_k_cls(
                        top_k=generation_config.top_k,
                        min_tokens_to_keep=min_tokens_to_keep,
                    )
                )

            top_p_cls = _warper_classes["TopPLogitsWarper"]
            if (
                top_p_cls is not None
                and generation_config.top_p is not None
                and generation_config.top_p < 1.0
            ):
                warpers.append(
                    top_p_cls(
                        top_p=generation_config.top_p,
                        min_tokens_to_keep=min_tokens_to_keep,
                    )
                )

            min_p_cls = _warper_classes["MinPLogitsWarper"]
            if (
                min_p_cls is not None
                and getattr(generation_config, "min_p", None) is not None
            ):
                warpers.append(
                    min_p_cls(
                        min_p=generation_config.min_p,
                        min_tokens_to_keep=min_tokens_to_keep,
                    )
                )

            typical_cls = _warper_classes["TypicalLogitsWarper"]
            if (
                typical_cls is not None
                and getattr(generation_config, "typical_p", None) is not None
                and generation_config.typical_p < 1.0
            ):
                warpers.append(
                    typical_cls(
                        mass=generation_config.typical_p,
                        min_tokens_to_keep=min_tokens_to_keep,
                    )
                )

            epsilon_cls = _warper_classes["EpsilonLogitsWarper"]
            epsilon_cutoff = getattr(generation_config, "epsilon_cutoff", None)
            if (
                epsilon_cls is not None
                and epsilon_cutoff is not None
                and 0.0 < epsilon_cutoff < 1.0
            ):
                warpers.append(
                    epsilon_cls(
                        epsilon=epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep
                    )
                )

            eta_cls = _warper_classes["EtaLogitsWarper"]
            eta_cutoff = getattr(generation_config, "eta_cutoff", None)
            if (
                eta_cls is not None
                and eta_cutoff is not None
                and 0.0 < eta_cutoff < 1.0
            ):
                warpers.append(
                    eta_cls(
                        epsilon=eta_cutoff,
                        min_tokens_to_keep=min_tokens_to_keep,
                        device=getattr(self, "device", None),
                    )
                )

            if LogitNormalization is not None and getattr(
                generation_config, "renormalize_logits", False
            ):
                warpers.append(LogitNormalization())

            return warpers

        NewGenerationMixin._get_logits_warper = _compat_get_logits_warper

    def patched_generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        seed=0,
        **kwargs,
    ):
        prepared_config = generation_config
        if prepared_config is None:
            prepared_config = getattr(self, "generation_config", None)

        if prepared_config is not None and not isinstance(
            prepared_config, StreamGenerationConfig
        ):
            if hasattr(prepared_config, "to_dict"):
                prepared_config = StreamGenerationConfig(**prepared_config.to_dict())
            else:
                prepared_config = StreamGenerationConfig.from_model_config(self.config)

        if prepared_config is not None and hasattr(self, "_prepare_special_tokens"):
            prepared_config = copy.deepcopy(prepared_config)
            prepared_config.update(**kwargs)
            kwargs_has_attention_mask = kwargs.get("attention_mask") is not None
            device_for_tokens = getattr(inputs, "device", getattr(self, "device", None))
            self._prepare_special_tokens(
                prepared_config,
                kwargs_has_attention_mask=kwargs_has_attention_mask,
                device=device_for_tokens,
            )

        return original_generate(
            self,
            inputs=inputs,
            generation_config=prepared_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            seed=seed,
            **kwargs,
        )

    patched_generate._voiceforge_xtts_special_tokens_patch = True
    NewGenerationMixin.generate = patched_generate
    print("[Startup] XTTS compatibility patch enabled for newer transformers versions.")


_patch_xtts_transformers_compatibility()

# Base model  → voice cloning (generate_voice_clone)
_LOCAL_BASE_DIR = os.path.join(_FASTAPI_DIR, "Qwen3-TTS-12Hz-0.6B-Base")
BASE_MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# CustomVoice model → built-in speaker IDs (generate_custom_voice)
_LOCAL_CUSTOM_DIR = os.path.join(_FASTAPI_DIR, "Qwen3-TTS-12Hz-0.6B-CustomVoice")
CUSTOM_MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def _ensure_local_hf_model(local_dir: str, repo_id: str) -> str:
    config_path = os.path.join(local_dir, "config.json")
    if os.path.isdir(local_dir) and os.path.isfile(config_path):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required for first-run model download. "
            "Install transformers dependencies before starting the TTS server."
        ) from exc

    print(f"[Startup] Downloading {repo_id} into {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    return local_dir


# XTTSv2 supported languages
XTTS_SUPPORTED_LANGUAGES = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "ja",
    "hu",
    "ko",
    "hi",
}

# Max reference audio duration (model only needs ~3s, 5s for safety)
MAX_REF_DURATION_SEC = 5
XTTS_TARGET_SPEED = 1.28
XTTS_TEMPERATURE = 0.7
XTTS_TOP_K = 20
XTTS_TOP_P = 0.75
XTTS_REPETITION_PENALTY = 2.0
XTTS_JOIN_SILENCE_MS = 60
XTTS_CHUNK_MARGIN_CHARS = 24
XTTS_MAX_CHARS_DEFAULT = 200
XTTS_MAX_CHARS_HI = 100


# ─────────────────────────────────────────────
# VRAM-aware ModelManager
# ─────────────────────────────────────────────
class ModelManager:
    """
    Hot-swaps model groups in VRAM on an RTX 3070 (8 GB).

    Groups:
      'qwen'  — Base (~1.3 GB) + CustomVoice (~1.3 GB) = ~2.6 GB bfloat16
      'xtts'  — XTTSv2 (~2.5 GB)

    Only one group is resident in VRAM at a time.  Models load lazily
    on the first request that needs them; the outgoing group is deleted
    and VRAM is freed before the incoming group is loaded.

    This leaves headroom for Phase 6 video generation (SadTalker/Wav2Lip
    ~3-4 GB): unload TTS → free ~2.5 GB → load video model.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active_group: Optional[str] = None
        self._qwen_clone = None  # Qwen3-TTS Base — voice cloning
        self._qwen_speaker = None  # Qwen3-TTS CustomVoice — built-in speakers
        self._xtts = None  # XTTSv2 — Coqui

    # ── VRAM helpers ───────────────────────────────────────────────────────
    def _free_vram(self):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  [VRAM] After free: {allocated:.2f} GB allocated")

    def _log_vram(self, label: str):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"  [VRAM] {label}: {allocated:.2f} GB alloc / {reserved:.2f} GB reserved"
            )

    # ── Qwen group ─────────────────────────────────────────────────────────
    def _ensure_qwen(self):
        with self._lock:
            if self._active_group == "qwen":
                return

            if self._xtts is not None:
                print("[ModelManager] Unloading XTTSv2 to free VRAM for Qwen...")
                del self._xtts
                self._xtts = None
                self._free_vram()

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            base_model_path = _ensure_local_hf_model(_LOCAL_BASE_DIR, BASE_MODEL_REPO)
            custom_model_path = _ensure_local_hf_model(
                _LOCAL_CUSTOM_DIR, CUSTOM_MODEL_REPO
            )

            print(f"[ModelManager] Loading Qwen Base model from: {base_model_path}")
            self._qwen_clone = Qwen3TTSModel.from_pretrained(
                base_model_path, device_map=device, dtype=dtype
            )
            self._log_vram("After Base model load")

            print(
                f"[ModelManager] Loading Qwen CustomVoice model from: {custom_model_path}"
            )
            self._qwen_speaker = Qwen3TTSModel.from_pretrained(
                custom_model_path, device_map=device, dtype=dtype
            )
            self._log_vram("After CustomVoice load")

            self._active_group = "qwen"
            print("[ModelManager] Qwen group ready.")

    # ── XTTSv2 group ───────────────────────────────────────────────────────
    def _ensure_xtts(self):
        with self._lock:
            if self._active_group == "xtts":
                return

            if self._qwen_clone is not None or self._qwen_speaker is not None:
                print("[ModelManager] Unloading Qwen models to free VRAM for XTTSv2...")
                del self._qwen_clone, self._qwen_speaker
                self._qwen_clone = None
                self._qwen_speaker = None
                self._free_vram()

            try:
                from TTS.api import TTS  # Lazy import — Coqui TTS is an optional dep
            except ImportError:
                raise RuntimeError(
                    "Coqui TTS is not installed. Run: pip install TTS>=0.22.0"
                )

            os.environ["COQUI_TOS_AGREED"] = "1"

            # Prefer the local model folder so nothing is downloaded to C: AppData.
            # Folder name mirrors Coqui's on-disk naming convention.
            _local_xtts_dir = os.path.join(
                _FASTAPI_DIR, "tts_models--multilingual--multi-dataset--xtts_v2"
            )
            _config_path = os.path.join(_local_xtts_dir, "config.json")

            if os.path.isdir(_local_xtts_dir) and os.path.isfile(_config_path):
                print(
                    f"[ModelManager] Loading XTTSv2 from local folder: {_local_xtts_dir}"
                )
                self._xtts = TTS(
                    model_path=_local_xtts_dir,
                    config_path=_config_path,
                    gpu=(device != "cpu"),
                )
            else:
                # Fallback — download on first run (model not yet in local folder)
                print(
                    f"[ModelManager] Local XTTSv2 folder not found; downloading model..."
                )
                self._xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
                    device
                )
            self._log_vram("After XTTSv2 load")
            self._active_group = "xtts"
            print("[ModelManager] XTTSv2 ready.")

    # ── Accessors ──────────────────────────────────────────────────────────
    def get_qwen_clone(self):
        self._ensure_qwen()
        return self._qwen_clone

    def get_qwen_speaker(self):
        self._ensure_qwen()
        return self._qwen_speaker

    def get_xtts(self):
        self._ensure_xtts()
        return self._xtts

    def unload_all(self):
        with self._lock:
            print("[ModelManager] Unloading all models...")
            del self._qwen_clone, self._qwen_speaker, self._xtts
            self._qwen_clone = self._qwen_speaker = self._xtts = None
            self._active_group = None
            self._free_vram()


model_manager = ModelManager()


# ─────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("VoiceForge AI ready — models load lazily on first request.")
    yield
    model_manager.unload_all()


app = FastAPI(title="VoiceForge AI — TTS Engine", lifespan=lifespan)


# ─────────────────────────────────────────────
# Request model
# ─────────────────────────────────────────────
class SpeechRequest(BaseModel):
    model: str = "tts-1"
    engine: str = "qwen"  # 'qwen' | 'xtts'
    input: str
    voice: Optional[str] = None  # Abs path to reference audio
    speaker_id: Optional[str] = None  # Built-in speaker name (Qwen only)
    ref_text: Optional[str] = None  # Optional transcript of ref audio (ICL mode)
    language: Optional[str] = None  # e.g. "english", "auto", "en", "hi"
    response_format: str = "wav"


# ─────────────────────────────────────────────
# Audio pre-processing
# ─────────────────────────────────────────────
def preprocess_reference_audio(audio_path: str) -> str:
    """Decode, VAD-trim, resample to 24 kHz mono, and trim to `MAX_REF_DURATION_SEC`."""
    print(f"  Pre-processing reference audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    original_duration = len(audio) / sr
    print(f"  Original duration: {original_duration:.1f}s, sample rate: {sr}")

    if nr is not None:
        denoised_audio = nr.reduce_noise(y=audio, sr=sr)
        audio = np.asarray(denoised_audio, dtype=np.float32)
        print("  Applied spectral noise reduction")
    else:
        print("  Skipped spectral noise reduction because noisereduce is unavailable")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = audio / peak
        print(f"  Peak-normalized audio (pre-normalization peak: {peak:.4f})")
    else:
        print("  Skipped peak normalization because audio peak was zero")

    trimmed_audio, _ = librosa.effects.trim(audio, top_db=30)
    if trimmed_audio.size > 0:
        trimmed_duration = len(trimmed_audio) / sr
        audio = np.asarray(trimmed_audio, dtype=np.float32)
        print(f"  Silence trimmed duration: {trimmed_duration:.1f}s")
    else:
        print("  Silence trimming found no voiced region; using original audio")

    max_samples = int(MAX_REF_DURATION_SEC * sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        print(f"  Trimmed to {MAX_REF_DURATION_SEC}s")

    temp_path = audio_path + ".preprocessed.wav"
    sf.write(temp_path, audio, sr, format="WAV")
    print(f"  Saved preprocessed audio to: {temp_path}")
    return temp_path


# ─────────────────────────────────────────────
# Token estimation (Qwen only)
# ─────────────────────────────────────────────
def _estimate_max_tokens(text: str) -> int:
    """Scale max_new_tokens to text length. Model runs at 12 Hz codec."""
    word_count = len(text.split())
    estimated_seconds = max(3.0, word_count / 2.8)
    max_tokens = int(estimated_seconds * 12 * 2)
    max_tokens = max(150, min(max_tokens, 800))
    print(
        f"  Words: {word_count}, est. speech: {estimated_seconds:.1f}s, max_new_tokens: {max_tokens}"
    )
    return max_tokens


# ─────────────────────────────────────────────
# Inference functions (sync — run in thread pool)
# ─────────────────────────────────────────────
def _run_voice_clone(
    text: str, temp_audio_path: str, ref_text: Optional[str], language: Optional[str]
):
    """Qwen voice clone inference using the Base model."""
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 4)
    max_tokens = _estimate_max_tokens(text)
    use_icl = ref_text is not None and ref_text.strip() != ""
    with torch.no_grad():
        wavs, sr = model_manager.get_qwen_clone().generate_voice_clone(
            text=text,
            language=language or "auto",
            ref_audio=temp_audio_path,
            ref_text=ref_text if use_icl else None,
            x_vector_only_mode=not use_icl,
            max_new_tokens=max_tokens,
        )
    return wavs, sr


def _run_custom_voice(text: str, speaker_id: str, language: Optional[str]):
    """Qwen built-in speaker inference using the CustomVoice model."""
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 4)
    max_tokens = _estimate_max_tokens(text)
    with torch.no_grad():
        wavs, sr = model_manager.get_qwen_speaker().generate_custom_voice(
            text=text,
            speaker=speaker_id,
            language=language or "auto",
            max_new_tokens=max_tokens,
        )
    return wavs, sr


def _normalize_xtts_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def _split_xtts_sentences(text: str) -> list[str]:
    normalized = _normalize_xtts_text(text)
    if not normalized:
        return []

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?।])\s+", normalized)
        if sentence.strip()
    ]
    return sentences or [normalized]


def _word_wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def _chunk_xtts_text(text: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    for sentence in _split_xtts_sentences(text):
        if len(sentence) > max_chars:
            chunks.extend(_word_wrap_text(sentence, max_chars))
        else:
            chunks.append(sentence)
    return chunks


def _get_xtts_max_chars(tts_api, language: str) -> int:
    default_limit = XTTS_MAX_CHARS_HI if language == "hi" else XTTS_MAX_CHARS_DEFAULT
    try:
        tts_model = tts_api.synthesizer.tts_model
        tokenizer = getattr(tts_model, "tokenizer", None)
        char_limits = getattr(tokenizer, "char_limits", None) or {}
        language_limit = char_limits.get(language)
        if isinstance(language_limit, int) and language_limit > XTTS_CHUNK_MARGIN_CHARS:
            return min(default_limit, language_limit - XTTS_CHUNK_MARGIN_CHARS)
    except Exception:
        pass
    return default_limit


def _run_xtts(text: str, ref_audio_path: str, language: str) -> tuple[np.ndarray, int]:
    """XTTSv2 voice clone inference. Returns (wav_array, 24000)."""
    lang = (language or "en").lower()
    if lang not in XTTS_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Language '{lang}' is not supported by XTTSv2. "
            f"Supported: {sorted(XTTS_SUPPORTED_LANGUAGES)}"
        )

    tts_api = model_manager.get_xtts()
    max_chars = _get_xtts_max_chars(tts_api, lang)
    sentences = _split_xtts_sentences(text)
    if not sentences:
        raise ValueError("Input text became empty after XTTS normalization.")

    print(
        f"[xtts] Sentence chunking enabled — {len(sentences)} sentence(s), max_chars={max_chars}"
    )

    audio_segments: list[np.ndarray] = []
    join_silence = np.zeros(int(24000 * XTTS_JOIN_SILENCE_MS / 1000), dtype=np.float32)

    with torch.no_grad():
        for sentence_index, sentence in enumerate(sentences, start=1):
            sentence_chunks = _chunk_xtts_text(sentence, max_chars=max_chars)
            for chunk_index, chunk in enumerate(sentence_chunks, start=1):
                print(
                    f"[xtts] Sentence {sentence_index}/{len(sentences)}"
                    f" chunk {chunk_index}/{len(sentence_chunks)}: {chunk}"
                )
                wav = tts_api.tts(
                    text=chunk,
                    speaker_wav=ref_audio_path,
                    language=lang,
                    split_sentences=False,
                    enable_text_splitting=False,
                    speed=XTTS_TARGET_SPEED,
                    temperature=XTTS_TEMPERATURE,
                    top_k=XTTS_TOP_K,
                    top_p=XTTS_TOP_P,
                    repetition_penalty=XTTS_REPETITION_PENALTY,
                )
                chunk_audio = np.asarray(wav, dtype=np.float32)
                if chunk_audio.size == 0:
                    continue
                if audio_segments:
                    audio_segments.append(join_silence)
                audio_segments.append(chunk_audio)

    if not audio_segments:
        raise RuntimeError("XTTSv2 produced no audio for the requested text.")

    combined = np.concatenate(audio_segments).astype(np.float32)
    return combined, 24000


# ─────────────────────────────────────────────
# Main endpoint
# ─────────────────────────────────────────────
@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    engine = (request.engine or "qwen").lower()
    text = request.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'input' must not be empty.")

    language = request.language

    # ── XTTSv2 engine ─────────────────────────────────────────────────────
    if engine == "xtts":
        ref_path = request.voice
        if not ref_path or not os.path.exists(ref_path):
            raise HTTPException(
                status_code=400,
                detail="Reference audio file not found. XTTSv2 requires a reference audio file.",
            )

        lang = (language or "en").lower()
        print(f"[xtts] Generating — language: {lang}")
        print(f'  Text: "{text[:80]}{"…" if len(text) > 80 else ""}"')
        print(f"  Reference: {ref_path}")

        temp_audio_path = None
        try:
            temp_audio_path = preprocess_reference_audio(ref_path)
            loop = asyncio.get_event_loop()
            wav_array, sr = await loop.run_in_executor(
                None, _run_xtts, text, temp_audio_path, lang
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except (sf.LibsndfileError, audioread.exceptions.DecodeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Reference audio could not be decoded. Please upload or record a valid audio file: {str(e)}",
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            print("XTTSv2 generation failed:", traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Voice generation failed: {str(e)}"
            )
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass

        buffer = io.BytesIO()
        sf.write(buffer, wav_array, sr, format="WAV")
        buffer.seek(0)
        print("[xtts] Audio generated successfully.")
        return Response(content=buffer.read(), media_type="audio/wav")

    # ── Qwen engine ───────────────────────────────────────────────────────

    # Built-in speaker mode
    if request.speaker_id:
        speaker_id = request.speaker_id.strip().lower()
        try:
            supported = model_manager.get_qwen_speaker().get_supported_speakers() or []
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        supported_lower = [s.lower() for s in supported]
        if speaker_id not in supported_lower:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown speaker '{speaker_id}'. Supported: {supported}",
            )

        print(
            f"[qwen] Generating — built-in speaker: {speaker_id}, language: {language}"
        )
        print(f'  Text: "{text}"')
        try:
            loop = asyncio.get_event_loop()
            wavs, sr = await loop.run_in_executor(
                None, _run_custom_voice, text, speaker_id, language
            )
        except Exception as e:
            print("Generation failed:", traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Voice generation failed: {str(e)}"
            )

    # Voice clone mode
    else:
        ref_audio_path = request.voice
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise HTTPException(
                status_code=400,
                detail="Reference audio file not found. Provide 'voice' (file path) or 'speaker_id'.",
            )

        temp_audio_path = None
        try:
            temp_audio_path = preprocess_reference_audio(ref_audio_path)
            print(f"[qwen] Generating — voice clone, language: {language}")
            print(f'  Text: "{text}"')
            loop = asyncio.get_event_loop()
            wavs, sr = await loop.run_in_executor(
                None,
                _run_voice_clone,
                text,
                temp_audio_path,
                request.ref_text,
                language,
            )
        except Exception as e:
            print("Generation failed:", traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Voice cloning failed: {str(e)}"
            )
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass

    # ── Encode and return WAV ──────────────────────────────────────────────
    output_wav = wavs[0]
    buffer = io.BytesIO()
    sf.write(buffer, output_wav, sr, format="WAV")
    buffer.seek(0)
    print(f"[qwen] Audio generated successfully.")
    return Response(content=buffer.read(), media_type="audio/wav")


# ─────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────
def _vram_info():
    if torch.cuda.is_available():
        return {
            "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        }
    return {}


@app.get("/health")
async def health():
    vram_info = _vram_info()
    return {
        "status": "ok",
        "device": device,
        "active_model_group": model_manager._active_group,
        **vram_info,
    }


@app.post("/admin/unload-models")
async def unload_models():
    model_manager.unload_all()
    return {
        "status": "ok",
        "message": "All TTS models unloaded from memory.",
        "active_model_group": model_manager._active_group,
        **_vram_info(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
