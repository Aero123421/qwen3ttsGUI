from __future__ import annotations

import gc
import tempfile
import threading
from dataclasses import dataclass

import soundfile as sf
import torch

from .audio_utils import PreparedAudio
from .config import AppConfig


WHISPER_LANGUAGE_MAP = {
    "Auto": None,
    "Japanese": "ja",
    "English": "en",
    "Chinese": "zh",
    "Korean": "ko",
    "German": "de",
    "French": "fr",
    "Russian": "ru",
    "Portuguese": "pt",
    "Spanish": "es",
    "Italian": "it",
}


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    detected_language: str | None
    backend: str


class WhisperService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._backend = ""
        self._loaded_model_name = ""
        self._load_note = ""
        self._lock = threading.RLock()

    @property
    def load_note(self) -> str:
        return self._load_note

    def _load_faster_whisper(self, model_name: str) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_name,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type,
        )
        self._backend = "faster-whisper"
        self._loaded_model_name = model_name

    def _load_openai_whisper(self, model_name: str) -> None:
        import whisper

        fallback_name = "large-v3" if model_name.startswith("distil-") else model_name
        device = "cuda" if self.config.whisper_device.startswith("cuda") else "cpu"
        self._model = whisper.load_model(fallback_name, device=device)
        self._backend = "openai-whisper"
        self._loaded_model_name = fallback_name

    def ensure_loaded(self, model_name: str | None = None) -> None:
        with self._lock:
            target_model = model_name or self.config.whisper_model
            if self._model is not None and self._loaded_model_name == target_model:
                return

            self.release()
            try:
                self._load_faster_whisper(target_model)
                self._load_note = f"`{target_model}` を faster-whisper で読み込みました。"
            except Exception as exc:
                self._load_openai_whisper(target_model)
                self._load_note = (
                    "faster-whisper の読み込みに失敗したため openai-whisper へフォールバックしました。"
                    f" 理由: {type(exc).__name__}: {exc}"
                )

    def transcribe(self, prepared: PreparedAudio, language: str) -> TranscriptionResult:
        with self._lock:
            self.ensure_loaded()

            language_code = WHISPER_LANGUAGE_MAP.get(language, None)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
                sf.write(temp_audio.name, prepared.waveform, prepared.sample_rate)

                if self._backend == "faster-whisper":
                    segments, info = self._model.transcribe(
                        temp_audio.name,
                        language=language_code,
                        beam_size=5,
                        vad_filter=True,
                        condition_on_previous_text=False,
                    )
                    text = " ".join(segment.text.strip() for segment in segments).strip()
                    result = TranscriptionResult(
                        text=text,
                        detected_language=getattr(info, "language", None),
                        backend=self._backend,
                    )
                else:
                    result_raw = self._model.transcribe(
                        temp_audio.name,
                        language=language_code,
                        fp16=self.config.whisper_device.startswith("cuda"),
                    )
                    result = TranscriptionResult(
                        text=result_raw.get("text", "").strip(),
                        detected_language=result_raw.get("language"),
                        backend=self._backend,
                    )

            if self.config.unload_asr_after_transcribe:
                self.release()
            return result

    def release(self) -> None:
        with self._lock:
            self._model = None
            self._loaded_model_name = ""
            self._backend = ""
            self._load_note = ""
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
