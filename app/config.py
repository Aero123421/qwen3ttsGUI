from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    qwen_model_id: str = field(default_factory=lambda: os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    qwen_device: str = field(default_factory=lambda: os.getenv("QWEN_DEVICE", "cuda:0"))
    qwen_dtype: str = field(default_factory=lambda: os.getenv("QWEN_DTYPE", "bfloat16"))
    qwen_attn_implementation: str | None = field(
        default_factory=lambda: os.getenv("QWEN_ATTN_IMPLEMENTATION", "flash_attention_2")
    )

    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cuda"))
    whisper_compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16"))

    server_name: str = field(default_factory=lambda: os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
    server_port: int = field(default_factory=lambda: int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    ssl_certfile: str | None = field(default_factory=lambda: os.getenv("GRADIO_SSL_CERTFILE") or None)
    ssl_keyfile: str | None = field(default_factory=lambda: os.getenv("GRADIO_SSL_KEYFILE") or None)

    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "/data/outputs"))

    qwen_max_new_tokens: int = field(default_factory=lambda: int(os.getenv("QWEN_MAX_NEW_TOKENS", "2048")))
    qwen_temperature: float = field(default_factory=lambda: float(os.getenv("QWEN_TEMPERATURE", "0.9")))
    qwen_top_k: int = field(default_factory=lambda: int(os.getenv("QWEN_TOP_K", "50")))
    qwen_top_p: float = field(default_factory=lambda: float(os.getenv("QWEN_TOP_P", "1.0")))
    qwen_repetition_penalty: float = field(
        default_factory=lambda: float(os.getenv("QWEN_REPETITION_PENALTY", "1.05"))
    )
    qwen_subtalker_top_k: int = field(default_factory=lambda: int(os.getenv("QWEN_SUBTALKER_TOP_K", "50")))
    qwen_subtalker_top_p: float = field(default_factory=lambda: float(os.getenv("QWEN_SUBTALKER_TOP_P", "1.0")))
    qwen_subtalker_temperature: float = field(
        default_factory=lambda: float(os.getenv("QWEN_SUBTALKER_TEMPERATURE", "0.9"))
    )

    unload_tts_before_asr: bool = field(default_factory=lambda: _get_env_bool("UNLOAD_TTS_BEFORE_ASR", True))
    unload_asr_after_transcribe: bool = field(
        default_factory=lambda: _get_env_bool("UNLOAD_ASR_AFTER_TRANSCRIBE", True)
    )
