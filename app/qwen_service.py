from __future__ import annotations

import gc
import re
import threading
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from .audio_utils import PreparedAudio
from .config import AppConfig


def _torch_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.strip().lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _safe_filename(stem: str) -> str:
    compact = re.sub(r"\s+", "_", stem.strip())
    compact = re.sub(r"[^0-9A-Za-z_\-ぁ-んァ-ヶ一-龠]", "", compact)
    return compact[:40] or "qwen3_tts"


class QwenTTSService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model: Qwen3TTSModel | None = None
        self._loaded_model_id: str | None = None
        self._load_note = ""
        self._lock = threading.RLock()
        self._prompt_cache: dict[str, object] = {}
        self._prompt_cache_limit = 8

    @property
    def load_note(self) -> str:
        return self._load_note

    def ensure_loaded(self, model_id: str) -> None:
        with self._lock:
            if self._model is not None and self._loaded_model_id == model_id:
                return

            self.release()
            dtype = _torch_dtype(self.config.qwen_dtype)
            attn_impl = self.config.qwen_attn_implementation or None

            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.config.qwen_device,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )
                self._load_note = f"モデルを `{model_id}` として読み込みました。attention: `{attn_impl or 'default'}`"
            except Exception as exc:
                if attn_impl is None:
                    raise
                self._model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=self.config.qwen_device,
                    dtype=dtype,
                )
                self._load_note = (
                    f"`{attn_impl}` でのロードに失敗したため標準attentionへフォールバックしました。"
                    f" 理由: {type(exc).__name__}: {exc}"
                )

            self._loaded_model_id = model_id

    def _prompt_cache_key(
        self,
        model_id: str,
        prepared: PreparedAudio,
        ref_text: str,
        x_vector_only_mode: bool,
    ) -> str:
        return "|".join(
            [
                model_id,
                prepared.cache_key,
                ref_text.strip(),
                "xvec" if x_vector_only_mode else "icl",
            ]
        )

    def create_or_get_prompt(
        self,
        model_id: str,
        prepared: PreparedAudio,
        ref_text: str,
        x_vector_only_mode: bool,
    ) -> tuple[object, bool]:
        with self._lock:
            self.ensure_loaded(model_id)
            assert self._model is not None

            cache_key = self._prompt_cache_key(model_id, prepared, ref_text, x_vector_only_mode)
            if cache_key in self._prompt_cache:
                return self._prompt_cache[cache_key], True

            prompt = self._model.create_voice_clone_prompt(
                ref_audio=prepared.as_qwen_audio(),
                ref_text=ref_text.strip() if ref_text.strip() else None,
                x_vector_only_mode=x_vector_only_mode,
            )
            self._prompt_cache[cache_key] = prompt
            while len(self._prompt_cache) > self._prompt_cache_limit:
                oldest_key = next(iter(self._prompt_cache))
                self._prompt_cache.pop(oldest_key, None)
            return prompt, False

    def generate(
        self,
        *,
        model_id: str,
        prepared: PreparedAudio,
        ref_text: str,
        target_text: str,
        language: str,
        x_vector_only_mode: bool,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        use_prompt_cache: bool,
    ) -> tuple[str, tuple[int, object], str]:
        with self._lock:
            self.ensure_loaded(model_id)
            assert self._model is not None

            if use_prompt_cache:
                voice_clone_prompt, cache_hit = self.create_or_get_prompt(
                    model_id=model_id,
                    prepared=prepared,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
                ref_audio = None
                ref_text_value = None
                note = (
                    "既存キャッシュを再利用しました。"
                    if cache_hit
                    else "新しく voice clone prompt を作成してキャッシュしました。"
                )
            else:
                voice_clone_prompt = None
                ref_audio = prepared.as_qwen_audio()
                ref_text_value = ref_text.strip() if ref_text.strip() else None
                note = "毎回参照音声から prompt を再計算しました。"

            wavs, sample_rate = self._model.generate_voice_clone(
                text=target_text.strip(),
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text_value,
                x_vector_only_mode=x_vector_only_mode,
                voice_clone_prompt=voice_clone_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                subtalker_dosample=True,
                subtalker_top_k=self.config.qwen_subtalker_top_k,
                subtalker_top_p=self.config.qwen_subtalker_top_p,
                subtalker_temperature=self.config.qwen_subtalker_temperature,
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_name = f"{timestamp}_{_safe_filename(target_text[:24])}.wav"
        output_path = output_dir / output_name
        sf.write(output_path, wavs[0], sample_rate)

        summary = "\n".join(
            [
                "### 合成結果",
                f"- 保存先: `{output_path}`",
                f"- モデル: `{model_id}`",
                f"- 言語: `{language}`",
                f"- `x_vector_only_mode`: `{x_vector_only_mode}`",
                f"- Promptキャッシュ: {'有効' if use_prompt_cache else '無効'}",
                f"- メモ: {note}",
                f"- ロード情報: {self.load_note}",
            ]
        )

        return str(output_path), (sample_rate, wavs[0]), summary

    def clear_prompt_cache(self) -> str:
        with self._lock:
            self._prompt_cache.clear()
            return "voice clone prompt のキャッシュをクリアしました。"

    def release(self, clear_prompt_cache: bool = True) -> None:
        with self._lock:
            self._model = None
            self._loaded_model_id = None
            if clear_prompt_cache:
                self._prompt_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
