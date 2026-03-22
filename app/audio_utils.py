from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import audioread
import librosa
import numpy as np
import soundfile as sf


MAX_REFERENCE_DURATION_SEC = 30.0


@dataclass(slots=True)
class PreparedAudio:
    waveform: np.ndarray
    sample_rate: int
    original_duration_sec: float
    trimmed_duration_sec: float
    leading_silence_sec: float
    trailing_silence_sec: float
    peak: float
    rms: float
    clipped: bool
    cache_key: str
    warnings: list[str]

    def as_qwen_audio(self) -> tuple[np.ndarray, int]:
        return self.waveform.astype(np.float32), int(self.sample_rate)

    def as_gradio_audio(self) -> tuple[int, np.ndarray]:
        return int(self.sample_rate), self.waveform.astype(np.float32)


def _dbfs(value: float) -> float:
    return -120.0 if value <= 0 else 20.0 * math.log10(value)


def _probe_duration_sec(audio_path: str | Path) -> float | None:
    try:
        info = sf.info(str(audio_path))
        if info.samplerate > 0 and info.frames > 0:
            return float(info.frames / info.samplerate)
    except Exception:
        pass

    try:
        with audioread.audio_open(str(audio_path)) as input_file:
            return float(input_file.duration)
    except Exception:
        return None


def prepare_reference_audio(audio_path: str | Path) -> PreparedAudio:
    probed_duration = _probe_duration_sec(audio_path)
    if probed_duration is not None and probed_duration > MAX_REFERENCE_DURATION_SEC:
        raise ValueError("参照音声が長すぎます。30 秒以下、できれば 3〜5 秒に切り出してください。")

    waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    waveform = np.clip(waveform.astype(np.float32), -1.0, 1.0)

    if sample_rate <= 0 or waveform.size == 0:
        raise ValueError("参照音声を読み取れませんでした。3〜5 秒ほどの音声ファイルを指定してください。")

    original_duration = float(len(waveform) / sample_rate) if sample_rate else 0.0
    if original_duration > MAX_REFERENCE_DURATION_SEC:
        raise ValueError("参照音声が長すぎます。30 秒以下、できれば 3〜5 秒に切り出してください。")

    trimmed_waveform, trim_index = librosa.effects.trim(waveform, top_db=30)
    if trimmed_waveform.size == 0:
        raise ValueError("発話区間を検出できませんでした。無音ではない参照音声を指定してください。")
    trimmed_waveform = np.clip(trimmed_waveform.astype(np.float32), -1.0, 1.0)

    leading_silence = float(trim_index[0] / sample_rate) if sample_rate else 0.0
    trailing_silence = float((len(waveform) - trim_index[1]) / sample_rate) if sample_rate else 0.0
    trimmed_duration = float(len(trimmed_waveform) / sample_rate) if sample_rate else 0.0

    peak = float(np.max(np.abs(trimmed_waveform))) if trimmed_waveform.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(trimmed_waveform)))) if trimmed_waveform.size else 0.0
    clipped = peak >= 0.999

    warnings: list[str] = []
    if trimmed_duration < 3.0:
        warnings.append("参照音声が短すぎます。3秒以上を推奨します。")
    elif trimmed_duration > 10.0:
        warnings.append("参照音声が長めです。3〜10秒程度に切り出すと安定しやすいです。")

    if leading_silence > 0.35 or trailing_silence > 0.35:
        warnings.append("冒頭または末尾に長めの無音があります。短く整えると精度が上がりやすいです。")

    if clipped:
        warnings.append("音割れの可能性があります。ピークが強すぎる参照音声は品質を落とします。")

    if _dbfs(rms) < -35:
        warnings.append("音量がかなり小さいです。マイクを近づけた録音のほうが有利です。")

    if _dbfs(rms) < -50 or peak < 0.01:
        raise ValueError("参照音声がほぼ無音です。声が十分に入った 3〜5 秒の音声を使ってください。")

    digest = hashlib.sha256()
    digest.update(trimmed_waveform.tobytes())
    digest.update(str(sample_rate).encode("utf-8"))

    return PreparedAudio(
        waveform=trimmed_waveform,
        sample_rate=sample_rate,
        original_duration_sec=original_duration,
        trimmed_duration_sec=trimmed_duration,
        leading_silence_sec=leading_silence,
        trailing_silence_sec=trailing_silence,
        peak=peak,
        rms=rms,
        clipped=clipped,
        cache_key=digest.hexdigest(),
        warnings=warnings,
    )


def format_audio_report(prepared: PreparedAudio, transcript_language: str | None = None) -> str:
    lines = [
        "### 参照音声チェック",
        f"- 元の長さ: {prepared.original_duration_sec:.2f} 秒",
        f"- トリム後の長さ: {prepared.trimmed_duration_sec:.2f} 秒",
        f"- 先頭無音: {prepared.leading_silence_sec:.2f} 秒",
        f"- 末尾無音: {prepared.trailing_silence_sec:.2f} 秒",
        f"- サンプルレート: {prepared.sample_rate} Hz",
        f"- ピーク: {_dbfs(prepared.peak):.1f} dBFS",
        f"- RMS: {_dbfs(prepared.rms):.1f} dBFS",
    ]

    if transcript_language:
        lines.append(f"- Whisper推定言語: `{transcript_language}`")

    if prepared.warnings:
        lines.append("")
        lines.append("### 警告")
        lines.extend(f"- {warning}" for warning in prepared.warnings)
    else:
        lines.append("")
        lines.append("- 推奨条件におおむね近い参照音声です。")

    lines.append("")
    lines.append("### 推奨条件")
    lines.append("- 3〜5秒を目安に、単一話者・BGMなし・静かな部屋で録音")
    lines.append("- 文字起こしは必ず確認し、音声内容と一致させる")
    lines.append("- 中立寄りの自然な話し方だと汎用クローンが安定しやすい")
    return "\n".join(lines)
