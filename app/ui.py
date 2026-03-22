from __future__ import annotations

import gradio as gr

from .asr import WhisperService
from .audio_utils import format_audio_report, prepare_reference_audio
from .config import AppConfig
from .qwen_service import QwenTTSService


MODEL_CHOICES = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
]

ASR_LANGUAGE_CHOICES = [
    "Auto",
    "Japanese",
    "English",
    "Chinese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

TTS_LANGUAGE_CHOICES = [
    "Japanese",
    "English",
    "Chinese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]


def build_demo(config: AppConfig) -> gr.Blocks:
    qwen_service = QwenTTSService(config)
    whisper_service = WhisperService(config)

    def reset_reference_state(audio_path: str | None) -> tuple[str, tuple[int, object] | None, str, str, str, None, str, str]:
        if not audio_path:
            message = "参照音声が未入力です。"
        else:
            message = "参照音声が変更されました。参照テキストと診断結果を更新するには、再度 Whisper を実行してください。"
        return "", None, "", message, "", None, "", ""

    def bind_ref_text_to_audio(audio_path: str | None, ref_text: str) -> str:
        if not audio_path or not ref_text.strip():
            return ""

        try:
            prepared = prepare_reference_audio(audio_path)
        except Exception:
            return ""
        return prepared.cache_key

    def clear_generation_state() -> tuple[None, str, str]:
        return None, "", ""

    def transcribe_reference(
        audio_path: str | None,
        language: str,
    ) -> tuple[str, tuple[int, object] | None, str, str, str, None, str, str]:
        if not audio_path:
            return "", None, "参照音声をアップロードするか、マイクで録音してください。", "参照音声が未入力です。", "", None, "", ""

        try:
            if config.unload_tts_before_asr:
                qwen_service.release()

            prepared = prepare_reference_audio(audio_path)
            asr_result = whisper_service.transcribe(prepared, language=language)
            report = format_audio_report(prepared, transcript_language=asr_result.detected_language)
            status = (
                f"Whisper backend: `{asr_result.backend}` / "
                "文字起こし結果は必ず確認してから合成してください。"
            )
            if whisper_service.load_note:
                status = f"{status} / {whisper_service.load_note}"
            return asr_result.text, prepared.as_gradio_audio(), report, status, prepared.cache_key, None, "", ""
        except Exception as exc:
            return (
                "",
                None,
                "参照音声の解析または文字起こしに失敗しました。",
                f"{type(exc).__name__}: {exc}",
                "",
                None,
                "",
                "",
            )

    def synthesize(
        audio_path: str | None,
        ref_text: str,
        target_text: str,
        ref_audio_key: str,
        model_id: str,
        synthesis_language: str,
        x_vector_only_mode: bool,
        use_prompt_cache: bool,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[tuple[int, object] | None, str, str]:
        if not audio_path:
            return None, "", "参照音声が未入力です。"

        if not target_text or not target_text.strip():
            return None, "", "読み上げる本文を入力してください。"

        if not x_vector_only_mode and not ref_text.strip():
            return None, "", (
                "ICL モードでは参照テキストが必須です。"
                " Whisper で文字起こししてから、必要なら修正して使ってください。"
            )

        try:
            prepared = prepare_reference_audio(audio_path)
            if ref_audio_key and ref_audio_key != prepared.cache_key:
                return None, "", (
                    "### 合成エラー\n"
                    "- 参照音声が変更されました。Whisper を再実行するか、参照テキストを現在の音声に合わせて入れ直してください。"
                )
            output_path, output_audio, summary = qwen_service.generate(
                model_id=model_id,
                prepared=prepared,
                ref_text=ref_text,
                target_text=target_text,
                language=synthesis_language,
                x_vector_only_mode=x_vector_only_mode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_prompt_cache=use_prompt_cache,
            )
            return output_audio, output_path, summary
        except Exception as exc:
            return None, "", f"### 合成エラー\n- {type(exc).__name__}: {exc}"

    def clear_runtime_cache() -> tuple[str, None, str, str]:
        whisper_service.release()
        qwen_service.release()
        qwen_service.clear_prompt_cache()
        return (
            "ASR/TTS モデルを解放し、voice clone prompt キャッシュもクリアしました。次回実行時は再ロードが走ります。",
            None,
            "",
            "",
        )

    with gr.Blocks(title="Qwen3-TTS Voice Clone Studio") as demo:
        ref_audio_key_state = gr.State("")

        gr.Markdown(
            """
# Qwen3-TTS Voice Clone Studio

Windows + Docker Compose + NVIDIA GPU 前提で使う、日本語向けの Qwen3-TTS ボイスクローン GUI です。  
`Qwen/Qwen3-TTS-12Hz-1.7B-Base` を既定にし、参照音声は Whisper で文字起こししてから使う流れにしています。
"""
        )
        gr.Markdown(
            """
**重要メモ**
- 参照音声は 3〜5 秒、単一話者、BGM なし、長い無音なしが有利です。
- `x_vector_only_mode` は参照テキストなしでも動きますが、公式に品質低下が明記されています。
- `localhost` で使うぶんには通常の HTTP でもマイク入力できます。LAN や別端末から開くなら HTTPS を推奨します。
"""
        )

        with gr.Row():
            with gr.Column(scale=6):
                reference_audio = gr.Audio(
                    label="1. 参照音声",
                    sources=["upload", "microphone"],
                    type="filepath",
                )
                reference_preview = gr.Audio(label="前処理後プレビュー", interactive=False)
                transcribe_button = gr.Button("参照音声を解析して Whisper で文字起こし")
                transcription_language = gr.Dropdown(
                    label="Whisper 文字起こし言語ヒント",
                    choices=ASR_LANGUAGE_CHOICES,
                    value="Japanese",
                )
                ref_text = gr.Textbox(
                    label="2. 参照テキスト",
                    lines=4,
                    placeholder="Whisper 文字起こし後、必ず内容を確認して修正してください。",
                )
                audio_report = gr.Markdown()
                transcription_status = gr.Textbox(label="文字起こしステータス", interactive=False)

            with gr.Column(scale=6):
                target_text = gr.Textbox(
                    label="3. 読み上げたい本文",
                    lines=8,
                    placeholder="ここに日本語テキストを入力します。",
                )
                synth_button = gr.Button("4. Qwen3-TTS で音声合成", variant="primary")
                generated_audio = gr.Audio(label="合成音声", interactive=False)
                saved_path = gr.Textbox(label="保存先", interactive=False)
                generation_report = gr.Markdown()

        with gr.Accordion("詳細設定", open=False):
            with gr.Row():
                model_id = gr.Dropdown(
                    label="Qwen3-TTS モデル",
                    choices=MODEL_CHOICES,
                    value=config.qwen_model_id,
                )
                synthesis_language = gr.Dropdown(
                    label="合成言語",
                    choices=TTS_LANGUAGE_CHOICES,
                    value="Japanese",
                )

            with gr.Row():
                x_vector_only_mode = gr.Checkbox(
                    label="x_vector_only_mode を使う",
                    value=False,
                    info="参照テキストなしでも動きますが、品質は下がりやすいです。",
                )
                use_prompt_cache = gr.Checkbox(
                    label="参照 prompt をキャッシュ再利用する",
                    value=True,
                    info="同じ参照音声を何度も使うときに高速化できます。",
                )

            with gr.Row():
                max_new_tokens = gr.Slider(
                    label="max_new_tokens",
                    minimum=256,
                    maximum=4096,
                    value=config.qwen_max_new_tokens,
                    step=64,
                )
                temperature = gr.Slider(
                    label="temperature",
                    minimum=0.1,
                    maximum=1.5,
                    value=config.qwen_temperature,
                    step=0.05,
                )

            with gr.Row():
                top_k = gr.Slider(
                    label="top_k",
                    minimum=1,
                    maximum=100,
                    value=config.qwen_top_k,
                    step=1,
                )
                top_p = gr.Slider(
                    label="top_p",
                    minimum=0.1,
                    maximum=1.0,
                    value=config.qwen_top_p,
                    step=0.05,
                )
                repetition_penalty = gr.Slider(
                    label="repetition_penalty",
                    minimum=1.0,
                    maximum=1.4,
                    value=config.qwen_repetition_penalty,
                    step=0.01,
                )

        with gr.Accordion("品質を上げるコツ", open=False):
            gr.Markdown(
                """
- 参照音声は 3〜5 秒くらいに切り出す
- 16-bit mono WAV が安定しやすい
- 参照テキストは実際の音声内容と一致させる
- 単一話者、BGM なし、重なり話者なし
- 長い無音、音割れ、強いノイズは避ける
"""
            )

        with gr.Row():
            clear_cache_button = gr.Button("モデルとキャッシュを解放")
            clear_cache_status = gr.Textbox(label="解放ステータス", interactive=False)

        reference_audio.change(
            fn=reset_reference_state,
            inputs=[reference_audio],
            outputs=[
                ref_text,
                reference_preview,
                audio_report,
                transcription_status,
                ref_audio_key_state,
                generated_audio,
                saved_path,
                generation_report,
            ],
        )
        ref_text.change(
            fn=bind_ref_text_to_audio,
            inputs=[reference_audio, ref_text],
            outputs=[ref_audio_key_state],
        )
        ref_text.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        target_text.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        model_id.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        synthesis_language.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        x_vector_only_mode.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        use_prompt_cache.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        max_new_tokens.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        temperature.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        top_k.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        top_p.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        repetition_penalty.change(
            fn=clear_generation_state,
            inputs=[],
            outputs=[generated_audio, saved_path, generation_report],
        )
        transcribe_button.click(
            fn=transcribe_reference,
            inputs=[reference_audio, transcription_language],
            outputs=[
                ref_text,
                reference_preview,
                audio_report,
                transcription_status,
                ref_audio_key_state,
                generated_audio,
                saved_path,
                generation_report,
            ],
        )
        synth_button.click(
            fn=synthesize,
            inputs=[
                reference_audio,
                ref_text,
                target_text,
                ref_audio_key_state,
                model_id,
                synthesis_language,
                x_vector_only_mode,
                use_prompt_cache,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            ],
            outputs=[generated_audio, saved_path, generation_report],
        )
        clear_cache_button.click(
            fn=clear_runtime_cache,
            inputs=[],
            outputs=[clear_cache_status, generated_audio, saved_path, generation_report],
        )

    demo.queue(default_concurrency_limit=1)
    return demo
