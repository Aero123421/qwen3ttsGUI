# Qwen3-TTS Voice Clone Studio

Windows PC 上で `Qwen3-TTS` を Docker Compose 経由で簡単に起動するための、Gradio ベース日本語 GUI です。  
主目的は `Qwen/Qwen3-TTS-12Hz-1.7B-Base` を使った voice clone で、参照音声の `ref_text` は Whisper で補助的に文字起こしできるようにしています。

## この構成にした理由

- Qwen3-TTS 公式では、voice clone は `Base` モデルで `ref_audio + ref_text` を与える形です。
- `x_vector_only_mode=True` なら `ref_text` なしでも使えますが、公式に品質低下が明記されています。
- そのため、本アプリでは「参照音声を入れる -> Whisper で仮文字起こし -> 人が確認して修正 -> Qwen3-TTS で合成」という流れを標準にしています。
- `1.7B` は `0.6B` より品質面で有利ですが、VRAM を食いやすいので、Whisper 側は都度解放する実装にして同時常駐を避けています。

## 主な機能

- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` / `0.6B-Base` 切り替え
- 参照音声のアップロードまたはブラウザマイク録音
- Whisper による参照音声の文字起こし
- 参照音声を差し替えたときの stale state 自動クリア
- 参照音声の前処理と診断
  - mono 化
  - 無音トリム
  - 長さチェック
  - クリッピング警告
  - 音量チェック
- `create_voice_clone_prompt` ベースの prompt キャッシュ再利用
- Docker Compose + `start.bat` / `stop.bat`

## 事前条件

### Windows / Docker / GPU

- Windows 10/11
- Docker Desktop
- Docker Desktop の `WSL 2 backend` を有効化
- NVIDIA GPU
- WSL2 GPU-PV 対応ドライバ

Docker Desktop の GPU サポートは Windows では WSL2 backend 前提です。  
Compose の GPU 予約では `capabilities: [gpu]` が必須です。

### GPU の選び方

- `.env` の `NVIDIA_VISIBLE_DEVICES=0` は「ホスト側の GPU 0 をコンテナへ見せる」という意味です
- 2 枚以上ある場合は `1` や `2` に変えてください
- コンテナ内では見えている GPU が再番号付けされるので、通常は `QWEN_DEVICE=cuda:0` のままで問題ありません

### 推奨 GPU メモリ

- `1.7B Base` を使うなら 12GB クラス以上を推奨
- ただし ASR を同時常駐させると厳しくなることがあるため、このアプリでは Whisper を使い終わったら解放します
- 余裕を見たいなら 16GB 以上が安心です

## 使い方

### 1. 起動

```bat
start.bat
```

初回は以下を行うので時間がかかります。

- Docker イメージのビルド
- Python パッケージのインストール
- Qwen3-TTS / Whisper モデルのダウンロード

`start.bat` は `docker compose up --build -d` を実行するので、起動用の黒い画面を閉じてもコンテナは動き続けます。
成功時はブラウザを自動で開きます。失敗時はエラー表示のまま止まるので、その内容を確認してください。

起動後はブラウザで以下を開きます。

```text
http://localhost:7860
```

`GRADIO_SERVER_PORT` を変えた場合は、そのポート番号で開いてください。  
Compose 側も同じ番号で公開されるようにしてあります。

### 2. 停止

```bat
stop.bat
```

## `.env` で調整できる主な項目

`.env` が存在しない場合、`start.bat` が `.env.example` から自動生成します。

```env
QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
QWEN_DEVICE=cuda:0
QWEN_DTYPE=float16
QWEN_ATTN_IMPLEMENTATION=flash_attention_2
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float16
NVIDIA_VISIBLE_DEVICES=0
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
GRADIO_SSL_CERTFILE=
GRADIO_SSL_KEYFILE=
UNLOAD_TTS_BEFORE_ASR=true
UNLOAD_ASR_AFTER_TRANSCRIBE=true
OUTPUT_DIR=/data/outputs
```

### VRAM が厳しいときの実用設定

- 古めの NVIDIA GPU では `QWEN_DTYPE=float16` を推奨します
- 12GB 未満で厳しい場合は `QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base` を試してください
- 文字起こしが重い場合は `WHISPER_MODEL=medium` か `WHISPER_MODEL=distil-large-v3` へ落とすと楽になります

## HTTPS について

- `localhost` で使うだけなら通常は HTTP でもマイク入力できます
- LAN 内の別 PC やスマホから開く場合は HTTPS を推奨します
- `GRADIO_SSL_CERTFILE` / `GRADIO_SSL_KEYFILE` に証明書パスを入れると HTTPS で起動できます

証明書は `./certs` をコンテナにマウントしているので、そこへ置く想定です。
`.env` にはホストの Windows パスではなく、コンテナ内パスを書いてください。

```env
GRADIO_SSL_CERTFILE=/data/certs/server.crt
GRADIO_SSL_KEYFILE=/data/certs/server.key
```

片方だけ設定すると起動時にエラーへします。

## 保存先とキャッシュ

- 合成した WAV は作業フォルダの `outputs/` に保存されます
- UI のダウンロードボタンを押すと、任意の保存先へ別途ダウンロードできます
- UI に出る `/data/outputs/...` はコンテナ内パスですが、実体はこの `outputs/` フォルダです
- Hugging Face のモデルキャッシュは Docker volume `hf-cache` に保持されます
- 初回は数十 GB 単位で Docker Desktop のディスクを使うことがあるので、容量に余裕を持たせてください

## 参照音声のベストプラクティス

調査結果を実装方針に反映し、UI にもそのまま出しています。

- 3〜5 秒を目安にする
- 16-bit mono WAV が安定しやすい
- 参照テキストは実際の発話内容と一致させる
- 単一話者、BGM なし、重なり話者なし
- 冒頭末尾の長い無音を避ける
- 音割れ、強いノイズ、強い残響を避ける
- まずは感情が強すぎない自然な音声で試す

## 実装メモ

- Docker イメージは `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` ベース
- PyTorch は CUDA 12.8 wheel を使用
- `flash-attn` は自動導入を試みますが、失敗しても標準 attention にフォールバックします
- Whisper は `faster-whisper` を第一候補にし、必要時に OpenAI Whisper へ落とせる構造にしています

## 参考にした一次情報

- Qwen3-TTS GitHub: <https://github.com/QwenLM/Qwen3-TTS>
- Qwen3-TTS Technical Report: <https://arxiv.org/abs/2601.15621>
- Qwen3-TTS 1.7B Base: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>
- Docker Desktop GPU support: <https://docs.docker.com/desktop/features/gpu/>
- Docker Compose GPU support: <https://docs.docker.com/compose/how-tos/gpu-support/>
- NVIDIA zero-shot voice cloning guide: <https://docs.nvidia.com/nim/speech/26.02.0/tts/voice-cloning.html>
- OpenAI Whisper: <https://github.com/openai/whisper>
- faster-whisper: <https://github.com/SYSTRAN/faster-whisper>
