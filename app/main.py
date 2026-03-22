from __future__ import annotations

from .config import AppConfig
from .ui import build_demo


def main() -> None:
    config = AppConfig()

    if bool(config.ssl_certfile) != bool(config.ssl_keyfile):
        raise ValueError(
            "HTTPS を使う場合は GRADIO_SSL_CERTFILE と GRADIO_SSL_KEYFILE を両方設定してください。"
        )

    demo = build_demo(config)

    launch_kwargs: dict[str, object] = {
        "server_name": config.server_name,
        "server_port": config.server_port,
        "allowed_paths": [config.output_dir],
    }
    if config.ssl_certfile:
        launch_kwargs["ssl_certfile"] = config.ssl_certfile
    if config.ssl_keyfile:
        launch_kwargs["ssl_keyfile"] = config.ssl_keyfile

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
