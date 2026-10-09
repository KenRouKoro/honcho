"""FoxBot 分叉的单进程入口 —— ``python -m src.run``。

与上游的差别：

1. **显式指定事件循环**（``loop=loop_spec()``）。上游靠 ``fastapi dev`` 的
   reload 子进程绕开 Windows 上 psycopg 不兼容 ProactorEventLoop 的问题，
   改用 ``fastapi run`` 会崩。详见 ``src/loop.py``。
2. **API 与 deriver 同进程**：deriver 由 app 的 lifespan 启动（见 ``src/main.py``），
   于是只有一个 Python 进程要监督、只有一处事件循环策略。
3. 不依赖 ``fastapi`` CLI —— 那个 CLI 不接受自定义 loop。

用法::

    uv run python -m src.run                 # 默认 127.0.0.1:8000
    uv run python -m src.run --port 8010
    uv run python -m src.run --no-deriver    # 只起 API（调测）

环境变量 ``HONCHO_HOST`` / ``HONCHO_PORT`` 可作默认值。
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from src.loop import loop_spec
from src.main import app

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r 不是合法整数，改用 %d", name, raw, default)
        return default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.run",
        description="FoxBot 的 Honcho 单进程服务（API + deriver）",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HONCHO_HOST", "127.0.0.1"),
        help="监听地址（默认 127.0.0.1，仅本机）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("HONCHO_PORT", 8000),
        help="监听端口（默认 8000）",
    )
    parser.add_argument(
        "--no-deriver",
        action="store_true",
        help="不启动内嵌 deriver，只跑 API（调测用）",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "").strip().lower() or None,
        help="覆盖日志级别（默认取 settings.LOG_LEVEL）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 供 app lifespan 决定是否内嵌 deriver
    os.environ["FOXBOT_EMBED_DERIVER"] = "0" if args.no_deriver else "1"

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        loop=loop_spec(),
        log_level=args.log_level,
        access_log=True,
    )
    logger.info(
        "启动 FoxBot Honcho | %s:%d | loop=%s | deriver=%s",
        args.host,
        args.port,
        loop_spec(),
        "off" if args.no_deriver else "embedded",
    )
    uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
