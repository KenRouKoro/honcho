"""事件循环工厂 —— 修复 Windows 上 psycopg 与 uvicorn 的循环不兼容。

## 问题

uvicorn 的 asyncio loop factory 按启动方式分支
(``uvicorn/loops/asyncio.py``)::

    if sys.platform == "win32" and not use_subprocess:
        return asyncio.ProactorEventLoop      # 无 reload / 单 worker
    return asyncio.SelectorEventLoop          # 有 reload 子进程 / 多 worker

而 ``use_subprocess`` 的定义是 (``uvicorn/config.py``)::

    return bool(self.reload or self.workers > 1)

psycopg 的异步模式**硬拒绝** ProactorEventLoop
(``psycopg/connection_async.py``)::

    raise InterfaceError("Psycopg cannot use the 'ProactorEventLoop' to run in async mode.")

于是 Windows 上：

* ``fastapi dev``（reload=True → use_subprocess=True）→ SelectorEventLoop → 正常
* ``fastapi run``（reload=False, workers=1 → use_subprocess=False）→ ProactorEventLoop
  → **首次异步 DB 查询即抛 InterfaceError**

上游因此只能靠 ``fastapi dev`` 工作，一旦改用更自然的 ``fastapi run`` 就会崩。
此处显式指定工厂，与启动方式无关。

## 用法

``uvicorn.Config(app, loop="src.loop:selector_loop_factory")``。
``Config.get_loop_factory()`` 对非内置取值会直接返回 import 到的对象作为零参工厂，
所以这里的签名必须是零参。
"""

from __future__ import annotations

import asyncio
import sys


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """零参事件循环工厂：Windows 上强制 SelectorEventLoop，其它平台用默认。

    ``asyncio.new_event_loop()`` 在非 Windows 上返回平台默认（Linux 即 Selector），
    因此这里不需要平台判断之外的额外分支。
    """
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def loop_spec() -> str:
    """给 ``uvicorn.Config(loop=...)`` 用的取值。

    Windows 必须显式指定（见模块 docstring）；其它平台返回 ``"auto"``，
    以保留 uvicorn 原生的 uvloop 选择，不做出无谓降级。
    """
    if sys.platform == "win32":
        return "src.loop:selector_loop_factory"
    return "auto"
