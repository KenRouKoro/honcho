"""内嵌 deriver —— 让 API 进程内跑队列消费者，取代第二个 Python 进程。

## 为什么要合并

上游把 API 与 deriver 拆成两个进程，理由是"Never block HTTP on LLM work"
（见 ``CLAUDE.md``）。但那条约束的真实含义是 **API 必须"入队即返回"**，
而不是"必须分进程"——deriver 全程 async，与 HTTP 共用事件循环不会让请求变慢。

单用户本地部署下，拆进程的代价却很实在：

* 两个 Python 进程 + 一套监督逻辑（上游的 ``start.ps1`` 要用
  ``taskkill /T /F`` 杀进程树，并互相监视退出码）
* 事件循环策略要设两遍（deriver 在 ``__main__`` 里设了 selector，
  API 侧没有——见 ``src/loop.py`` 的说明）
* 迁移到低配设备时多一份要 always-on 的东西

## 与上游的差异

* **不注册信号处理**：``install_signal_handlers=False``。内嵌时 uvicorn 拥有
  SIGTERM/SIGINT，再注册会互抢导致关不干净。
* **带看护**：崩溃后按指数退避重启，而不是让后台任务静默死掉。
* **关闭时排空在途**：``stop()`` 先发 shutdown 请求（drain ``active_tasks``），
  再带超时等待循环退出；超时才强杀。

## 注意：CPU 密集任务会阻塞 HTTP

合并的前提是 deriver 全程 async。若将来启用 ``SURPRISAL``（``src/dreamer/surprisal.py``
用 scikit-learn 建树，``_build_tree`` 是纯 CPU），必须把那段挪到
``asyncio.to_thread``，否则会阻塞同进程的 HTTP 处理——这正是本项目在
``doc/architecture.md`` 里记过的 2026-09-24 事故形态（事件循环被同步调用卡住）。
当前 ``SURPRISAL.ENABLED`` 默认为 False，所以暂时不受影响。
"""

from __future__ import annotations

import asyncio
import logging

from .queue_manager import QueueManager

logger = logging.getLogger(__name__)

# 重启退避上限（秒）
_MAX_BACKOFF_SECONDS = 60.0
# 关闭时等待轮询循环退出的上限（秒）：排空在途 LLM 调用可能需要一会儿
_SHUTDOWN_TIMEOUT_SECONDS = 120.0

_manager: QueueManager | None = None
_task: asyncio.Task[None] | None = None


async def _supervised_loop() -> None:
    """跑 QueueManager，异常退出后按指数退避重启；被取消或正常结束则退出。"""
    global _manager
    backoff = 1.0
    while True:
        _manager = QueueManager()
        try:
            await _manager.initialize(install_signal_handlers=False)
            # initialize() 正常返回 = shutdown_event 已置位、轮询循环干净退出
            logger.info("deriver 轮询循环已正常退出")
            return
        except asyncio.CancelledError:
            logger.info("deriver 被取消")
            raise
        except Exception:
            logger.exception("deriver 异常退出，%.0fs 后重启", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
        else:
            backoff = 1.0
        finally:
            _manager = None


def start() -> asyncio.Task[None]:
    """在当前事件循环里启动内嵌 deriver，返回其任务句柄。"""
    global _task
    if _task is not None and not _task.done():
        logger.debug("deriver 已在运行，忽略重复启动")
        return _task
    _task = asyncio.create_task(_supervised_loop(), name="foxbot-deriver")
    logger.info("deriver 已内嵌启动（单进程模式）")
    return _task


async def stop(*, timeout: float = _SHUTDOWN_TIMEOUT_SECONDS) -> None:
    """请求内嵌 deriver 关闭，并排空在途工作。

    顺序很重要：先置 ``shutdown_event``（停止取新任务并 drain ``active_tasks``），
    再等循环退出；超时最后才强杀。
    """
    global _task, _manager
    task, manager = _task, _manager
    _task = None

    if manager is not None:
        try:
            await manager.shutdown(sig=None)
        except Exception:
            logger.exception("deriver shutdown() 报错，继续等待任务退出")

    if task is None:
        return
    if task.done():
        _manager = None
        # 取回异常，避免 "Task exception was never retrieved"
        with_exc = task.exception() if not task.cancelled() else None
        if with_exc is not None:
            logger.error("deriver 任务已结束但带异常: %r", with_exc)
        return

    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        logger.info("deriver 已优雅退出")
    except asyncio.TimeoutError:
        logger.warning("deriver 未在 %.0fs 内退出，强制取消", timeout)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    except Exception:
        logger.exception("等待 deriver 退出时报错")
    finally:
        _manager = None
