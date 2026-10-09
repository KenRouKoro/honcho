"""场地描述压缩（session profile）。

peer 有两层记忆（`peer_card` 稳定标记 + `peer_profile` 长期叙事），而 **session 一层都没有**——
只有每 20/60 条消息重写一次的 `summary`（流水，不是描述）。这个模块补的就是那一层：
把"这是什么地方"压成一份稳定文本，存进 `Session.internal_metadata`。

四个设计要点（doc/memory_layer_decision.md §23、§25）：

1. **触发跟着 `summary` 走，不跟 dream。** summary 是**按 session 节流**的现成载体；
   dream 是按 `(observer, observed)` collection 调的，一个群 N 个成员就有 N 个 collection，
   挂 dream 会让同一个群的描述被写 N 遍、内容互相打架（§25.2）。
2. **数字由代码算，叙事由模型写。** 成员/潜水/常驻的数目从
   `crud.get_session_stats` 来，逐字塞进提示词并要求原样引用——模型不数人头（§25.3）。
3. **潜水者只能来自客户端名册。** Honcho 的 peer 是发言驱动创建的，
   从未发言的人**没有任何对象**；没有名册时宁可不写这个数字，也不写 0（§23.4）。
4. **稳定优先于新鲜。** 分节固定、刷新频率刻意低于 summary（默认 200 条消息一次），
   这样它才能进提示词的静态前缀、被 prompt cache 复用。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from inspect import cleandoc as c

from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.exceptions import ResourceNotFoundException
from src.llm import honcho_llm_call
from src.llm.types import LLMTelemetryContext
from src.telemetry.events.llm import CallPurpose
from src.telemetry.logging import accumulate_metric
from src.utils.formatting import format_datetime_utc, utc_now_iso

logger = logging.getLogger(__name__)


# 描述正文的固定分节。
#
# 与 peer profile 的分节**互补**：那边写"这个人是谁"，这边写"这是什么地方"。
# 固定分节是为了字节级稳定（§18.4 的半静态带）——同一批原料压两次结构一致，
# 才能当提示词前缀被缓存。
SESSION_PROFILE_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "这是什么地方",
        "场地的性质与用途：这群人为什么聚在这里、围绕什么。只写对话能支持的部分。",
    ),
    ("氛围与规矩", "日常气氛、说话风格、明说过的或反复出现的规矩与禁忌。"),
    ("话题域", "长期反复出现的话题。只写反复出现的，一次性的话题不要写。"),
    ("黑话与内部梗", "群内特有的说法、梗、称呼方式、圈内缩写。"),
    (
        "成员构成",
        "成员的角色与构成。**必须逐字引用 <session_stats> 里的数字**，不要自己估。",
    ),
)


def _format_section_spec() -> str:
    return "\n".join(f"## {name}\n{desc}" for name, desc in SESSION_PROFILE_SECTIONS)


def session_profile_prompt(
    *,
    observations_text: str,
    observation_count: int,
    stats_text: str,
    max_tokens: int,
    today: str,
    previous_profile_text: str | None = None,
) -> str:
    """构造场地描述的压缩提示词。

    Args:
        observations_text: 该 session 的 explicit 观察（时间正序）
        observation_count: 参与压缩的观察条数
        stats_text: `crud.stats_block` 渲染好的统计块（**代码算的**）
        max_tokens: 正文输出上限
        today: 当前日期（ISO date）。观察只有日期，没有"现在"就判不了时效
        previous_profile_text: 仅在原料被截断时提供
    """
    if previous_profile_text:
        carry_section = c("""
        **本次给了你上一版描述。** 说明原料比护栏多，你手上的观察只是最近的一段，
        更早的信息只存在于上一版描述里。请把两者**合并**：上一版里仍然成立、
        仍然重要的内容必须保留，不要因为这次没看到原始观察就丢掉它。
        上一版里已被新观察推翻或更新的内容，按新的写。

        <previous_profile>
        {previous_profile}
        </previous_profile>
        """).format(previous_profile=previous_profile_text)
    else:
        carry_section = "这次给你的是**全量**原料，请从零重写整份描述，不要保留任何你记忆里的旧版本。"

    # 必须先 dedent 模板、再插值——否则插进来的多行文本自带 0 缩进，
    # 会把 cleandoc 的最小缩进拉到 0，整段模板的缩进全部残留进提示词。
    template = c("""
    你在为一个"场地"（一个群 / 一个长期会话）写长期描述。下面是从这个 session 的对话里
    抽取出来的观察（explicit observations），按时间从早到晚排列。**今天是 {today}。**
    请把它们压成一份**场地描述**，供助理进入这个场地时反复当作背景使用。

    {carry_section}

    写作要求：

    1. **只写观察支持得住的内容。** 不要推测、不要补全、不要写观察里没有的细节。
       尤其**不要**从"这些人聊游戏"推断"这是个游戏群"——除非对话本身说得出来。
    2. **区分"声明"与"实际"。** 如果观察到有人立过规矩、或者规矩被反复打破，
       写成「据说/规定 ……，但实际 ……」。这种不一致本身是最有价值的信息。
    3. **保住具体性。** 群内特有的说法、梗、称呼、专有名词要原样留住。
    4. **成员构成一节必须逐字引用 <session_stats> 的数字**，不要自己数、不要自己估。
       特别地：**纯潜水的人不会出现在下面的观察里**——如果统计给了潜水人数，
       就照写；如果统计说名册未知，就写"人数不明"，**不要假装群里只有出现过的人**。
    5. 用「今天」判断时效。已经过去的临时事件（某次活动、某场考试）不要写成现状。
    6. **每条观察都要有归宿。** 哪一节都不太契合但长期有价值的，写进最接近的一节，
       不要因为"分节不合适"就把内容丢掉。
    7. 用**观察本身的语言**输出（观察是中文就写中文）。
    8. 输出就是描述正文本身——不要前言、不要解释、不要元评论、不要复述这些要求。

    固定分节（按此顺序，节标题原样使用）：

    {section_spec}

    {stats_text}

    <observations count="{observation_count}">
    {observations_text}
    </observations>

    硬上限：{max_tokens} token。宁可短而准，不要长而虚。
    """)

    return template.format(
        today=today,
        carry_section=carry_section,
        section_spec=_format_section_spec(),
        stats_text=stats_text,
        observation_count=observation_count,
        observations_text=observations_text,
        max_tokens=max_tokens,
    )


def _format_observations(observations: list[tuple[str, datetime]]) -> str:
    """``[YYYY-MM-DD] 内容`` 的时间正序列表。日期用 UTC，与其余元信息一致。"""
    lines: list[str] = []
    for content, created_at in observations:
        day = format_datetime_utc(created_at)[:10]
        lines.append(f"[{day}] {content}")
    return "\n".join(lines)


async def _fetch_session_observations(
    db: AsyncSession,
    workspace_name: str,
    session_name: str,
    limit: int,
) -> list[tuple[str, datetime]]:
    """取该 session 的 explicit 观察（时间正序）。

    刻意**不**按 observer 过滤：同一条消息会被 deriver 写进每个 observer 的 collection
    （§16.4），所以没有任何单一 observer 是"权威"的。改为按内容去重，
    这样无论这个 session 有哪些 collection 都能拿到一份干净的原料。

    多取 ``limit`` 条是为了给去重留余量（重复率最高约 2×）。
    """
    stmt = (
        select(models.Document.content, models.Document.created_at)
        .where(
            models.Document.workspace_name == workspace_name,
            models.Document.session_name == session_name,
            models.Document.level == "explicit",
            models.Document.deleted_at.is_(None),
        )
        .order_by(models.Document.created_at.desc(), models.Document.id.desc())
        .limit(limit * 2)
    )
    rows = (await db.execute(stmt)).all()

    seen: set[str] = set()
    deduped: list[tuple[str, datetime]] = []
    for content, created_at in rows:
        key = content.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((content, created_at))
        if len(deduped) >= limit:
            break

    deduped.reverse()  # 时间正序送进提示词
    return deduped


async def refresh_session_profile_if_needed(
    workspace_name: str,
    session_name: str,
    message_seq_in_session: int,
    *,
    run_id: str | None = None,
    task_name: str | None = None,
) -> schemas.SessionProfile | None:
    """按条数节流地刷新场地描述。

    触发点与 ``summarizer.summarize_if_needed`` 并列（`deriver/consumer.py::process_summary_task`），
    用的也是同一种节流：``message_seq_in_session % MIN_MESSAGES_BETWEEN == 0``。
    不同的是常数大得多——描述是稳定层，不是流水。

    和 `compress_profile` 一样，这个函数**不抛异常**：它是会话维护的增强步骤，
    失败只该告警，不该影响 summary 或队列推进。

    Args:
        workspace_name: 工作区名
        session_name: 会话名
        message_seq_in_session: 触发消息在该 session 内的序号
        run_id: 上游 run_id，仅用于遥测串联
        task_name: metric task 名；给了就记进度指标

    Returns:
        刷新成功时返回新描述；跳过或失败时返回 None。
    """
    cfg = settings.SESSION_PROFILE
    if not cfg.ENABLED:
        return None

    if message_seq_in_session % cfg.MIN_MESSAGES_BETWEEN != 0:
        return None

    start = time.perf_counter()
    previous: schemas.SessionProfile | None = None

    # 第一个 DB 窗口：只读。取完原料就关掉——外部调用期间绝不持有会话。
    async with tracked_db("session_profile.fetch") as db:
        observations = await _fetch_session_observations(
            db, workspace_name, session_name, cfg.MAX_INPUT_OBSERVATIONS
        )
        stats = await crud.get_session_stats(
            db, workspace_name, session_name=session_name
        )
        if stats is None:
            logger.debug(
                "Session %s/%s vanished before profile refresh",
                workspace_name,
                session_name,
            )
            return None
        truncated = len(observations) >= cfg.MAX_INPUT_OBSERVATIONS
        if truncated:
            # 整体替换语义 + 截断 = 更早的原料会永久消失，所以带上上一版一起压。
            try:
                previous = await crud.get_session_profile(
                    db, workspace_name, session_name=session_name
                )
            except ResourceNotFoundException:
                previous = None

    if not observations:
        logger.debug(
            "No explicit observations for %s/%s; skipping session profile",
            workspace_name,
            session_name,
        )
        return None

    prompt = session_profile_prompt(
        observations_text=_format_observations(observations),
        observation_count=len(observations),
        stats_text=crud.stats_block(stats),
        max_tokens=cfg.MAX_TOKENS,
        today=utc_now_iso()[:10],
        previous_profile_text=previous.text if previous else None,
    )

    trace_id = generate_nanoid()
    try:
        response = await honcho_llm_call(
            model_config=cfg.MODEL_CONFIG,
            prompt=prompt,
            max_tokens=cfg.MAX_TOKENS,
            telemetry=LLMTelemetryContext(
                workspace_name=workspace_name,
                call_purpose=CallPurpose.DREAM_SESSION_PROFILE.value,
                parent_category="summary",
                run_id=run_id,
                trace_id=trace_id,
                span_id=trace_id,
                session_id=session_name,
                track_name="Session Profile",
            ),
        )
    except Exception:
        logger.exception(
            "Session profile LLM call failed for %s/%s", workspace_name, session_name
        )
        if task_name:
            accumulate_metric(task_name, "session_profile_error", 1, "count")
        return None

    text = (response.content or "").strip()
    if not text:
        # 空输出**绝不覆盖**已有描述——旧的总比没有好。
        logger.error(
            "Empty session profile output for %s/%s (finish_reasons=%s)",
            workspace_name,
            session_name,
            response.finish_reasons,
        )
        if task_name:
            accumulate_metric(task_name, "session_profile_empty", 1, "count")
        return None

    window_oldest = format_datetime_utc(observations[0][1])[:10]
    window_newest = format_datetime_utc(observations[-1][1])[:10]
    if truncated and previous is not None:
        oldest = previous.oldest or window_oldest
        observation_count = previous.observation_count + len(observations)
    else:
        oldest = window_oldest
        observation_count = len(observations)

    profile = schemas.SessionProfile(
        text=text,
        observation_count=observation_count,
        sample_messages=stats.message_count,
        speaker_count=stats.speaker_count,
        member_count=stats.member_count,
        lurker_count=stats.lurker_count,
        oldest=oldest,
        newest=window_newest,
        compressed_at=utc_now_iso(),
        truncated=truncated,
    )

    # 第二个 DB 窗口：只写。
    async with tracked_db("session_profile.save") as db:
        try:
            await crud.set_session_profile(
                db, workspace_name, profile, session_name=session_name
            )
        except Exception:
            logger.exception(
                "Failed to save session profile for %s/%s",
                workspace_name,
                session_name,
            )
            if task_name:
                accumulate_metric(task_name, "session_profile_save_error", 1, "count")
            return None

    logger.info(
        "Session profile refreshed for %s/%s: seq=%s, %s observations, %s speakers%s",
        workspace_name,
        session_name,
        message_seq_in_session,
        observation_count,
        stats.speaker_count,
        f", {stats.lurker_count} lurkers" if stats.lurker_count is not None else "",
    )
    if task_name:
        accumulate_metric(
            task_name, "session_profile_observations", len(observations), "count"
        )
        accumulate_metric(task_name, "session_profile_chars", len(text), "count")
        accumulate_metric(
            task_name,
            "session_profile_duration",
            (time.perf_counter() - start) * 1000,
            "ms",
        )

    return profile
