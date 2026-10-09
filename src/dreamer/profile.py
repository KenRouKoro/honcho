"""长期印象压缩（profile compression）。

把某个 peer 的**全部 explicit 观察**压成一份稳定的"长期印象"，在 dream 收尾阶段运行，
产出存进 ``Peer.internal_metadata``（label ``"profile"``）。

三个设计事实（推导见 doc/memory_layer_decision.md §17–§19）：

1. **压缩，不是截断。** 观察条数多但不等值（§19.4），按条数截断等于假设"每条价值相同"。
   交给模型按信息量取舍，并把"这是一层压缩结果、原料多少条、覆盖到何时"作为元信息带出去，
   读取方就知道还能用检索补细节（§17 的 P3：把未知的未知变成已知的未知）。
2. **输入是稳定全集，不是每轮的选择性子集。** 读取路径
   （``crud/representation.py`` 的三路混合）每轮都变——``search_query`` 每轮不同——
   基于它压缩会每轮失稳、无法被 prompt cache 复用（§19.3）。所以这里直接查 collection：
   不分桶、不做语义筛选、只取 explicit。
3. **非 agentic 的单次 LLM 调用**，照 ``utils/summarizer.py`` 而不是 ``specialists.py``。
   压缩的输入固定、无需探索、不需要工具、输出只有一段文本，
   用带工具循环的 ``BaseSpecialist`` 只是徒增成本与不确定性（§19.8）。

截断处理：``PROFILE.MAX_INPUT_OBSERVATIONS`` 是防超长输入的护栏，不是设计目标。
一旦真的触发（集合比护栏大），**必须把上一版画像一并喂进去**——因为每次压缩都是
**整体替换**而不是累积，不这样做的话更早的原料会在第一次截断时永久消失。
护栏以内则做**无状态全量重压**，避免"压缩的压缩"逐轮漂移。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from inspect import cleandoc as c

from nanoid import generate as generate_nanoid

from src import crud, schemas
from src.config import settings
from src.dependencies import tracked_db
from src.exceptions import ResourceNotFoundException
from src.llm import honcho_llm_call
from src.llm.types import LLMTelemetryContext
from src.telemetry.events.llm import CallPurpose
from src.telemetry.logging import accumulate_metric
from src.utils.formatting import format_datetime_utc, utc_now_iso

logger = logging.getLogger(__name__)


# 画像正文的固定分节。
#
# 与 peer card 的四类前缀（IDENTITY / ATTRIBUTE / RELATIONSHIP / INSTRUCTION）**互补**：
# card 记"稳定标记"，这里记"长期叙事"——即 card 刻意排除的那一类内容（§13.1）。
# 分节固定是为了让产出**字节级稳定**（§18.4 的半静态带要求）：
# 同一批原料无论压缩几次，结构都一致，可以直接进 prompt 前缀。
PROFILE_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "生活与处境",
        "当前的生活状态、工作或学业、所在地、作息，以及明显影响其状态的事。",
    ),
    ("关注与兴趣", "长期关注的话题、正在做的项目、反复出现的兴趣点。"),
    ("关系与互动", "与其他成员、与助理的关系和互动模式，称呼、默契、在群体中的角色。"),
    ("重要事件", "有时间线意义的事件，按先后顺序，保留大致时间。"),
    ("交流风格", "说话方式、语气、口头习惯、幽默感、情绪表达特点、敏感或忌讳的话题。"),
    ("待跟进", "对方提过但尚未了结的事、答应过的事、以后可能需要主动关心的点。"),
)


def _format_section_spec() -> str:
    return "\n".join(f"## {name}\n{desc}" for name, desc in PROFILE_SECTIONS)


def profile_prompt(
    observations_text: str,
    *,
    observation_count: int,
    max_tokens: int,
    previous_profile_text: str | None = None,
) -> str:
    """构造压缩提示词。

    Args:
        observations_text: 已格式化的观察列表（时间正序）
        observation_count: 参与本次压缩的观察条数
        max_tokens: 正文输出上限
        previous_profile_text: 仅在原料被截断时提供——上一版画像覆盖了更早的原料
    """
    if previous_profile_text:
        # 先 dedent 模板、再插值——理由见下面主模板处的注释。
        carry_section = c("""
        **本次给了你上一版画像。** 说明原料比护栏多，你手上的观察只是最近的一段，
        更早的信息只存在于上一版画像里。请把两者**合并**：上一版里仍然成立、
        仍然重要的内容必须保留，不要因为这次没看到原始观察就丢掉它。
        上一版里已被新观察推翻或更新的内容，按新的写。

        <previous_profile>
        {previous_profile}
        </previous_profile>
        """).format(previous_profile=previous_profile_text)
    else:
        carry_section = "这次给你的是**全量**原料，请从零重写整份画像，不要保留任何你记忆里的旧版本。"

    # 注意：必须**先 dedent 模板、再插值**。反过来（`c(f"...{x}...")`）时，
    # 插进来的多行文本自带 0 缩进，会把 cleandoc 的最小缩进拉到 0，
    # 于是整段模板的 4 空格缩进全部残留进提示词（`utils/summarizer.py` 就是这个形状）。
    template = c("""
    你是长期记忆的压缩器。下面是从一个用户的对话里逐条抽取出来的观察（explicit
    observations），按时间从早到晚排列。请把它们压成一份**长期印象**，
    供助理在往后的对话里反复当作背景使用。

    {carry_section}

    写作要求：

    1. **只写观察支持得住的内容。** 不要推测、不要补全、不要写观察里没有的细节。
    2. **保住具体性。** 人名、地点、数字、时间、专有名词、关键的原话措辞都要留住。
       概括到只剩抽象标签的画像没有价值。
    3. **合并重复与同类项**，但**不要丢掉少数派信息**——只出现一次的事实往往最有用。
    4. 随时间变化的（换了工作、兴趣变了、关系变了）要写出**变化的方向和先后**，
       不要取平均、不要写成"有时 A 有时 B"。
    5. **只写与下面分节相关的内容。** 某节实在没有内容就写"（暂无）"，
       不要为了填满而编，也不要另起新分节。
    6. 用**观察本身的语言**输出（观察是中文就写中文）。
    7. 输出就是画像正文本身——不要前言、不要解释、不要元评论、不要复述这些要求。

    固定分节（按此顺序，节标题原样使用）：

    {section_spec}

    <observations count="{observation_count}">
    {observations_text}
    </observations>

    硬上限：{max_tokens} token。宁可短而准，不要长而虚。
    """)

    return template.format(
        carry_section=carry_section,
        section_spec=_format_section_spec(),
        observation_count=observation_count,
        observations_text=observations_text,
        max_tokens=max_tokens,
    )


def _format_observations(
    observations: list[tuple[str, datetime]],
) -> str:
    """把观察渲染成 ``[YYYY-MM-DD] 内容`` 的时间正序列表。

    刻意只带日期、不带 session 名：画像的粒度是**全局 per user**（§18.6），
    按群切分会把同一份原料重复压 N 遍；跨群的差异交给每轮 ``search_query`` 语义召回补。
    """
    lines: list[str] = []
    for content, created_at in observations:
        # created_at 是 timezone-aware 的（models.Document 用 DateTime(timezone=True)），
        # format_datetime_utc 对 naive 值也会按 UTC 兜底。
        day = format_datetime_utc(created_at)[:10]
        lines.append(f"[{day}] {content}")
    return "\n".join(lines)


async def compress_profile(
    workspace_name: str,
    observer: str,
    observed: str,
    *,
    run_id: str | None = None,
    task_name: str | None = None,
) -> schemas.PeerProfile | None:
    """压缩某个 peer 的长期印象并落库。

    读取的是 ``(observer, observed)`` collection（§16.4 已证 deriver 对同一 observed
    写出的内容与 observer 无关），但**写入 observed 自己的** ``internal_metadata``——
    这就是"全局 per user"的粒度（§18.6）。

    这个函数**不会抛异常**：它是 dream 的收尾增强步骤，失败只该告警，不该影响
    dream 已经产出的 deductive/inductive 结果（§19.8 方案 B 的独立性要求）。

    Args:
        workspace_name: 工作区名
        observer: 读取哪个 observer 的 collection
        observed: 被观察者，同时也是画像的归属者
        run_id: 上游 dream 的 run_id，仅用于遥测串联
        task_name: 上游 dream 的 metric task 名；给了就记进度指标

    Returns:
        落库成功时返回新画像；跳过或失败时返回 None。
    """
    if not settings.PROFILE.ENABLED:
        return None

    # 局部导入：`src.crud.representation` 会在模块级导入 `src.dreamer.dream_scheduler`，
    # 从而触发 `src.dreamer.__init__` → orchestrator → 本模块。放在模块级会构成循环导入
    # （与本仓库里 `surprisal` 的处理方式一致）。
    from src.crud.representation import RepresentationManager

    start = time.perf_counter()
    limit = settings.PROFILE.MAX_INPUT_OBSERVATIONS
    manager = RepresentationManager(
        workspace_name, observer=observer, observed=observed
    )

    previous: schemas.PeerProfile | None = None
    # 第一个 DB 窗口：只读，取完全部需要的原料就关掉——外部调用期间绝不持有会话。
    async with tracked_db("profile.fetch_observations") as db:
        documents, truncated = await manager.get_all_explicit(db, limit)
        if truncated:
            # 整体替换语义 + 截断 = 更早的原料会永久消失，所以把上一版画像带上一起压。
            try:
                previous = await crud.get_peer_profile(
                    db, workspace_name, peer_name=observed
                )
            except ResourceNotFoundException:
                previous = None
        observations = [(doc.content, doc.created_at) for doc in documents]

    if not observations:
        logger.debug(
            "No explicit observations for %s/%s/%s; skipping profile compression",
            workspace_name,
            observer,
            observed,
        )
        return None

    if truncated:
        carried = (
            "carried forward" if previous else "MISSING -- older material not covered"
        )
        logger.warning(
            "Profile compression for %s/%s is truncated at %s observations (%s)",
            workspace_name,
            observed,
            limit,
            carried,
        )

    prompt = profile_prompt(
        _format_observations(observations),
        observation_count=len(observations),
        max_tokens=settings.PROFILE.MAX_TOKENS,
        previous_profile_text=previous.text if previous else None,
    )

    trace_id = generate_nanoid()
    try:
        response = await honcho_llm_call(
            model_config=settings.PROFILE.MODEL_CONFIG,
            prompt=prompt,
            max_tokens=settings.PROFILE.MAX_TOKENS,
            telemetry=LLMTelemetryContext(
                workspace_name=workspace_name,
                call_purpose=CallPurpose.DREAM_PROFILE.value,
                parent_category="dream",
                run_id=run_id,
                trace_id=trace_id,
                span_id=trace_id,
                observer=observer,
                observed=observed,
                peer_name=observed,
                track_name="Profile Compression",
            ),
        )
    except Exception:
        logger.exception(
            "Profile compression LLM call failed for %s/%s/%s",
            workspace_name,
            observer,
            observed,
        )
        if task_name:
            accumulate_metric(task_name, "profile_compress_error", 1, "count")
        return None

    text = (response.content or "").strip()
    if not text:
        # 空输出**绝不覆盖**已有画像——旧的总比没有好。
        logger.error(
            "Empty profile compression output for %s/%s/%s (finish_reasons=%s)",
            workspace_name,
            observer,
            observed,
            response.finish_reasons,
        )
        if task_name:
            accumulate_metric(task_name, "profile_compress_empty", 1, "count")
        return None

    # documents 是时间正序（get_all_explicit 已保证）。
    window_oldest = format_datetime_utc(observations[0][1])[:10]
    window_newest = format_datetime_utc(observations[-1][1])[:10]
    if truncated and previous is not None:
        # 上一版覆盖了更早的一段，所以覆盖范围与条数都要往前累加。
        oldest = previous.oldest or window_oldest
        observation_count = previous.observation_count + len(observations)
    else:
        oldest = window_oldest
        observation_count = len(observations)

    profile = schemas.PeerProfile(
        text=text,
        observation_count=observation_count,
        oldest=oldest,
        newest=window_newest,
        compressed_at=utc_now_iso(),
        truncated=truncated,
    )

    # 第二个 DB 窗口：只写。
    async with tracked_db("profile.save") as db:
        try:
            await crud.set_peer_profile(db, workspace_name, profile, peer_name=observed)
        except Exception:
            logger.exception(
                "Failed to save profile for %s/%s", workspace_name, observed
            )
            if task_name:
                accumulate_metric(task_name, "profile_save_error", 1, "count")
            return None

    logger.info(
        "Profile compressed for %s/%s: %s observations (%s..%s), %s chars",
        workspace_name,
        observed,
        observation_count,
        oldest or "?",
        window_newest,
        len(text),
    )
    if task_name:
        accumulate_metric(task_name, "profile_observations", len(observations), "count")
        accumulate_metric(task_name, "profile_chars", len(text), "count")
        accumulate_metric(
            task_name,
            "profile_compress_duration",
            (time.perf_counter() - start) * 1000,
            "ms",
        )
        if truncated:
            accumulate_metric(task_name, "profile_truncated", 1, "count")

    return profile
