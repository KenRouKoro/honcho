# foxbot 侧 Dream 调度优化任务

> 交接对象：foxbot 维护 agent
> 背景：LLM 按峰谷计费，Honcho 侧已将 dream（记忆固化）限制在每日 02:00–07:00（Asia/Shanghai）窗口内触发。但目前约 **44–79% 的 dream 运行发生在窗口外**，根源在 foxbot 自己的定期 dream sweep。

## 问题定位

`foxbot_plugin/plugins/foxbot_core/__init__.py` 约 1409–1426 行：

```python
# 定期 dream 固化：为近期活跃用户与祥凤视角调度 Honcho dream
dream_min = float(config.foxbot_honcho_dream_interval_minutes)
if _honcho is not None and dream_min > 0:
    ...
    for gid in groups_with_activity(within_sec=86400.0):
        for uid in recent_active_users(gid, limit=max_peers):
            asyncio.create_task(_honcho.schedule_dream(pid, session_id=gid))
            asyncio.create_task(_honcho.schedule_dream(AGENT_PEER_ID, observed=pid, session_id=gid))
```

这个 sweep 调用的是 Honcho 的**手动** `POST /v3/workspaces/{id}/schedule_dream` 接口。该接口的语义就是**绕过所有自动调度保护**：

- 绕过文档阈值（`DREAM_DOCUMENT_THRESHOLD=50`）
- 绕过最小间隔（`DREAM_MIN_HOURS_BETWEEN_DREAMS=8h`）
- 绕过去重（pending 检查）
- **绕过 Honcho 新加的谷时窗口（`DREAM_ALLOWED_WINDOW_*`）**

## 证据（本地计量数据，09-12 09:38 → 09-13 16:17）

| 时段 | specialist 运行数 | 输入 tokens |
|---|---|---|
| 窗口内 (02:00–07:00) | 192 (21%) | 33.0M (22%) |
| **窗口外** | **721 (79%)** | **118.0M (78%)** |

09-13 全天（积压已清完，为稳态参考）：窗口内 192 次 vs 窗口外 148 次，即仍有 **44%** 由 sweep 在白天触发。单次 dream specialist 平均消耗 ~165k 输入 token。

## 要求改动（二选一，推荐 A）

### 方案 A（推荐）：关闭 foxbot 的 dream sweep

将 `foxbot_honcho_dream_interval_minutes` 配置设为 `0`。

理由：Honcho 的自动调度已经覆盖 dream 触发 —— 每个 collection 在积累 ≥50 条新显式结论且距上次 dream ≥8 小时时自动调度，且被窗口约束在 02:00–07:00。foxbot 的 sweep 是重复建设，且因绕过全部保护机制而产生大量低价值 dream（材料不足的空跑约占 28%，每次空跑也烧 ~165k 输入 token）。关闭后 dream 总量预计下降 40% 以上，且全部落在谷时。

### 方案 B：保留 sweep，但接入睡眠窗口门控

如果 foxbot 业务上必须保留主动触发（例如希望活跃用户当天就完成固化），则将 sweep 的执行门控到 foxbot 已有的 `sleep_win`（同文件约 1378 行已有的睡眠窗口概念）：

- 仅当当前时间处于睡眠窗口内时执行 dream sweep；
- 窗口外的 sweep 周期直接跳过（不要补跑）。

注意：方案 B 只是时间移位，不能减少总次数；方案 A 才能同时减量 + 移位。

## 不要做的事

- 不要修改 Honcho 侧代码或配置来"适配" sweep —— 窗口和手动接口语义是有意设计；
- 不要给手动 `schedule_dream` 接口加窗口（它是运维逃生门，保留即时语义）。

## 验证方式

改动生效后，次日白天（07:00 后）联系 Honcho 侧跑一句 metrics.jsonl 聚合统计：窗口外时段不应再出现 `dreamer_deduction` / `dreamer_induction` 运行记录（或仅剩 Honcho 自动调度在窗口边界处理的个位数残留）。

## 参考数据（单次 dream 成本画像，供权衡）

- 单次 specialist 平均输入 165k tokens（其中 ~76% 命中 DeepSeek 上下文缓存，命中价约 1/10）
- 单次 dream（deduction + induction 两个 specialist）有效计费输入约 10 万 token 当量
- 当前 dream 占 Honcho 全部 LLM 输入的 **99.2%**，是唯一值得优化的部分
