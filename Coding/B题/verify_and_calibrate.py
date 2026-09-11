"""
一次跑完两件事：环境验证 + 日志标定。

为什么要标定：题目规定**加密日志 ≤2 MB**，超限属"异常高频循环"。
我们目前只有 6 条动作 = 9219 B 这一个数据点，外推不可靠。
本次跑 N 条额外动作，用「新增字节数 / 新增动作数」量出真实的 B/动作。

用法：
    python verify_and_calibrate.py          # 默认 200 条标定动作
    python verify_and_calibrate.py 400      # 自定义条数

跑完后需要**人工退出测试**（脚本会自己 /exit）。
接着看新生成的 .jlog 大小即可反推。
"""

from __future__ import annotations

import math
import random
import sys
import time

from config import LOG_DIR, ROBOT_ID
from robot_client import RobotClient

# 官方文档第 10 节的基准步（验证用）
BASELINE = [
    ("/measure", 300, 400, 1, 105.0),
    ("/measure", 300, 400, 2, 6.0),
    ("/clear",   300, 0,   3, 83.0),
    ("/measure", 300, 0,   2, 5.0),
]
TOL = 1e-6


def main() -> int:
    n_calib = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    c = RobotClient()
    t_real0 = time.time()

    # ── 进入 ────────────────────────────────────────────────
    resp = c.enter()
    if not (resp and resp.get("accepted") is True):
        print(f"[FAIL] /enter 未接受: {resp}")
        print("       确认模拟器里已在测试中、且倒计时已结束。")
        return 1

    print("=" * 64)
    print(f"B题 验证 + 标定   robot_id = {ROBOT_ID}")
    print(f"接口 = {c.base_url}")
    print(f"现实可用 = {c.remaining_real_s(resp):.0f} s")
    print("=" * 64)

    # ── 第一阶段：官方基准验证 ───────────────────────────────
    print("\n【第一阶段】官方基准验证")
    print(f"{'步':<3}{'指令':<11}{'位置':<16}{'频道':<6}{'期望':<8}{'实际':<8}结果")
    print("-" * 64)

    passed = 0
    for i, (path, x, y, ch, expect) in enumerate(BASELINE, start=1):
        act = c.measure(x, y, ch) if path == "/measure" else c.clear(x, y, ch)
        ok = False
        if act.accepted:
            if path == "/clear":
                ok = abs(act.virtual_delta_s - 83.0) < TOL or abs(act.virtual_delta_s - 85.0) < TOL
            else:
                ok = abs(act.virtual_delta_s - expect) < TOL
        passed += ok
        extra = f"  {act.result}"
        if act.svd_deg is not None:
            extra += f" {act.svd_deg}°"
        print(f"{i:<3}{path:<11}{f'({x},{y})':<16}{ch:<6}{expect:<8.0f}"
              f"{act.virtual_delta_s:<8.0f}{'✓' if ok else '✗'}{extra}")
    print(f"\n基准验证：{passed}/{len(BASELINE)} 通过  "
          f"虚拟时刻 {c.virtual_time_s:.0f} s")

    # ── 第二阶段：标定 ──────────────────────────────────────
    # 小步随机游走，控制虚拟时间；频道轮转，覆盖面均匀。
    print(f"\n【第二阶段】标定：额外发 {n_calib} 条 /measure")
    rng = random.Random(20260910)
    x, y = 300.0, 0.0
    t0 = time.time()
    accepted = 0
    results: dict[str, int] = {}

    for i in range(n_calib):
        ang = rng.uniform(0, 2 * math.pi)
        step = rng.uniform(50.0, 200.0)
        nx = max(-1700.0, min(1700.0, x + step * math.cos(ang)))
        ny = max(-1700.0, min(1700.0, y + step * math.sin(ang)))
        ch = 1 + (i % 20)
        act = c.measure(nx, ny, ch)
        if act.accepted:
            accepted += 1
            results[act.result] = results.get(act.result, 0) + 1
        x, y = nx, ny

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1:>4}/{n_calib}  虚拟 {c.virtual_time_s:>8.0f}s  "
                  f"现实 {el:>5.1f}s  ({i+1:>4.0f}/{el:.0f} ≈ {(i+1)/max(el,1e-9):.1f} 条/秒)")

    t_calib = time.time() - t0

    # ── 退出 ────────────────────────────────────────────────
    c.exit()
    t_total = time.time() - t_real0

    # ── 报告 ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("【标定结果】")
    print(f"  额外动作数        : {n_calib}（被接受 {accepted}）")
    print(f"  标定阶段现实耗时  : {t_calib:.1f} s")
    print(f"  实际请求速率      : {n_calib / max(t_calib, 1e-9):.1f} 条/秒")
    print(f"  全程现实耗时      : {t_total:.1f} s")
    print(f"  最终虚拟时刻      : {c.virtual_time_s:.0f} s  （上限 360000）")
    print(f"  结果分布          : {results}")
    print()
    print(f"  日志动作总数 ≈ {len([a for a in c.log if a.accepted])} 条（含基准步与 enter/exit）")
    print()
    print("  → 现在去 behavior-logs 里找**最新**的 .jlog，看它多大。")
    print("  → 用 (新 .jlog 字节数 - 9219) / (动作数 - 6) 得到 B/动作。")
    print("=" * 64)

    log_path = f"{LOG_DIR}/verify_and_calibrate.json"
    c.dump_log(log_path)
    print(f"我方明文日志：{log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
