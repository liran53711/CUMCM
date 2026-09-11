"""
可用性测试：验证与模拟器进程的通信、四条指令、虚拟计时规则。

用的动作序列与官方文档第 10 节「完整计时示例」**完全相同**，
因此每一步的虚拟耗时都有官方基准值可以逐条核对。

── 使用步骤 ──────────────────────────────────────────────
1. 打开模拟器，登录（队号 202610038113）
2. 进入「问题3演练测试」→ 确认开始
3. 等 5 秒倒计时结束，界面显示机器狗接口已就绪
4. 运行：  python usability_test.py
   （脚本会自动等接口就绪，倒计时期间连不上是正常的）

── 官方基准（虚拟秒）─────────────────────────────────────
步  指令              位置        频道   本步耗时   累计
1   /enter             -           -      0         0
2   /measure       (300,400)       1      105       105
3   /measure       (300,400)       2      6         111
4   /clear         (300,0)         3      83或85    194或196
5   /measure       (300,0)         2      5         199或201
6   /exit              -           -      0         199或201
"""

from __future__ import annotations

import sys
import time

from config import LOG_DIR, ROBOT_ID
from robot_client import RobotClient

# 官方文档第 10 节的基准（步/路径/位置/频道/期望本步虚拟耗时）
STEPS = [
    ("/measure", 300, 400, 1, 105.0, "移动100s(500m) + 检测5s，未切频道"),
    ("/measure", 300, 400, 2, 6.0,   "未移动 + 切频道1s + 检测5s"),
    ("/clear",   300, 0,   3, 83.0,  "移动80s(400m) + 未发现3s；若命中则为85s"),
    ("/measure", 300, 0,   2, 5.0,   "未移动 + 未切频道(仍为2) + 检测5s"),
]
TOL = 1e-6


def wait_for_interface(c: RobotClient, timeout_s: int = 90) -> bool:
    """等接口就绪。倒计时期间 /enter 连不上是正常的，这里轮询。"""
    print(f"等待机器狗接口就绪（最多 {timeout_s} 秒）...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        resp = c.enter()
        if resp and resp.get("accepted") is True:
            print(f"  接口已就绪，用时 {time.time() - t0:.1f}s")
            return True
        # 未就绪：重置序号，避免 request_id 堆积
        c._seq = 0
        time.sleep(2)
    print("  超时：接口未就绪。请确认已在模拟器里点「开始测试」并等倒计时结束。")
    return False


def main() -> int:
    no_wait = "--no-wait" in sys.argv
    c = RobotClient()
    print("=" * 62)
    print(f"B题模拟器可用性测试   robot_id = {ROBOT_ID}")
    print(f"接口地址 = {c.base_url}")
    print("=" * 62)

    # ── 1. /enter ───────────────────────────────────────────
    resp = c.enter()
    if not (resp and resp.get("accepted") is True):
        if no_wait:
            print(f"[FAIL] /enter 失败: {resp}")
            return 1
        if not wait_for_interface(c):
            return 1
        resp = {"accepted": True, "virtual_time_s": 0.0,
                "remaining_real_duration_s": 1200}

    remaining = c.remaining_real_s(resp)
    print(f"\n[1] /enter  OK")
    print(f"    虚拟时刻      = {c.virtual_time_s}")
    print(f"    现实可用时长  = {remaining:.0f} s")
    print(f"    虚拟世界限时  = {resp.get('max_virtual_duration_s')} s")
    if remaining < 1200:
        print(f"    ⚠ 少于 1200s，说明进入偏晚，代码必须按实际值控制时间")

    # ── 2~5. 四条动作 ───────────────────────────────────────
    results: list[tuple[str, float, float, bool]] = []
    print(f"\n{'步':<3}{'指令':<11}{'位置':<16}{'频道':<6}{'期望':<8}{'实际':<8}{'结果'}")
    print("-" * 62)

    for i, (path, x, y, ch, expect, desc) in enumerate(STEPS, start=2):
        act = c.measure(x, y, ch) if path == "/measure" else c.clear(x, y, ch)

        if not act.accepted:
            print(f"{i:<3}{path:<11}{f'({x},{y})':<16}{ch:<6}{expect:<8.0f}"
                  f"{'-':<8}✗ 未接受  {act.note}")
            results.append((path, expect, -1, False))
            continue

        actual = act.virtual_delta_s
        # /clear 有两种合法结果：未发现 3s(83) 或 命中 5s(85)
        if path == "/clear":
            ok = abs(actual - 83.0) < TOL or abs(actual - 85.0) < TOL
        else:
            ok = abs(actual - expect) < TOL

        extra = ""
        if act.result == "direction":
            extra = f"  示向度 {act.svd_deg}°"
        elif act.result == "near":
            extra = "  ⚡near（≤5m，可直接 clear）"
        elif act.result == "success":
            extra = "  ★清除成功"
        elif act.result == "no_signal":
            extra = "  无信号"

        print(f"{i:<3}{path:<11}{f'({x},{y})':<16}{ch:<6}{expect:<8.0f}"
              f"{actual:<8.0f}{'✓' if ok else '✗'}{extra}")
        results.append((path, expect, actual, ok))
        time.sleep(0.05)   # 串行，不并发

    # ── 6. /exit ────────────────────────────────────────────
    vt_before_exit = c.virtual_time_s
    c.exit()
    print(f"{6:<3}{'/exit':<11}{'-':<16}{'-':<6}{0:<8.0f}"
          f"{c.virtual_time_s - vt_before_exit:<8.0f}"
          f"{'✓' if abs(c.virtual_time_s - vt_before_exit) < TOL else '✗'}")

    # ── 汇总 ────────────────────────────────────────────────
    print("-" * 62)
    passed = sum(1 for _, _, _, ok in results if ok)
    total = len(results)
    print(f"\n计时规则校验：{passed}/{total} 通过")
    print(f"最终虚拟时刻：{c.virtual_time_s:.0f} s")
    print(f"测向机最终频道：{c.current_channel}")

    log_path = f"{LOG_DIR}/usability_test.json"
    c.dump_log(log_path)
    print(f"动作日志已导出：{log_path}")

    if passed == total:
        print("\n✅ 环境就绪：通信正常、四条指令可用、虚拟计时与官方基准一致。")
        return 0
    print("\n⚠ 存在不一致，请把上面的表格发我排查。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
