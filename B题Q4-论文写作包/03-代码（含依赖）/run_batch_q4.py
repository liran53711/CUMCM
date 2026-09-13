r"""
**官方模拟器 · 问题4批量演练** —— 自动等接口、自动汇总。

    python run_batch_q4.py --runs 10

## 与 `run_batch_official.py` 的区别

那个脚本是 **Q3 专用**：策略分支只有 joint/m5/shared、提示文案写的是
「问题3演练测试」、全清判据用的是 Q3 的「7 点 no_signal 覆盖」。
**Q4 停点是 36 个、证空判据是角度覆盖**，两者都不能复用，故单独写一份。

## 你只需要做一件事

脚本会打印

    === 第 k/N 局：请在模拟器里启动「问题4演练测试」 ===

**此时你去模拟器点「问题4演练测试 → 确认」，等 5 秒倒计时结束，脚本自动接管。**
跑完自动 `/exit`、解析、汇总，然后提示你启动下一局。

脚本用 `/enter` 轮询：被拒绝时**什么也不消耗**（`accepted=false` 是无害的），
可以放心让它一直等。轮询 8 s，避免触发 429 流量保护。

## 输出

- 每局：`03-实验数据/official_q4_<时间戳>.json`（我们的动作日志）
- 汇总：控制台表格 + `03-实验数据/official_q4_batch_summary.csv`
- ⚠ **官方加密日志仍需你手动从模拟器日志列表导出**（不改文件名）放支撑材料
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, r"D:\Contest\数学建模大赛\CUMCM\Coding\B题")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from claim_tester import enable_utf8_console  # noqa: E402

# ★★ 输出目录自适应：本脚本在 q3/02-代码/ 与 B题Q4-论文写作包/03-代码（含依赖）/
#    两处都会跑。写死 `../03-实验数据` 会在 Q4 包里**整局跑完之后**写日志时崩
#    （实测 2026-09-12 20:36 踩过：`FileNotFoundError`，一局官方演练作废）。
#    见 `_q4paths.OUT`（按存在性依次探 `03-实验数据` → `04-验证数据` → `./out`）。
try:
    from _q4paths import OUT as _OUT_DIR
    DATA = str(_OUT_DIR)
except ImportError:                          # 兜底：脚本被单独拷走时
    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03-实验数据")
    os.makedirs(DATA, exist_ok=True)


def wait_for_interface(client, tag: str, poll_s: float = 8.0,
                       timeout_s: float = 900.0):
    """轮询 `/enter` 直到被接受。被拒绝不消耗任何机会。"""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        resp = client.enter()
        if resp and resp.get("accepted") is True:
            return resp
        last = (resp or {}).get("message", "连不上/接口未开放")
        if time.time() - t0 > 3:                    # 前 3 秒静默，避免倒计时刷屏
            print(f"    等待中…（{last}）已等 {time.time() - t0:.0f} s", flush=True)
        time.sleep(poll_s)
    raise TimeoutError(f"{tag}: {timeout_s:.0f} s 内接口未开放（最后状态：{last}）")


def analyze_q4(path: str) -> dict:
    """Q4 全清证据。

    **不能用 Q3 的口径**（「某频道恰有 7 次 no_signal」）—— Q4 停点是 36 个，
    且证空靠的是角度覆盖判据。这里的可靠口径是：

        有源的直接证据 = 该频道拿到过 `direction`（示向度）
        （定向源只在扇区内给示向度，拿到就一定是真源）

    于是 全清 ⟺ 拿到过 direction 的频道集合 ⊆ 被成功清除的频道集合。
    """
    d = json.load(open(path, encoding="utf-8"))
    acts = d["actions"]
    mv = 0.0
    px = py = 0.0
    nmeas = nsw = ncl = nok = 0
    prev = None
    per: dict[int, dict] = {}
    for a in acts:
        if a["path"] in ("/measure", "/clear"):
            mv += math.hypot(a["x"] - px, a["y"] - py)
            px, py = a["x"], a["y"]
        if a["path"] == "/measure":
            nmeas += 1
            c = per.setdefault(a["channel"], {"dir": 0, "ns": 0, "ok": 0, "miss": 0})
            if a["result"] == "direction":
                c["dir"] += 1
            elif a["result"] == "no_signal":
                c["ns"] += 1
            if prev is not None and a["channel"] != prev:
                nsw += 1
            prev = a["channel"]
        elif a["path"] == "/clear":
            ncl += 1
            c = per.setdefault(a["channel"], {"dir": 0, "ns": 0, "ok": 0, "miss": 0})
            if a["result"] == "success":
                c["ok"] += 1
                nok += 1
            else:
                c["miss"] += 1
    src = sorted(ch for ch in per if per[ch]["dir"] > 0)
    clr = sorted(ch for ch in per if per[ch]["ok"] > 0)
    unclear = sorted(set(src) - set(clr))
    n = d.get("final_virtual_time_s", 0.0)
    return dict(
        t=n, acts=d["action_count"], n_src=len(src), n_clr=len(clr),
        unclear=unclear,
        # ⚠ 必须要求 n_src > 0：一局完全没跑起来时 src 与 clr 都是空集，
        #   `set() == set()` 会判 True，把「什么都没做」误报成全清。
        all_clear=(len(src) > 0 and not unclear),
        s_per=(n / len(src) if src else float("nan")),
        move=mv, nmeas=nmeas, nsw=nsw, ncl=ncl,
    )


def main(argv=None) -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="跑几局")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="每局等待接口开放的秒数")
    ap.add_argument("--expected-requests", type=int, default=600)
    ap.add_argument("--reserve", type=float, default=60.0)
    ap.add_argument("--drop-outer", action="store_true", default=None,
                    help="去掉方格最外圈 8 个 r≈1789 的点（36→28 点）")
    ap.add_argument("--keep-outer", dest="drop_outer", action="store_false",
                    help="保留外圈（36 点，现行）")
    ap.add_argument("--clear-mode", choices=["prod", "localmesh", "raymesh"],
                    default="prod",
                    help="清源原语（**默认 prod**）：官方同源数分档对照显示 "
                         "localmesh **+62.7 s/个（4 档全部更差）**，已否决；"
                         "raymesh 离线 +126 s/个，已否决。")
    ap.add_argument("--meas-uncleared", action="store_true", default=False,
                    help="★ 2026-09-13【本轮要验的那一条】：常规停点**继续测**"
                         "「已定位未清除」的频道（原 `stop_two=True` 一律跳过）。\n"
                         "离线依据：5/5 配置符号一致为负（−140 / −299 / −553 / −25 "
                         "/ −177 s/局，均值 ≈ −239），且 `capture` 触发率一致下降"
                         "（2.1→1.3 / 2.3→0.9 / 1.9→0.8 / 3.7→3.0 / 1.3→0.5）；"
                         "但**单配置都不显著**（|t| ≤ 1.62），零漏清。\n"
                         "按 Q4-06 §4.3 它**替换了分支行为**（capture 触发率变了）"
                         "⟹ 离线不可信，必须官方对照。\n"
                         "⚠ 同批的 `defer_fail` 已否决（漏清 + 样本外不复现），"
                         "**不要一起打开**。")
    ap.add_argument("--no-scan-current-first", dest="scan_current_first",
                    action="store_false", default=True,
                    help="关掉「当前频道优先」（**默认开**，2026-09-13 起）。\n"
                         "开着时每个测站的待测频道表从**机器当前所在频道**起轮转，"
                         "省掉一次换频（1.073 s）；测量集合完全不变。\n"
                         "依据：与队友独立实现的隔离配对对照（`_iso_compare/`，"
                         "4 配置×20 局×5 臂=400 次配对）：**4/4 配置为负、"
                         "每一配置 20/20 局更快、跨配置 t=−18.76、零漏清**；"
                         "同批实验里「换布局」那个看起来更大的改动反而符号不稳"
                         "且引入漏清。详见 `Q4-08` §3。\n"
                         "⚠ 仅用于复现历史基线，**不要**在正式演练里关它。")
    ap.add_argument("--policy", choices=["pick", "roll"], default="roll",
                    help="清源调度（**默认 roll**，2026-09-13 起）："
                         "roll=滚动重规划（动态旅行商）。离线 68 配对局 −37.4 s/个；"
                         "官方**同源数分档对照 −32.0 s/个、5/5 档方向一致**，"
                         "走位 6 档里 5 档更低。pick=固定巡回+顺路才清（旧生产）。")
    a = ap.parse_args(argv)

    from robot_client import RobotClient
    from official_robot import RobotOfficial, TimeBudgetExceeded
    import q3_strategies as Q
    import q4_strategies as S4
    import run_official as RO

    print("=" * 86)
    print(f"  官方模拟器 · 问题4批量演练：{a.runs} 局")
    # ★ 布局由命令行决定，**不再从 run_official 读** —— 实测该跨模块读取在
    #   某些启动方式下与源文件不一致（诊断了两轮未定位），直接去掉这个耦合。
    _drop = bool(a.drop_outer) if a.drop_outer is not None else bool(
        getattr(RO, "Q4_DROP_OUTER", False))
    _npts = len(S4.C.square_grid(RO.Q4_STEP)) + RO.Q4_KRING - (8 if _drop else 0)
    _tag = "，已去外圈 8 点" if _drop else ""
    print(f"  [启动自检] drop_outer={_drop}  停点={_npts}（应为 {'28' if _drop else '36'}）"
          f"  policy={a.policy}  clear_mode={a.clear_mode}"
          f"  meas_uncleared={bool(a.meas_uncleared)}"
          f"  scan_current_first={bool(a.scan_current_first)}"
          f"  defer_fail=0（已否决，恒为 0）", flush=True)
    print(f"  布局：内层方格 {RO.Q4_STEP:.0f} m + 外推环 "
          f"ρ={1800.0 + RO.Q4_RINGEPS:.0f} m × {RO.Q4_KRING} 点"
          f"（共 {_npts} 停点{_tag}）")
    print("=" * 86, flush=True)

    rows = []
    for idx in range(1, a.runs + 1):
        print()
        print("=" * 86)
        print(f"=== 第 {idx}/{a.runs} 局：请在模拟器里启动「问题4演练测试」 ===")
        print("    （点「问题4演练测试 → 确认」，等 5 秒倒计时；脚本会自动接管）")
        print("=" * 86, flush=True)

        client = RobotClient()
        try:
            enter = wait_for_interface(client, f"第{idx}局", timeout_s=a.timeout)
        except TimeoutError as e:
            print(f"  ✗ {e}")
            rows.append(dict(run=idx, ok=False, note=str(e)))
            continue

        rb = RobotOfficial(client=client, reserve_real_s=a.reserve,
                           expected_requests=a.expected_requests)
        rb.enter_resp = enter
        rb._budget = client.remaining_real_s(enter)
        rb._t0 = time.time()
        log = os.path.join(DATA, f"official_q4_{time.strftime('%m%d_%H%M%S')}.json")
        note = ""
        if rb._budget <= 0:                       # 这一局压根没开起来
            note = (f"本局现实预算为 0（{enter.get('message', '')}）—— "
                    f"模拟器没有开新的一局，本局作废、不计数")
            print(f"  ✗ {note}", flush=True)
            try:
                rb.exit()
            except Exception:                     # noqa: BLE001
                pass
            rows.append(dict(run=idx, ok=False, note=note))
            continue

        prev_clear = Q.go_clear
        try:
            import q3_mec_clear as MC
            Q.go_clear = MC.go_clear_mec          # Q4 生产用的清除原语
            S4.run_q4(rb, Q.Knowledge(dir_aware=True),
                      step=RO.Q4_STEP, k_ring=RO.Q4_KRING,
                      ring_eps=RO.Q4_RINGEPS,
                      drop_outer=_drop, policy=a.policy,
                      clear_mode=a.clear_mode,
                      meas_uncleared=bool(a.meas_uncleared),
                      scan_current_first=bool(a.scan_current_first))
        except TimeBudgetExceeded as e:
            note = f"现实预算耗尽：{e}"
        except Exception as e:                    # noqa: BLE001
            note = f"异常：{type(e).__name__}: {e}"
        finally:
            Q.go_clear = prev_clear
            try:
                rb.exit()
            except Exception:                     # noqa: BLE001
                pass
            rb.dump_log(log)

        r = analyze_q4(log)
        r.update(run=idx, ok=r["all_clear"], note=note, log=log,
                 policy=a.policy, meas_uncleared=bool(a.meas_uncleared))
        rows.append(r)
        print(f"  虚拟 {r['t']:.1f} s ｜ 源 {r['n_src']} 全清 {r['n_clr']} "
              f"｜ **{r['s_per']:.1f} s/个** ｜ 移动 {r['move']:.0f} m "
              f"｜ 检测 {r['nmeas']} 次 ｜ 请求 {r['acts']}")
        print(f"  {'✓ 全清' if r['all_clear'] else '✗ 有源未清：' + str(r['unclear'])}"
              f"{'  ' + note if note else ''}", flush=True)
        if rb.req_times:
            print(f"  官方单请求延迟：均值 {rb.mean_latency * 1000:.1f} ms "
                  f"（{len(rb.req_times)} 次采样）", flush=True)

    # ── 汇总 ────────────────────────────────────────────────
    print()
    print("=" * 86)
    print("  汇总")
    print("=" * 86)
    g = [r for r in rows if "t" in r]
    if not g:
        print("  没有任何一局跑成功。")
        return 0

    def med(v):
        s = sorted(v)
        return s[len(s) // 2] if s else float("nan")

    ok = sum(1 for r in g if r["all_clear"])
    print(f"  有效局数 {len(g)}/{a.runs}      全清 {ok}/{len(g)}")
    print(f"  s/个   中位 {med([r['s_per'] for r in g]):.1f}   "
          f"均值 {sum(r['s_per'] for r in g) / len(g):.1f}   "
          f"全距 {min(r['s_per'] for r in g):.1f}–{max(r['s_per'] for r in g):.1f}")
    print(f"  源数   中位 {med([r['n_src'] for r in g]):.0f}   "
          f"分布 {sorted(r['n_src'] for r in g)}")
    print(f"  虚拟时 中位 {med([r['t'] for r in g]):.0f} s   "
          f"请求 中位 {med([r['acts'] for r in g]):.0f}")
    print()
    print(f"  {'局':>3}{'源':>4}{'清':>4}{'虚拟s':>9}{'s/个':>8}"
          f"{'移动m':>8}{'检测':>6}{'请求':>6}  判定")
    print("  " + "-" * 74)
    for r in g:
        print(f"  {r['run']:>3}{r['n_src']:>4}{r['n_clr']:>4}{r['t']:>9.0f}"
              f"{r['s_per']:>8.1f}{r['move']:>8.0f}{r['nmeas']:>6}{r['acts']:>6}"
              f"  {'✓' if r['all_clear'] else '✗ ' + str(r['unclear'])}")

    out = os.path.join(DATA, "official_q4_batch_summary.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["run", "n_src", "n_clr", "t", "s_per",
                                          "move", "nmeas", "nsw", "ncl", "acts",
                                          "all_clear", "unclear", "note", "log",
                                          "policy", "meas_uncleared"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print()
    print(f"  CSV → {out}")
    print(f"  ⚠ 官方加密日志仍需你手动从模拟器日志列表导出（不改文件名）放支撑材料")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    sys.exit(main())
