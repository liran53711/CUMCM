"""
问题 1 验证命令行入口。

    python verify_q1.py                       # geometry 自检 + 22 条 claim
    python verify_q1.py --only cover_band      # 只跑一条
    python verify_q1.py --skip n_generality
    python verify_q1.py --mutant=wedge_cross_reversed    # 伪证检验，必须退出 1
    python verify_q1.py --stats omega          # Ω 活化率/失配率/方向（论文数字）
    python verify_q1.py --sweep gamma --csv logs/q1_gamma.csv
    python verify_q1.py --quick                # 缩小实例数，冒烟用

退出码：全部通过 0；有 claim 挂 1。

`--mutant=X` 的退出码是**反的**，这是计划里的验收标准原文——
「每个 mutant 都必须退出 1，否则测试台抓不住静默错」：
    退出 1 = mutant 被抓（suite 变红）✓ 期望结果
    退出 0 = mutant 逃逸（suite 仍绿）✗ 测试台有盲区
所以这里 **0 是坏消息**。脚本会把结论原样打出来，别只看退出码。
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from claim_tester import (
    MUTANTS,
    Outcome,
    apply_mutant,
    enable_utf8_console,
    restore_geometry,
    run_claim,
    run_sweep,
    write_csv,
)
import claims_q1 as Q1
import geometry

_REFUTED_NOTE = "⚠ 设计上必须红：该猜想已被证伪，**不得写入论文**"


# ── 呈现 ────────────────────────────────────────────────────
def _print_table(outs: list[Outcome]) -> None:
    print()
    print("=" * 100)
    print("表 Q1-1  结论注册表 —— 逐条验证结果（论文可直接引用）")
    print("=" * 100)
    print(f"{'#':>2}  {'claim':<32} {'来源':<16} {'状态':<20} {'检验':>6} {'跳过':>6} {'反例':>5}")
    print("-" * 100)
    for k, o in enumerate(outs, 1):
        print(f"{k:>2}  {o.claim.name:<32} {o.claim.source:<16} "
              f"{o.status:<20} {o.checked:>6} {o.skipped:>6} {len(o.failures):>5}")
    print("-" * 100)
    n_hold = sum(1 for o in outs if o.claim.expect == "hold")
    n_ref = len(outs) - n_hold
    print(f"必须成立 {n_hold} 条 + 必须被证伪 {n_ref} 条 = 共 {len(outs)} 条")
    for o in outs:
        if o.claim.expect == "refuted":
            print(f"  {_REFUTED_NOTE}：{o.claim.name}")
    print()


def _print_failures(outs: list[Outcome]) -> None:
    for o in outs:
        if o.ok:
            continue
        print(f"[✗] {o.claim.name} —— {o.claim.statement}")
        print(f"    来源：{o.claim.source}")
        for f in o.failures:
            print(f.describe())
        print()


# ── 统计与扫描 ──────────────────────────────────────────────
ERR_MODES = ("zero", "uniform", "extreme")


def stats_omega(n: int = 20000, seed: int = 20260911) -> dict[str, dict]:
    """Ω 的活化率、与窗口定理的失配率、失配方向。**论文引用的就是这几个数。**

    复用 `sample_instance` 的物理自洽采样律（单边 |S−G| ≤ r_eff）。
    这个采样律必须与文档一致，否则数字无法被复核。

    ⚠ **误差模型是必须写明的第三个旋钮。** 同样 20000 个实例，只换 δ 的取法：

        err_mode=zero     δ_i ≡ 0（观测无误差）      失配 17.3%
        err_mode=uniform  δ_i ~ U(−ε,ε)              失配 24.6%
        err_mode=extreme  |δ_i| = ε（最坏情形）       失配 28.3%

    早先文档里的 17.1% 就是 `zero` 那一档——当时的临时脚本没记录误差模型，
    于是这个数被当成了「Ω 的失配率」，其实它只是三档里最乐观的一档。
    与 §2.2 的采样约定教训同源：**测试台必须把自己的采样律写全**（源分布、
    站点约束、误差模型三者缺一不可）。
    """
    import random
    from claim_tester import sample_instance

    out: dict[str, dict] = {}
    for mode in ERR_MODES:
        rng = random.Random(seed)
        total = act = checked = mismatch = brk = repair = 0
        buckets: dict[tuple[int, int], list[int]] = {}
        while total < n:
            i = sample_instance(rng, n=2, err_mode=mode)
            if i is None:
                continue
            r = Q1.region_of(i)
            if not r.bounded:
                continue
            total += 1
            if not r.omega_active:
                continue
            act += 1
            g = Q1.raw_gamma(i.bearings[0], i.bearings[1])
            pred_not = 90.0 < g < 90.0 + 2.0 * i.err_deg
            lo = int(g // 10) * 10
            b = buckets.setdefault((lo, lo + 10), [0, 0, 0])
            b[0] += 1
            if pred_not:
                b[1] += 1
            if not r.diameter_circle_ok:
                b[2] += 1
            checked += 1
            if r.diameter_circle_ok == pred_not:
                mismatch += 1
                if not r.diameter_circle_ok:
                    brk += 1             # 定理说覆盖，Ω 让它变成不覆盖
                else:
                    repair += 1          # 定理说不覆盖，Ω 让它变成覆盖
        out[mode] = dict(total=total, activated=act, checked=checked,
                         mismatch=mismatch, breaks=brk, repairs=repair,
                         buckets=buckets)
    return out


def _print_stats(all_s: dict[str, dict]) -> None:
    print()
    print("=" * 84)
    print("表 Q1-2  Ω 对覆盖窗口定理的影响（按误差模型分档；Ω 活化子域内统计）")
    print("=" * 84)
    head = f"  {'误差模型':<10}{'有界实例':>9}{'Ω 活化':>10}{'活化率':>8}" \
           f"{'失配':>7}{'失配率':>8}{'方向(break:repair)':>20}"
    print(head)
    print("  " + "-" * 80)
    for mode, s in all_s.items():
        print(f"  {mode:<10}{s['total']:>9}{s['activated']:>10}"
              f"{100.0 * s['activated'] / max(1, s['total']):>7.1f}%"
              f"{s['mismatch']:>7}"
              f"{100.0 * s['mismatch'] / max(1, s['checked']):>7.1f}%"
              f"{str(s['breaks']) + ':' + str(s['repairs']):>20}")
    print("  " + "-" * 80)
    print("  读法：break = 定理说覆盖、Ω 让它变成不覆盖；repair = 反向。")
    print("        Ω 几乎单向地**制造不覆盖**。论文引用必须同时给出误差模型。")
    print()

    main = all_s["extreme"]
    print(f"  最坏情形（err_mode=extreme）按 γ 分箱：")
    print(f"  {'γ 区间':<14}{'Ω 活化':>9}{'定理预测不覆盖':>16}{'实际不覆盖':>12}")
    print("  " + "-" * 52)
    for (lo, hi), b in sorted(main['buckets'].items()):
        print(f"  [{lo:>3}°,{hi:>3}°)   {b[0]:>9}{b[1]:>16}{b[2]:>12}")
    print("=" * 84)
    print()


def sweep_gamma(instances_seed: int = 20260911, n_sites: int = 2) -> list[dict]:
    """γ 扫描：窗口定理的论文图。Ω 不活化的实例才计入「定理预测」。"""
    import random
    from claim_tester import sample_instance

    rng = random.Random(instances_seed)
    rows: list[dict] = []
    want = 2400
    while len(rows) < want:
        i = sample_instance(rng, n=n_sites, err_mode="uniform")
        if i is None:
            continue
        r = Q1.region_of(i)
        if not r.bounded:
            continue
        g = Q1.raw_gamma(*i.bearings) if i.n == 2 else 0.0
        rows.append({
            "n": i.n,
            "gamma": round(g, 6),
            "eps": i.err_deg,
            "diameter": r.diameter,
            "cover": int(r.diameter_circle_ok),
            "omega_active": int(r.omega_active),
            "cover_margin": r.cover_margin,
            "m": r.m,
        })
    return rows


# ── 主流程 ──────────────────────────────────────────────────
def sweep_n(ns=(2, 3, 4, 5, 6, 8), per_n: int = 4000, seed: int = 20260911,
            err_mode: str = "uniform") -> list[dict]:
    """n 扫描：失败率 / 𝒟 / 𝒟/2 随 n 的变化。**带 Ω**。

    ⚠ 计划文档 §一.4 的同名表**不可比**，三处差别都在这里：
      ① 旧表是 **Ω-free**（只用楔形半平面）；本表走带 Ω 的 `localization_region`。
      ② 旧表的采样律有错（双边站点约束），本表用单边 |S_i−G| ≤ r_eff（见 §2.2）。
      ③ **旧表报「平均直径」，本表报「中位数」。** 这不是格式差异——
         Ω-free 的 𝒟 在 γ→180° 近退化构形上会发散（实测均值 1.9×10¹⁰ m），
         均值是个无意义的数。中位数才是稳健的。
      实测 𝒟 中位数在 Ω-free 与带 Ω 下几乎相同（n=2：72.99 vs 69.70），
      所以**本表与旧表的差异主要来自 ② ③，不是 Ω**。

    记 `frac_corollary` 一列 = 𝒟 > 2·r_clear 的比例，即**命题 A 的推论真正触发的比例**。
    命题 A 的推论是**条件式**的（`𝒟/2 > r_clear` 才推出「单点不可能覆盖并清除」），
    𝒟 小时它**不构成阻碍**，而不是「不成立」。n≥4 时多数实例落在这一支——
    这是 Q3 必须同时处理两条分支的原因，不能只写「必须先逼近后清除」。
    """
    import random
    from claim_tester import sample_instance
    from geometry import CLEAR_RADIUS

    rows: list[dict] = []
    for n in ns:
        rng = random.Random(seed * 1000 + n)
        got = notcover = act = fires = 0
        ds: list[float] = []
        tries = 0
        cap = per_n * 30
        while got < per_n and tries < cap:
            tries += 1
            i = sample_instance(rng, n=n, err_mode=err_mode)
            if i is None:
                continue
            r = Q1.region_of(i)
            if not r.bounded or r.diameter <= 0.0:
                continue                      # 退化：不计入分母
            got += 1
            ds.append(r.diameter)
            if not r.diameter_circle_ok:
                notcover += 1
            if r.omega_active:
                act += 1
            if r.diameter > 2.0 * CLEAR_RADIUS:
                fires += 1
        ds.sort()
        rows.append({
            "n": n,
            "total": got,
            "notcover": notcover,
            "fail_rate": (notcover / got) if got else 0.0,
            "omega_active": act,
            "omega_rate": (act / got) if got else 0.0,
            "median_diameter": ds[len(ds) // 2] if ds else 0.0,
            "half_median": (ds[len(ds) // 2] / 2.0) if ds else 0.0,
            "corollary_fires": fires,
            "corollary_rate": (fires / got) if got else 0.0,
            "rejected": tries - got,
        })
    return rows


def _print_sweep_n(rows: list[dict], err_mode: str) -> None:
    from geometry import CLEAR_RADIUS

    print()
    print("=" * 100)
    print(f"表 Q1-3  覆盖失败率与定位精度随 n 的变化（**带 Ω**，err_mode={err_mode}）")
    print("=" * 100)
    print(f"  {'n':>2}{'有效实例':>10}{'覆盖=否':>9}{'失败率':>9}"
          f"{'Ω 活化':>9}{'活化率':>9}{'𝒟 中位数':>11}{'𝒟/2':>9}"
          f"{'命题A 推论触发':>15}{'比例':>8}")
    print("  " + "-" * 92)
    for r in rows:
        print(f"  {r['n']:>2}{r['total']:>10}{r['notcover']:>9}"
              f"{100.0 * r['fail_rate']:>8.2f}%{r['omega_active']:>9}"
              f"{100.0 * r['omega_rate']:>8.1f}%"
              f"{r['median_diameter']:>11.1f}{r['half_median']:>9.1f}"
              f"{r['corollary_fires']:>15}{100.0 * r['corollary_rate']:>7.1f}%")
    print("  " + "-" * 92)
    print("  读法：判据形式对任意 n 不变（仍查极点点积符号），但「窗口」不再是")
    print("        一维 γ 区间——失败率随 n 上升，只能作**数值观察**，不得冒充定理。")
    print(f"  触发列 = 𝒟 > 2·r_clear = {2.0 * CLEAR_RADIUS:.0f} m 的比例，即命题 A 的推论真正")
    print("        能推出「不存在单点能覆盖并清除」的比例。**它是条件式的**：")
    print("        n=2 时多数触发；n≥4 时多数不触发，推论对该实例不构成阻碍。")
    print("        → Q3 必须同时处理「必须逼近」与「直径圆足够小」两条分支。")
    print("=" * 100)
    print()


def build_lanes(n_random: int, seed: int) -> dict[str, list]:
    return {
        "physical": Q1.q1_instances(n_random=n_random, seed=seed),
        "refutation": Q1.q1_refutation_instances(n_random=max(400, n_random * 3),
                                                 seed=seed + 7),
        "frozen": Q1.q1_frozen_instances(),
    }


def main(argv: list[str] | None = None) -> int:
    enable_utf8_console()                      # 必须是第一条语句

    ap = argparse.ArgumentParser(
        description="B题 问题1 —— 覆盖判据的验证台")
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--instances", type=int, default=240)
    ap.add_argument("--only", default="", help="只跑这条 claim（逗号分隔）")
    ap.add_argument("--skip", default="", help="跳过这些 claim（逗号分隔）")
    ap.add_argument("--mutant", default="", choices=[""] + sorted(MUTANTS))
    ap.add_argument("--stats", default="", choices=["", "omega"])
    ap.add_argument("--sweep", default="", choices=["", "gamma", "n"])
    ap.add_argument("--csv", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-shrink", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.instances = 48

    rc = 0

    # 1) geometry 自身回归（14 项 + 新增）
    print("── 第 1 步：geometry.py 自身回归 ──")
    if not geometry._self_test():
        print("✗ geometry.py 自检失败——后续 claim 结果不可信")
        return 1

    # 2) Q1 claim suite
    claims = Q1.q1_claims()
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        claims = [c for c in claims if c.name in want]
    if args.skip:
        drop = {s.strip() for s in args.skip.split(",") if s.strip()}
        claims = [c for c in claims if c.name not in drop]
    if not claims:
        print("没有选中的 claim")
        return 1

    lanes = build_lanes(args.instances, args.seed)
    print(f"\n实例族：物理自洽 {len(lanes['physical'])} ／ "
          f"反例域 {len(lanes['refutation'])} ／ 冻结 {len(lanes['frozen'])}")

    if args.mutant:
        apply_mutant(args.mutant)
        print(f"⚠ 已注入伪证 mutant：{args.mutant}")

    print("\n── 第 2 步：Q1 claim suite ──")
    Q1.reset_run_state()
    outs = [run_claim(c, lanes[Q1.lane_of(c)], shrink=not args.no_shrink)
            for c in claims]

    if args.mutant:
        restore_geometry()
        caught = [o.claim.name for o in outs if not o.ok]
        print()
        if caught:
            print(f"✓ mutant `{args.mutant}` 被抓：{len(caught)} 条 claim 变红")
            for nm in caught:
                print(f"    - {nm}")
            print("测试台**能失败**——结论是真的测出来的，不是碰巧打印 PASS。")
            print("退出码 1（符合计划验收标准：每个 mutant 都必须退出 1）")
            return 1
        print(f"✗ mutant `{args.mutant}` **逃逸**：suite 依然全绿。")
        print("  这是测试台的盲区，必须先补 claim 再谈结论。")
        print("退出码 0（⚠ 这个 0 是坏消息）")
        return 0

    _print_table(outs)
    _print_failures(outs)
    if not all(o.ok for o in outs):
        rc = 1
    print("全部通过 ✓" if rc == 0 else "存在失败 ✗")

    # 3) 统计与扫描
    if args.stats == "omega":
        _print_stats(stats_omega(n=2000 if args.quick else 20000, seed=args.seed))

    if args.sweep == "gamma":
        rows = sweep_gamma(args.seed)
        cover = sum(r["cover"] for r in rows)
        act = sum(r["omega_active"] for r in rows)
        print(f"γ 扫描：{len(rows)} 条，覆盖 {cover}，不覆盖 {len(rows) - cover}，"
              f"其中 Ω 活化 {act}")
        if args.csv:
            os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            write_csv(args.csv, rows)
            print(f"已写入 {args.csv}")

    if args.sweep == "n":
        rows_n = sweep_n(per_n=500 if args.quick else 4000, seed=args.seed)
        _print_sweep_n(rows_n, "uniform")
        if args.csv:
            os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            write_csv(args.csv, rows_n)
            print(f"已写入 {args.csv}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
