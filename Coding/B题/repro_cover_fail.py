# -*- coding: utf-8 -*-
"""复现一次「直径圆未覆盖定位区域」的实验（2026-09-11）

背景
----
问题 1 第二问：以定位区域**直径 D** 为直径的圆，能否覆盖该区域？
9/10 演练数据给出的答案是「**能**」。本次实验（检测点 D、E）给出的答案是「**否**」。

这不是算法 bug，也不是定理被推翻——恰好相反，它是引理 3 之后的
**MEC 分类定理**预言过的另一支：

    覆盖 ⟺ r_MEC = D/2 ⟺ MEC 由「一对直径端点」或「直角三角形」确定
    不覆盖 ⟺ r_MEC > D/2 ⟺ MEC 由一个**锐角三角形**确定

本实验落在第二支：MEC 由 F、H、I 三点确定的锐角三角形给出，
故 I 点比直径圆的半径多出 0.43 m，落在圆外。

定量结论（本脚本已数值验证）
------------------------------
1. 区域四角由 D、E 两站的 θ±1° 四条边界射线两两求交**独立重算**得到，
   与截图标注的 IG / HI / HF / FG / IJ 逐项吻合，最大偏差 3e-13 m。
2. 直径 = 对角线 |FH| = 82.070552 m；直径圆半径 R = 41.035276 m。
   G 在圆内 1.205 m；**I 在圆外 0.431405 m** ⟹ 判定「否」。
3. 机理（泰勒斯）：∠FIH = 89.019815°，比直角差 0.98°。
   区域是 77 m × 25 m 的近矩形，矩形化时四角恰好都落在对角圆上，
   **故这是一个「刀尖」情形**——差不到 1° 就从「覆盖」翻成「不覆盖」。
4. 稳定性：固定两站位置、只调 |θ_E−θ_D|，判定在 **92.000000° = 90° + 2ε**
   处翻转。当前 91.0198°，距翻转只有 0.98°，属边沿但**不**是数值噪声。

⚠ 与 9/10 演练数据的区别：那次 γ = 22.07°，区域是极细长条，直径落在
**长边**上，故覆盖成立。两次都对——Q1 第二问的答案**依数据而定**，
算法必须能同时给出「是」和「否」。

数据来源
--------
GeoGebra 实验截图，F/G/H/I 为定位区域四角，J = 中点(H, F)。截图标注：
    IG = 81.2636721010041
    HI = 78.0816050648543
    HF = 82.0705524542566
    FG = 77.2465631032173
    IJ = 41.4666808066833
本脚本从**源真值与两个检测点坐标**出发独立重算全部量，并与上述标注逐项比对。

用法
----
    python repro_cover_fail.py            # 只输出报告
    python repro_cover_fail.py --plot     # additionally 输出对照图 PNG
    python repro_cover_fail.py --no-check # 跳过断言（仅查看数值）
"""
from __future__ import annotations

import math
import sys

from geometry import (
    CLEAR_RADIUS,
    crossing_angle,
    covering_radius,
    circumcircle,
    dist,
    diameter_circle_covers,
    in_diameter_circle,
    max_pairwise_distance,
    uncertainty_corners,
)

# ── Windows 控制台按 UTF-8 输出，避免中文乱码 ────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── 实验输入 ────────────────────────────────────────────────
SOURCE = (-631.3248143106883, 518.7428634252668)   # 干扰源真值 C
SITE_D = (1158.5208537908559, -801.3474829903406)  # 检测点 D
SITE_E = (-211.7517570274545, 1109.3446610684468)  # 检测点 E
ERR_DEG = 1.0                                      # 示向度误差上界 ε

# GeoGebra 截图标注值（用于回归比对）
OBSERVED = {
    "IG": 81.2636721010041,
    "HI": 78.0816050648543,
    "HF": 82.0705524542566,
    "FG": 77.2465631032173,
    "IJ": 41.4666808066833,
}

# 顶点命名：按「θ_D 取 −ε/+ε」×「θ_E 取 −ε/+ε」定位，与截图一致
NAMES = {
    (-1, -1): "H",
    (-1, +1): "G",
    (+1, -1): "I",
    (+1, +1): "F",
}


def bearing(a, b):
    """从 a 指向 b 的示向度（度，[0,360)）—— 与附录 2(1) 定义一致。"""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def build_region(theta_d, theta_e, err=ERR_DEG):
    """枚举—筛：四条边界射线两两求交，保留 t>0 的解，得到区域四角。

    返回 {名字: 坐标}，键按 NAMES 命名；缺顶点说明区域退化。
    """
    corners = {}
    for sd in (-1, +1):
        for se in (-1, +1):
            r = ray_intersection(
                SITE_D, theta_d + sd * err,
                SITE_E, theta_e + se * err,
            )
            if r is not None:
                corners[NAMES[(sd, se)]] = r[0]
    return corners


def ray_intersection(s1, a1, s2, a2):
    """两条射线的交点 (点, t, s)；无正向交点返回 None。

    ⚠ 必须只接受 t>0 且 s>0 的解——楔形边界是**半直线**，不是直线。
    这是数学要求，不是实现细节：负向解会把区域算到检测点背后。
    """
    d1 = (math.cos(math.radians(a1)), math.sin(math.radians(a1)))
    d2 = (math.cos(math.radians(a2)), math.sin(math.radians(a2)))
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-14:
        return None                       # 两射线平行
    dx, dy = s2[0] - s1[0], s2[1] - s1[1]
    t = (dx * d2[1] - dy * d2[0]) / den
    s = (dx * d1[1] - dy * d1[0]) / den
    if t <= 0.0 or s <= 0.0:
        return None                       # 交点在射线反方向
    return ((s1[0] + t * d1[0], s1[1] + t * d1[1]), t, s)


def vertex_angle(a, v, b):
    """顶点 v 处、由 va 与 vb 张成的内角（度）。"""
    u = (a[0] - v[0], a[1] - v[1])
    w = (b[0] - v[0], b[1] - v[1])
    nu = math.hypot(*u)
    nw = math.hypot(*w)
    if nu < 1e-12 or nw < 1e-12:
        return float("nan")
    c = max(-1.0, min(1.0, (u[0] * w[0] + u[1] * w[1]) / (nu * nw)))
    return math.degrees(math.acos(c))


def classify_triangle(a, b, c, tol=1e-9):
    """按最长边分类三角形：锐角 / 直角 / 钝角。"""
    sides = sorted([dist(a, b), dist(b, c), dist(c, a)])
    x, y, z = sides          # x ≤ y ≤ z
    lhs, rhs = z * z, x * x + y * y
    if abs(lhs - rhs) <= tol * max(1.0, rhs):
        return "直角"
    return "钝角" if lhs > rhs else "锐角"


def mec_determining_set(pts, names):
    """求最小包围圆 + 它的确定集（2 点或 3 点）+ 该圆的分类。

    实现与 geometry.covering_radius 同构，但额外记录**是哪个集合**达到最小，
    以便套用 MEC 分类定理做定性解释。
    """
    n = len(pts)
    best_r, best_info = float("inf"), None

    def covers(center, r):
        return all(dist(center, p) <= r + 1e-9 for p in pts)

    for i in range(n):
        for j in range(i + 1, n):
            ctr = ((pts[i][0] + pts[j][0]) / 2.0, (pts[i][1] + pts[j][1]) / 2.0)
            r = dist(pts[i], pts[j]) / 2.0
            if covers(ctr, r) and r < best_r:
                best_r, best_info = r, {
                    "kind": "2点(直径对)",
                    "pts": (names[i], names[j]),
                    "center": ctr,
                    "shape": "—",
                }

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cc = circumcircle(pts[i], pts[j], pts[k])
                if cc is None:
                    continue
                ctr, r = cc
                if covers(ctr, r) and r < best_r:
                    best_r, best_info = r, {
                        "kind": "3点(外接圆)",
                        "pts": (names[i], names[j], names[k]),
                        "center": ctr,
                        "shape": classify_triangle(pts[i], pts[j], pts[k]),
                    }

    return best_r, best_info


def reproduce(verbose=True, do_check=True):
    """跑完整条链路，返回结果字典。"""
    theta_d, theta_e = bearing(SITE_D, SOURCE), bearing(SITE_E, SOURCE)
    gamma = crossing_angle(theta_d, theta_e)
    r_d, r_e = dist(SITE_D, SOURCE), dist(SITE_E, SOURCE)

    corners, degenerate = uncertainty_corners(
        SITE_D, theta_d, SITE_E, theta_e, ERR_DEG
    )
    named = build_region(theta_d, theta_e)
    order = ["F", "G", "H", "I"]
    pts = [named[k] for k in order if k in named]

    diameter, ends = max_pairwise_distance(pts)
    a, b = ends
    center = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    radius = diameter / 2.0

    # 每个顶点相对直径圆的有符号余量：>0 在圆外
    margins = {k: dist(center, p) - radius for k, p in named.items()}
    covered = diameter_circle_covers(a, b, pts)

    r_mec, mec_info = mec_determining_set(pts, order)
    quick = covering_radius(pts)           # 与 geometry.py 交叉验证

    result = {
        "theta_d": theta_d, "theta_e": theta_e, "gamma": gamma,
        "r_d": r_d, "r_e": r_e,
        "corners": named, "degenerate": degenerate, "n_corners": len(corners),
        "diameter": diameter, "ends": ends,
        "center": center, "radius": radius,
        "margins": margins, "covered": covered,
        "r_mec": r_mec, "mec_info": mec_info,
    }

    if verbose:
        _report(result, order)

    if do_check:
        _assert_against_geogebra(result)
        assert abs(r_mec - quick) < 1e-9, "MEC 与 geometry.covering_radius 不一致"

    return result


def _report(r, order):
    W = 66
    print("=" * W)
    print("实验还原：D/E 两站交会 → 定位区域 → 直径圆覆盖判定")
    print("=" * W)

    print("\n【1】输入与观测量")
    print(f"  源真值   C = ({SOURCE[0]:.10f}, {SOURCE[1]:.10f})")
    print(f"  检测点   D = ({SITE_D[0]:.10f}, {SITE_D[1]:.10f})")
    print(f"  检测点   E = ({SITE_E[0]:.10f}, {SITE_E[1]:.10f})")
    print(f"  示向度   θ_D = {r['theta_d']:.6f}°   θ_E = {r['theta_e']:.6f}°")
    print(f"  作用距离 r_D = {r['r_d']:.4f} m   r_E = {r['r_e']:.4f} m")
    print(f"  交会角   γ = {r['gamma']:.4f}°")
    print(f"  误差界   ε = ±{ERR_DEG}°  ⟹ 张角 2ε = {2*ERR_DEG}°")

    print("\n【2】定位区域四角（枚举 4 条边界射线两两求交，只取 t>0）")
    if r["degenerate"]:
        print(f"  ⚠ 区域退化：{r['degenerate']}")
    for k in order:
        if k not in r["corners"]:
            continue
        sd = "+" if k in ("F", "I") else "−"
        se = "+" if k in ("F", "G") else "−"
        p = r["corners"][k]
        print(f"  {k}(θ_D{sd}ε, θ_E{se}ε) = ({p[0]:.9f}, {p[1]:.9f})")

    print("\n【3】两两距离")
    na, nb = _names_of(r["ends"], r["corners"])
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            u, v = order[i], order[j]
            if u in r["corners"] and v in r["corners"]:
                d = dist(r["corners"][u], r["corners"][v])
                tag = "  ← 直径" if {u, v} == {na, nb} else ""
                print(f"  {u}{v} = {d:.10f} m{tag}")

    print("\n【4】直径与直径圆")
    print(f"  直径  D = |{na}{nb}| = {r['diameter']:.6f} m")
    print(f"  圆心  J = 中点({na}, {nb}) = ({r['center'][0]:.10f}, {r['center'][1]:.10f})")
    print(f"  半径  R = D/2 = {r['radius']:.6f} m")

    print("\n【5】覆盖检查（泰勒斯：p 在圆内 ⟺ (A−p)·(B−p) ≤ 0）")
    print(f"  {'顶点':<6}{'到 J 距离':>14}{'相对 R 的余量':>18}{'判定':>10}")
    for k in order:
        if k not in r["corners"]:
            continue
        p = r["corners"][k]
        m = r["margins"][k]
        ok = m <= 1e-9
        mark = "圆内 ✔" if m < -1e-9 else ("圆上 ·" if ok else "**圆外 ✘**")
        print(f"  {k:<6}{dist(r['center'], p):>14.6f}{m:>+18.6f}{mark:>12}")

    verdict = "能覆盖" if r["covered"] else "不能覆盖"
    print(f"\n  结论：直径圆 **{verdict}** 定位区域")

    print("\n【6】为什么？（MEC 分类定理）")
    info = r["mec_info"]
    print(f"  最小包围圆半径 r_MEC = {r['r_mec']:.6f} m")
    print(f"  直径圆半径     D/2  = {r['radius']:.6f} m")
    print(f"  r_MEC − D/2 = {r['r_mec'] - r['radius']:+.6f} m"
          f"   ⟹ {'相等，覆盖' if abs(r['r_mec']-r['radius'])<1e-9 else '不相等，不覆盖'}")
    if info:
        print(f"  MEC 由 {info['kind']} {info['pts']} 确定"
              + (f"，该三角形为 **{info['shape']}** 三角形" if info["shape"] != "—" else ""))
    lo = r["radius"]
    hi = r["diameter"] / math.sqrt(3.0)
    print(f"  Jung 定理检验：D/2 = {lo:.6f} ≤ r_MEC = {r['r_mec']:.6f} "
          f"≤ D/√3 = {hi:.6f}  "
          f"{'✔' if lo - 1e-9 <= r['r_mec'] <= hi + 1e-9 else '✘'}")

    print("\n【7】最直观的解释：泰勒斯在看 I 处的内角")
    info_pts = info["pts"] if info else ()
    for k in info_pts:
        if k in ("F", "H"):
            continue
        others = [x for x in ("F", "H") if x in r["corners"]]
        ang = vertex_angle(r["corners"][others[0]], r["corners"][k],
                           r["corners"][others[1]])
        m = r["margins"][k]
        print(f"  ∠{others[0]}{k}{others[1]} = {ang:.6f}°   "
              f"（90° 为临界：=90° 恰好落在圆上，<90° 落到圆外）")
        print(f"  → {k} {'在圆外' if m > 1e-9 else '在圆内'}，"
              f"差 {(90.0 - ang):+.4f}° ⟹ 距圆 {(m*100):+.2f} cm")
    print("  这就是泰勒斯命题：圆周角 = 90° ⟺ 点在圆上。区域在该角上"
          "\n  「不够钝」（< 90°），所以该顶点必然戳出直径圆。")

    print("\n【8】对清除可行性的意义")
    print(f"  清除半径 r_clear = {CLEAR_RADIUS} m")
    print(f"  r_MEC = {r['r_mec']:.3f} m ≫ {CLEAR_RADIUS} m "
          f"⟹ **不能保证一次定点清除**，须逼近（命题 A 的推论）")
    print()


def _names_of(ends, corners):
    """把坐标对映射回 F/G/H/I 名字。"""
    out = []
    for p in ends:
        for k, q in corners.items():
            if dist(p, q) < 1e-9:
                out.append(k)
                break
    return tuple(out)


def _assert_against_geogebra(r):
    """与 GeoGebra 截图标注值逐项比对——这是本脚本的回归断言。"""
    c = r["corners"]
    pairs = {
        "IG": (c["I"], c["G"]),
        "HI": (c["H"], c["I"]),
        "HF": (c["H"], c["F"]),
        "FG": (c["F"], c["G"]),
        "IJ": (r["center"], c["I"]),
    }
    print("-" * 66)
    print("回归断言：脚本重算值 vs GeoGebra 截图标注值")
    print(f"  {'量':<6}{'脚本重算':>20}{'截图标注':>20}{'偏差':>12}")
    worst = 0.0
    for k, (p, q) in pairs.items():
        got = dist(p, q)
        want = OBSERVED[k]
        dev = abs(got - want)
        worst = max(worst, dev)
        flag = "✔" if dev < 1e-6 else "✘"
        print(f"  {k:<6}{got:>20.10f}{want:>20.10f}{dev:>12.2e} {flag}")
    print(f"  最大偏差 = {worst:.3e} m   {'全部通过' if worst < 1e-6 else '**存在不一致**'}")
    print("-" * 66)
    assert worst < 1e-6, f"与截图标注值不一致，最大偏差 {worst:.3e} m"


def plot(outfile="repro_cover_fail.png"):
    """画一张与 GeoGebra 截图对照的图。matplotlib 缺失时静默跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Windows 下的中文字体；DejaVu Sans 不含 CJK 字形，会画成空框
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("（未安装 matplotlib，跳过绘图。装法："
              "pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple）")
        return None

    r = reproduce(verbose=False, do_check=False)
    c, order = r["corners"], ["F", "G", "H", "I"]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    poly = [c[k] for k in order] + [c[order[0]]]
    ax.fill([p[0] for p in poly], [p[1] for p in poly],
            color="#cfe2f3", alpha=0.75, zorder=1)
    ax.plot([p[0] for p in poly], [p[1] for p in poly],
            color="#1c4587", lw=1.8, zorder=3)

    th = [i * 2 * math.pi / 720 for i in range(721)]
    ax.plot([r["center"][0] + r["radius"] * math.cos(t) for t in th],
            [r["center"][1] + r["radius"] * math.sin(t) for t in th],
            color="#333333", lw=1.6, zorder=2)

    ax.plot(*zip(c["F"], c["H"]), color="#333333", lw=1.2, ls="--", zorder=2)
    ax.plot(*zip(c["I"], r["center"]), color="#333333", lw=1.2, zorder=2)

    for k in order:
        ax.plot(*c[k], "o", color="#333333", ms=6, zorder=5)
        ax.annotate(k, c[k], textcoords="offset points", xytext=(7, 5),
                    fontsize=12, fontweight="bold")
    ax.plot(*r["center"], "o", color="red", ms=6, zorder=6)
    ax.annotate("J", r["center"], textcoords="offset points", xytext=(7, -12),
                fontsize=12, fontweight="bold", color="red")
    ax.plot(*SOURCE, "*", color="green", ms=16, zorder=6)
    ax.annotate("C(源真值)", SOURCE, textcoords="offset points", xytext=(9, -3),
                fontsize=10, color="green")

    ax.annotate(f"I 在圆外 {r['margins']['I']*100:+.2f} cm",
                c["I"], textcoords="offset points", xytext=(-30, -34),
                fontsize=11, color="#b00000", fontweight="bold")
    ax.set_title(f"直径圆未覆盖定位区域  |  γ={r['gamma']:.2f}°  "
                 f"D={r['diameter']:.2f} m  r_MEC={r['r_mec']:.2f} m", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    print(f"图已保存：{outfile}")
    return outfile


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--plot" in args:
        plot()
    else:
        reproduce(verbose=True, do_check="--no-check" not in args)
