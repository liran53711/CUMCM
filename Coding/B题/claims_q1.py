"""
问题 1 的结论注册表 —— 21+ 条 claim。

每条 claim 的 `source` 指向它在文档里的出处（清单#n / 引理 n / 命题 n），
**没有出处的 claim 不许留在论文里**。`expect="refuted"` 的 claim 必须报出反例，
它们记录的是「已被证伪、不得写进论文」的猜想。

设计要点（都是本项目踩出来的）：
  · 窗口定理的 claim 必须限定在 **Ω 不活化** 域内；Ω 活化时它是**错的**，
    故另有一条 `expect="refuted"` 的 claim 把这件事钉在案上。
  · $n=2$ 的 `localize()` 是 **Ω-free** 的，所以 `n2_bit_identical` 也只在
    Ω 不活化时可比。
  · γ 一律用**未折叠**值：`crossing_angle` 把它折到 [0,90]，喂不进窗口定理。

用法：
    python verify_q1.py                 # 跑全套
    python verify_q1.py --only cover_band
"""

from __future__ import annotations

import math
import random

import geometry
from claim_tester import (
    ARENA_R,
    SKIP,
    Claim,
    Instance,
    R_EFF_MAX,
    R_EFF_MIN,
    _point_in_poly_strict as point_in_poly,
    sample_instance,
)

Point = tuple[float, float]

# ── 冻结常量（回归锚点，改动必须是有意的）──────────────────────
# 数据集 A：2026-09-10 演练（S1/S2 两点交会）
DS_A = dict(s1=(300.0, 400.0), th1=214.82, s2=(300.0, 0.0), th2=192.75)
D_A = 175.81826749750502
R_A = 87.90913374875251

# 数据集 B：2026-09-11 演练（D/E），与 repro_cover_fail.py 的 GeoGebra 复现同源。
# ⚠ 见 `问题1-覆盖判据的适用域（Ω 修正）.md` §7.1：|D−C| = 2224 m 越出 r_eff 上界，
#   该算例**与 Ω 不相容**，只能在 Ω-free 的意义上引用。
DS_B = dict(s1=(1158.5208537908559, -801.3474829903406),
            th1=143.5895840236657,
            s2=(-211.7517570274545, 1109.3446610684468),
            th2=234.6093987107173)
D_B = 82.0705524542566          # 与 repro_cover_fail.OBSERVED["HF"] 同源

_EPS_TOL = 1e-6


# ── 工具 ────────────────────────────────────────────────────
def raw_gamma(t1: float, t2: float) -> float:
    """**未折叠**交会角，取值 [0°,180°]。

    ⚠ `geometry.crossing_angle` 折叠到 [0,90]：对 raw>90 返回 180−raw。
    窗口定理**必须**用本函数，用 `crossing_angle` 会静默算错
    （`repro_cover_fail.py` docstring 写 92° 而折叠值 88.98° 就是这个坑）。
    """
    return abs(geometry.norm180(t1 - t2))


def region_of(i: Instance, **kw):
    kw.setdefault("r_eff", i.r_eff)
    return geometry.localization_region(i.sites, i.bearings, i.err_deg, **kw)


def omega_certified_inactive(i: Instance, arena_r: float = ARENA_R) -> bool:
    """**独立重算**的 Ω 不活化证书。不要相信 `Region.omega_active`。

    为什么必须重算：`omega_active` 是 geometry 自己报的。一个「无脑强制裁剪」
    的实现错（`mutant_omega_always_clipped` 就是）会让它报 True，于是所有
    拿它当门禁的 claim 统统 SKIP —— mutant 就此**逃逸**（实测确实逃逸过）。
    证书由测试台自己按定义重算，几何想谎报也谎报不了。

    ⚠ **两个条件缺一不可**，只查余量会误判：
      1. 楔形之交**有界**（`wedge_bounded`）。γ < 2ε 时交会区域沿一条公共射线
         无限延伸，此时 `region_corners` 给的那几个角可能碰巧落在 Ω 内、
         余量为正——但区域本身是发散的，Ω 是**唯一**让它有界的东西，
         必然活化。这正是 tier 1 的门禁写作 `closed and len(poly) >= 3 and ...`
         的原因（实测踩到：γ=1.18°、基线仅 24.5 m 的一例让本 claim 误报）。
      2. 余量 > 0。
    """
    if not geometry.wedge_bounded(i.bearings, i.err_deg):
        return False
    corners = independent_corners(i)
    if len(corners) < 3:
        return False
    from claim_tester import oracle_omega_margin

    return oracle_omega_margin(corners, i.sites, arena_r, i.r_eff) > 0.0


def independent_corners(i: Instance) -> list[Point]:
    """不经过 `wedge_ok` 的候选角点。

    ⚠ 为什么不能用 `region_corners`：它内部就是用 `wedge_ok` 筛的。
    一个「叉积次序反了」的 mutant 会让它**返回空表**，于是以它为输入算出来的
    Ω 余量证书也跟着变成「不可用」——证书与被测代码同源就失去了独立性，
    mutant 顺势逃逸（实测逃逸过）。
    `uncertainty_corners` 只用 `ray_intersection`、完全不碰 `wedge_ok`，
    对 n=2 是现成的独立来源；n>2 没有等价的旧函数，退回 `region_corners`
    （此时该 mutant 由 `oracle_containment` 的 (b) 分支兜住）。
    """
    if i.n == 2:
        pts, why = geometry.uncertainty_corners(
            i.sites[0], i.bearings[0], i.sites[1], i.bearings[1], i.err_deg)
        if not why and len(pts) >= 3:
            return list(pts)
    return geometry.region_corners(i.sites, i.bearings, i.err_deg)


def near_site(q: Point, sites: list[Point], tol: float = 1e-6) -> bool:
    """顶点是否与某个站点**数值重合**（角度在此处无意义）。

    实测来源：γ=0.0688° 的近共线构形下，两站与源几乎共线，于是从 S₂ 指向
    源的射线几乎正过 S₁，交会四角的其中一个**落在 S₁ 上**（实测距离 7.6e-13 m）。
    此时「从 S₁ 看该顶点的方位角」是 0/0，atan2 给出 177.44° 的噪声——
    这不是几何错了，是**这个量在该点没有定义**。故此处一律 SKIP。
    容差取 1e-6 m：比坐标浮点分辨率（~2e-13）高 6 个数量级，
    比任何有物理意义的长度低 6 个数量级。
    """
    return any(geometry.dist(q, s) <= tol for s in sites)


def _idx(seq: list[Point], p: Point) -> int | None:
    for k, q in enumerate(seq):
        if q == p:
            return k
    return None


def _grid_inside(poly: list[Point], k: int = 24) -> list[Point]:
    """在 poly 的包围盒上打 k×k 网格，返回落在多边形内部的点。"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    sx = (x1 - x0) / (k - 1) if k > 1 else 0.0
    sy = (y1 - y0) / (k - 1) if k > 1 else 0.0
    return [
        (x0 + a * sx, y0 + b * sy)
        for a in range(k) for b in range(k)
        if point_in_poly(poly, (x0 + a * sx, y0 + b * sy))
    ]


def _edge_samples(poly: list[Point], m: int) -> tuple[list[Point], float]:
    """在每条边的**内部**取 m 个采样点（不含顶点），返回 (点, 最长边长)。

    这是能被**证明**的 oracle 构造。凸多边形的直径必在顶点处取得，故
    对任意一对顶点 $a,b$，各取所在边上最近的内部采样 $a',b'$，有

        𝒟 − (|e_a|+|e_b|)/(2m)  ≤  |a'b'|  ≤  𝒟

    左端因为 $a',b'$ 都在（闭）区域上所以是**区域内的点对**，右端是直径的定义。
    于是夹逼是**单边可证**的：采样值永不超真值，且亏空 ≤ 最长边/m。
    """
    pts: list[Point] = []
    longest = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        e = geometry.dist(p, q)
        longest = max(longest, e)
        if e <= 0.0:
            continue
        for k in range(m):
            t = (k + 0.5) / m
            pts.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return pts, longest


# ── 实例族 ──────────────────────────────────────────────────
def q1_instances(n_random: int = 240, seed: int = 20260911) -> list[Instance]:
    """物理自洽实例族：随机 n=2（三种 err_mode）+ 手工窄交会角 + 一般 n。

    ⚠ 窄 γ 必须显式包含：Ω 最容易咬的就是近于平行的两楔形，
    纯随机采样几乎抽不到 γ < 10°。
    """
    rng = random.Random(seed)
    out: list[Instance] = []

    # 1) 随机 n=2，三种误差模式
    for mode in ("zero", "uniform", "extreme"):
        k = 0
        while k < n_random // 3:
            i = sample_instance(rng, n=2, err_mode=mode)
            if i is not None:
                out.append(i)
                k += 1

    # 2) 手工指定交会角：源在原点，两站距源 d、角距恰为 g，故 γ = g 精确可控。
    #    r_eff 取 1500 的**上界**，使 |S−G| = d 离边界留足余量——否则 γ≈90°
    #    的近垂直构形会紧贴 Ω 边界，Ω 活化/不活化在一线之间，coverage 全丢。
    #    90° 与 92° 是窗口的**刀刃**，留着是为了让 `cover_band` 走 SKIP 分支。
    alpha, d, reff = 0.7, 1100.0, R_EFF_MAX
    for g_deg in (2.5, 3.0, 4.0, 6.0, 10.0, 20.0, 30.0, 45.0, 60.0, 75.0,
                  85.0, 88.0, 89.0, 89.5, 90.0, 90.5, 91.0, 91.5, 92.0, 92.5,
                  93.0, 94.0, 96.0, 100.0, 120.0, 150.0, 175.0):
        a, g = math.radians(alpha), math.radians(g_deg)
        sites = [(d * math.cos(a - g / 2), d * math.sin(a - g / 2)),
                 (d * math.cos(a + g / 2), d * math.sin(a + g / 2))]
        bs = [math.degrees(math.atan2(-s[1], -s[0])) % 360.0 for s in sites]
        out.append(Instance(sites=sites, bearings=bs, err_deg=1.0, r_eff=reff,
                            source=(0.0, 0.0), deltas=[0.0, 0.0],
                            note=f"指定γ={g_deg}°"))

    # 3) 一般 n
    for n in (3, 4, 5, 6, 8):
        k = 0
        while k < 24:
            i = sample_instance(rng, n=n, err_mode="uniform")
            if i is not None:
                out.append(i)
                k += 1
    return out


def q1_frozen_instances() -> list[Instance]:
    """只有两个冻结数据集——给 `n2_regression_frozen` 专用的一条泳道。

    单独开道的原因：数据集 B 的两个站点压根不落在同一个源的有效接收半径里
    （§7.1），把它混进物理族会让别的 claim 拿到不自洽的输入。
    """
    return [
        Instance(sites=[DS_A["s1"], DS_A["s2"]],
                 bearings=[DS_A["th1"], DS_A["th2"]],
                 err_deg=1.0, r_eff=R_EFF_MAX, note="冻结数据集 A（9/10）"),
        Instance(sites=[DS_B["s1"], DS_B["s2"]],
                 bearings=[DS_B["th1"], DS_B["th2"]],
                 err_deg=1.0, r_eff=R_EFF_MAX, note="冻结数据集 B（D/E，Ω-free）"),
    ]


def q1_refutation_instances(n_random: int = 4000, seed: int = 5150) -> list[Instance]:
    """**只在 `expect="refuted"` 的 claim 上跑**的族。

    引理 M2（「直径必在对角线上」）在本题域内成立、在一般域内**不**成立，
    故要被证伪就必须跑一般域（$\varepsilon$ 可达 10°、基线比不对称）。
    这批实例**不满足** B 题的物理环带——它们是纯几何反例。
    """
    rng = random.Random(seed)
    out: list[Instance] = []
    while len(out) < n_random:
        eps = rng.choice((0.5, 1.0, 2.0, 5.0, 10.0))
        d1 = rng.uniform(300.0, 1600.0)
        d2 = d1 / rng.uniform(1.0, 8.0)
        phi = rng.uniform(0.0, 2 * math.pi)
        g = math.radians(rng.uniform(5.0, 175.0))
        s1 = (d1 * math.cos(phi), d1 * math.sin(phi))
        s2 = (d2 * math.cos(phi + g), d2 * math.sin(phi + g))
        if math.dist(s1, s2) < 1e-6:
            continue
        bs = [math.degrees(math.atan2(-s[1], -s[0])) % 360.0 for s in (s1, s2)]
        out.append(Instance(sites=[s1, s2], bearings=bs, err_deg=eps,
                            r_eff=R_EFF_MAX, source=(0.0, 0.0),
                            deltas=[0.0, 0.0], note=f"一般域 ε={eps}"))
    return out


# ── claim #1–#6：核心不变量 ──────────────────────────────────
# `cover_r` 的预算。**这不是随手放的容差，是一个已定位的发现。**
#
# `Region.polygon` 是 `convex_hull()` 的输出，`Localization.corners` 是**同一个
# 多重集**的另一个循环起点（有时还反向）。既然多重集相同，`cover_r` 似乎也该相同——
# 但 `covering_radius` / `min_enclosing_circle` 都是**顺序相关**的：先对后三元组的
# 枚举遇上「两组候选半径几乎并列」时，换个起点就挑中另一组，浮点又不可结合。
#
# 实测（107 个 Ω 不活化、多重集逐位相同的可比实例）：
#     min_enclosing_circle   两种顺序  max 相对偏差 1.291e-16  （1 ULP）
#     covering_radius        两种顺序  max 相对偏差 1.291e-16  （1 ULP）
#     Welzl                   两种顺序  max 相对偏差 1.522e-14
#     跨实现（mec(polygon) vs covering_radius(corners)）同上量级
# 另有 1 例达 6.1e-13（只出现在 48 族里——`q1_instances` 每个 err_mode 各抽
# n_random//3 个，**共用一条 RNG 流**，所以 48 族与 120 族不是前缀关系）。
#
# 结论：要求严格 `==` 是在测浮点结合律，不是在测几何——那个门禁永远红。
# 所以几何量（𝒟、覆盖判定、顶点数、顶点多重集）仍要求**逐位相同**，
# `cover_r` 给 1e-11 相对预算（比实测最坏值高 5 个数量级，比 20 m 清除半径
# 低 9 个数量级）。**预算一旦被顶破，就是几何真的变了，不是舍入。**
_UR_TOL = 1e-11


def _n2_bit_identical(i: Instance) -> str | None:
    """n=2 且 Ω 不活化时，新实现与 `localize()` 一致。

    Ω 活化时不可比——`localize()` 是 **Ω-free** 的，两者**本来就该不同**。

    逐位（严格 `==`）的部分：𝒟、覆盖判定、顶点多重集、顶点数。
    `cover_r` 用 4 ULP 预算，理由见 `_UR_TOL` 上方那段。
    """
    if i.n != 2:
        return SKIP
    if not omega_certified_inactive(i):     # 门禁用独立证书，不用 r.omega_active
        return SKIP
    r = region_of(i)
    if not r.bounded:
        return SKIP
    loc = geometry.localize(i.sites[0], i.bearings[0],
                            i.sites[1], i.bearings[1], i.err_deg)
    if not loc.bounded:
        return SKIP
    if r.diameter != loc.diameter:
        return f"𝒟 不逐位相同：{r.diameter!r} vs {loc.diameter!r}"
    if r.diameter_circle_ok != loc.diameter_circle_ok:
        return f"覆盖判定不同：{r.diameter_circle_ok} vs {loc.diameter_circle_ok}"
    if r.m != len(loc.corners):          # Localization 没有 m 字段，用 len(corners)
        return f"顶点数不同：{r.m} vs {len(loc.corners)}"
    a = sorted(r.polygon)
    b = sorted(loc.corners)
    if a != b:
        return f"顶点多重集不逐位相同（差 {len(set(a) ^ set(b))} 个）"
    reld = abs(r.cover_r - loc.cover_r) / max(abs(r.cover_r), 1.0)
    if reld > _UR_TOL:
        return (f"cover_r 相对偏差 {reld:.3e} > 4 ULP = {_UR_TOL:.3e}"
                f"（{r.cover_r!r} vs {loc.cover_r!r}）")
    return None


def _corner_back_substitution(i: Instance) -> str | None:
    """清单#1：每个顶点回代各站，方位角与 θ_i 之差 ≤ ε。

    与站点数值重合的顶点跳过（角度在那里无定义，见 `near_site`）。
    """
    r = region_of(i)
    if not r.bounded:
        return SKIP
    tested = 0
    for q in r.polygon:
        for s, b in zip(i.sites, i.bearings):
            if near_site(q, [s]):
                continue
            tested += 1
            off = abs(geometry.norm180(
                math.degrees(math.atan2(q[1] - s[1], q[0] - s[0])) - b))
            if off > i.err_deg + _EPS_TOL:
                return f"顶点 {q} 相对站 {s} 偏差 {off:.6f}° > ε={i.err_deg}°"
    """若所有顶点都退化则本 claim 无话可说——不许当成通过。"""
    return None if tested else SKIP


def _convexity(i: Instance) -> str | None:
    """清单#3：区域必须 CCW 凸、面积为正。"""
    r = region_of(i)
    if not r.bounded:
        return SKIP
    if not r.convex:
        return "区域不是 CCW 凸多边形"
    if r.area <= 0.0:
        return f"面积非正：{r.area!r}"
    return None


def _cover_iff_mec_eq_half_diameter(i: Instance) -> str | None:
    """清单#4 + 等价命题：覆盖 ⟺ r_MEC = 𝒟/2。

    ⚠ 容差必须**紧**。这里是**严格等价**，不是近似关系：区域被直径圆覆盖
    ⟺ 直径圆就是一个包围圆 ⟹ r_MEC ≤ 𝒟/2，与命题 A 的 𝒟/2 ≤ r_MEC 夹出相等；
    反向，𝒟/2 = r_MEC 时直径对是 MEC 的一条直径弦 ⟹ 中点即圆心 ⟹ 覆盖。
    早先用 1e-6·𝒟 的松容差，会把 𝒟/2 < r_MEC 只差 4e-9 的实例误判成「相等」，
    于是 lhs=False / rhs=True 报假失败——**松容差是在掩盖真判据**。
    """
    r = region_of(i)
    if not r.bounded or r.diameter <= 0:
        return SKIP
    gap = r.cover_r - r.diameter / 2
    tol = 1e-12 * max(1.0, r.diameter)
    if gap < -tol:
        return (f"r_MEC − 𝒟/2 = {gap:.3e} < 0：违反命题 A"
                f"（r_MEC={r.cover_r!r}, 𝒟={r.diameter!r}）")
    want = gap <= tol                      # 等价 ⟹ 覆盖 ⟺ r_MEC == 𝒟/2
    if r.diameter_circle_ok != want:
        return f"覆盖={r.diameter_circle_ok} 但 r_MEC−𝒟/2={gap:.3e}（tol={tol:.1e}）"
    return None


def _jung_bounds(i: Instance) -> str | None:
    """命题 A + Jung：𝒟/2 ≤ r_MEC ≤ 𝒟/√3。"""
    r = region_of(i)
    if not r.bounded or r.diameter <= 0:
        return SKIP
    lo, hi = r.diameter / 2, r.diameter / math.sqrt(3.0)
    tol = 1e-9 * max(1.0, r.diameter)
    if r.cover_r < lo - tol:
        return f"r_MEC={r.cover_r!r} < 𝒟/2={lo!r}（违反命题 A）"
    if r.cover_r > hi + tol:
        return f"r_MEC={r.cover_r!r} > 𝒟/√3={hi!r}（违反 Jung）"
    return None


def _mec_three_way(i: Instance) -> str | None:
    """清单#4 的第三路：covering_radius == min_enclosing_circle == Welzl。"""
    from claim_tester import oracle_mec

    r = region_of(i)
    if not r.bounded or len(r.polygon) < 2:
        return SKIP
    a = geometry.covering_radius(r.polygon)
    b = geometry.min_enclosing_circle(r.polygon)
    if b is None:
        return "min_enclosing_circle 返回 None"
    if a != b.radius:
        return f"孪生体不逐位相同：covering_radius={a!r} vs MEC={b.radius!r}"
    c = oracle_mec(r.polygon)
    if abs(a - c[1]) > 1e-9:
        return f"与 Welzl 不一致：{a!r} vs {c[1]!r}"
    return None


# ── claim #7–#9：单调性与尺度 ────────────────────────────────
def _scale_arena_only(i: Instance) -> str | None:
    """清单#5（一）：只缩放场地。Ω 不活化时场地根本不该被碰到 ⟹ 逐位不变。

    门禁用 `omega_certified_inactive`（独立重算），**不用** `r.omega_active`：
    后者是 geometry 自报的，强制裁剪的实现错能让它谎报 True 从而让本 claim 跳过。
    """
    if not omega_certified_inactive(i):
        return SKIP
    r = region_of(i)
    if not r.bounded:
        return SKIP
    if r.omega_active:
        return ("证书说 Ω 不活化，Region 却报活化——"
                "裁剪被无条件打开了（尺度不变性已被破坏）")
    big = region_of(i, arena_r=ARENA_R * 100.0)
    if big.diameter != r.diameter:
        return f"只把场地 ×100 就改变了 𝒟：{r.diameter!r} → {big.diameter!r}"
    return None


def _scale_both(i: Instance) -> str | None:
    """清单#5（二）：站点、场地、r_eff **同时** ×100 ⟹ 𝒟 恰好 ×100。

    ⚠ **站点必须一起缩放**，否则这不是相似变换：只放大 Ω 而站点不动，
    几何形状整个变了，𝒟 当然不会 ×100（早先版本就这么错了，报出 0.99 的偏差）。
    示向度不变（方向是无量纲量）。
    """
    r = region_of(i)
    if not r.bounded or r.diameter <= 0:
        return SKIP
    k = 100.0
    sites = [(p[0] * k, p[1] * k) for p in i.sites]
    big = geometry.localization_region(
        sites, i.bearings, i.err_deg, arena_r=1800.0 * k, r_eff=i.r_eff * k)
    if not big.bounded or big.diameter <= 0:
        return "同时缩放后区域退化"
    # 阈值取计划 §四 的验收标准 1e-6（相对），不是自设的更严值：
    # ×100 后坐标到 1e5 量级，射线求交的相减抵消会把舍入放大到 ~1e-9，
    # 实测最坏 2.8e-9——那仍是浮点，不是几何。（`scale_arena_only` 之所以
    # 能用严格 `==`，是因为 Ω 不活化时 tier 1 走的是**同一条**代码路径。）
    rel = abs(big.diameter / (r.diameter * k) - 1.0)
    if rel > 1e-6:
        return f"缩放后 𝒟/(100𝒟) − 1 = {rel:.3e} > 1e-6"
    return None


def _epsilon_monotone(i: Instance) -> str | None:
    """清单#6：ε 变大 ⟹ 区域变大 ⟹ 𝒟 不减。"""
    r = region_of(i)
    if not r.bounded or r.diameter <= 0:
        return SKIP
    eps2 = i.err_deg + 0.25
    if eps2 > 15.0:
        return SKIP
    big = geometry.localization_region(i.sites, i.bearings, eps2, r_eff=i.r_eff)
    if not big.bounded:
        return None                       # 变大后反而退化，不算反例
    if big.diameter < r.diameter - 1e-9 * max(1.0, r.diameter):
        return f"ε↑ 后 𝒟 反而变小：{r.diameter:.6f} → {big.diameter:.6f}"
    return None


def _reff_monotone(i: Instance) -> str | None:
    """清单#7：r_eff 变大 ⟹ Ω 变大 ⟹ 𝒟 不减。"""
    r = region_of(i)
    if not r.bounded:
        return SKIP
    up = i.r_eff + 50.0
    if up > 3000.0:
        return SKIP
    big = geometry.localization_region(i.sites, i.bearings, i.err_deg, r_eff=up)
    if not big.bounded:
        return None
    if big.diameter < r.diameter - 1e-9 * max(1.0, r.diameter):
        return f"r_eff↑ 后 𝒟 反而变小：{r.diameter:.6f} → {big.diameter:.6f}"
    return None


def _gamma_monotone(i: Instance) -> str | None:
    """清单#8：γ↓ ⟹ 𝒟↑（限 Ω 不活化且 γ > 2ε 的纯楔形域）。

    做法：站点**不动**，把两条示向度绕其角平分线**对称收拢**，使
    γ 精确取到预设序列。这是纯几何操作（不假装存在同一个源），
    正合本 claim 的对象——楔形之交。
    """
    if i.n != 2 or not geometry.wedge_bounded(i.bearings, i.err_deg):
        return SKIP
    r0 = region_of(i)
    if not r0.bounded or r0.omega_active or r0.diameter <= 0:
        return SKIP

    # 角平分线方向（用单位向量求和，避免 0/360 环绕）
    ux = sum(math.cos(math.radians(b)) for b in i.bearings)
    uy = sum(math.sin(math.radians(b)) for b in i.bearings)
    if math.hypot(ux, uy) < 1e-12:
        return SKIP
    alpha = math.degrees(math.atan2(uy, ux))

    prev_d = None
    for g in (raw_gamma(*i.bearings), 60.0, 45.0, 30.0, 20.0, 12.0, 8.0):
        if g <= 2.0 * i.err_deg + 0.5:
            break                                   # 再小就无界，不是本 claim 的域
        bs = [(alpha + g / 2) % 360.0, (alpha - g / 2) % 360.0]
        rr = geometry.localization_region(i.sites, bs, i.err_deg, r_eff=i.r_eff)
        if not rr.bounded or rr.omega_active or rr.diameter <= 0:
            break                                   # 掉出 Ω 不活化域，停止比较
        if prev_d is not None and rr.diameter < prev_d - 1e-9 * max(1.0, prev_d):
            return (f"γ {prev_g:.3f}°→{g:.3f}° 变小时 𝒟 反而减小："
                    f"{prev_d:.6f} → {rr.diameter:.6f}")
        prev_d, prev_g = rr.diameter, g
    return None


# ── claim #10–#13：oracle 交叉复核 ───────────────────────────
def _oracle_diameter_bracket(i: Instance, m: int = 32) -> str | None:
    """oracle：边内采样的直径 ≤ 真值，且亏空 ≤ 最长边/m；加密必须收敛。

    ⚠ 早先用的是**包围盒网格**，声称亏空 ≤ 2δ。实测最坏到 **9δ**——
    因为落在多边形外的网格点会被丢弃，格点离边界可以很远，那个界根本证不出来。
    换成边内采样后界是**构造性可证**的（见 `_edge_samples`）。
    """
    r = region_of(i)
    if not r.bounded or len(r.polygon) < 3 or r.diameter <= 0:
        return SKIP
    prev = None
    for mm in (m, 2 * m):
        pts, longest = _edge_samples(r.polygon, mm)
        if len(pts) < 2:
            return SKIP
        d_s, _ = geometry.max_pairwise_distance(pts)
        gap = r.diameter - d_s
        if gap < -1e-9 * max(1.0, r.diameter):
            return f"采样直径 {d_s!r} 超过真值 {r.diameter!r}（不可能）"
        bound = longest / mm
        if gap > bound + 1e-9 * max(1.0, r.diameter):
            return (f"真值 − 采样 = {gap:.6f} > 最长边/m = {bound:.6f}"
                    f"（m={mm}，|e|max={longest:.3f}）")
        if prev is not None and gap > prev + 1e-9 * max(1.0, r.diameter):
            return f"m 加倍后亏空反而变大：{prev:.6f} → {gap:.6f}"
        prev = gap
    return None


def _oracle_containment(i: Instance) -> str | None:
    """oracle：**角度式**谓词必须与叉积式一致地接受区域内的点。

    ⚠ 必须同时查 `corners_raw`（谓词自己的输出），不能只查 `r.polygon`。
    `mutant_wedge_cross_reversed` 把叉积次序反过来 = 把每个楔形旋转 180°：
    区域与谓词**一起**转，于是 `region_corners` 筛出的是一批**错的**点，
    最终凸包看起来依然自洽——只查 polygon 抓不到它（实测逃逸过）。
    而 `corners_raw` 正是叉积谓词的原始输出，用完全不用叉积的角度式谓词一查
    就现形。另外还要求它非空：反向谓词会把 `region_corners` 清空、静默退化到
    tier 2b，那种「什么都没筛出来」也必须算失败，不能算通过。
    """
    from claim_tester import oracle_contains

    r = region_of(i)
    if not r.bounded or len(r.polygon) < 3:
        return SKIP

    # (a) 谓词的原始输出必须非空——空集不许静默当通过
    if omega_certified_inactive(i) and len(r.corners_raw) < 3:
        return (f"Ω 不活化且有界，叉积谓词却只筛出 {len(r.corners_raw)} 个候选点"
                f"（角域之交应给出 3 个以上）")

    # (b) polygon 与 corners_raw 的每个点都要被角度式谓词接受
    tested = 0
    for name, pts in (("顶点", r.polygon), ("候选点", r.corners_raw)):
        for q in pts:
            for s, b in zip(i.sites, i.bearings):
                if near_site(q, [s]):
                    continue
                tested += 1
                if not oracle_contains(s, b, i.err_deg + _EPS_TOL, q):
                    return (f"{name} {q} 不被**角度式**谓词接受（站 {s}）——"
                            f"叉积式与角度式不一致（叉积次序反了？）")
    return None if tested else SKIP


def _criterion_by_angles(i: Instance) -> str | None:
    """泰勒斯：点积式覆盖判定 == 顶点张角式覆盖判定。"""
    from claim_tester import oracle_cover_by_angles, oracle_mec

    r = region_of(i)
    if not r.bounded or r.diameter_ends is None or len(r.polygon) < 3:
        return SKIP
    a, b = r.diameter_ends
    if geometry.diameter_circle_covers(a, b, r.polygon) != \
            oracle_cover_by_angles(a, b, r.polygon):
        return "点积式与角度式覆盖判定不一致"
    _ = oracle_mec
    return None


# ── claim #14–#16：窗口定理及其适用域 ────────────────────────
def _cover_band(i: Instance) -> str | None:
    """★ 窗口定理：**限 Ω 不活化的纯楔形域**。

    不覆盖 ⟺ 90° < γ < 90° + 2ε（γ 为**未折叠**交会角）。
    """
    if i.n != 2 or not geometry.wedge_bounded(i.bearings, i.err_deg):
        return SKIP
    r = region_of(i)
    if not r.bounded or r.omega_active:
        return SKIP
    g = raw_gamma(i.bearings[0], i.bearings[1])
    lo, hi = 90.0, 90.0 + 2.0 * i.err_deg
    if abs(g - lo) < 1e-6 or abs(g - hi) < 1e-6:
        return SKIP                                   # 窗口边界是刀刃，不算
    pred_not_cover = lo < g < hi
    if r.diameter_circle_ok == pred_not_cover:
        return (f"γ={g:.9f}° 预测{'不覆盖' if pred_not_cover else '覆盖'}，"
                f"实测相反（𝒟={r.diameter:.6f}，余量={r.cover_margin:.3e}）")
    return None


def _cover_band_omega_active(i: Instance) -> str | None:
    """⚠ **已被证伪的猜想**：窗口定理在 Ω 活化时也成立。

    必须报出反例。见 `问题1-覆盖判据的适用域（Ω 修正）.md`：
    20000 个实例里 Ω 活化 4489 个，其中 766 例失配（17.1%），
    方向几乎单向（Ω 制造不覆盖 759 例 vs 修复 7 例）。
    """
    if i.n != 2 or not geometry.wedge_bounded(i.bearings, i.err_deg):
        return SKIP
    r = region_of(i)
    if not r.bounded or not r.omega_active:
        return SKIP
    g = raw_gamma(i.bearings[0], i.bearings[1])
    pred_not_cover = 90.0 < g < 90.0 + 2.0 * i.err_deg
    if r.diameter_circle_ok == pred_not_cover:
        return (f"γ={g:.6f}° 预测{'不覆盖' if pred_not_cover else '覆盖'}，"
                f"Ω 却给出相反结果（亏空/𝒟={-r.cover_margin / r.diameter:.3e}）")
    return None


def _cover_site_independent(i: Instance) -> str | None:
    """窗口定理的判别只依赖 γ 与 ε，与站点位置、基线长度无关。

    ⚠ 前提是**两次都得 Ω 不活化**。旋转会挪动区域相对场地圆心的位置，
    于是完全可能「原构形 Ω 不活化、旋转后 Ω 活化」——那判定翻转是**应该的**，
    不是反例。早先只查了原构形的 `omega_active`，把这种情况误报成反例。
    """
    if i.n != 2 or not omega_certified_inactive(i):
        return SKIP
    r = region_of(i)
    if not r.bounded or r.omega_active:
        return SKIP
    g = raw_gamma(i.bearings[0], i.bearings[1])
    if 90.0 < g < 90.0 + 2.0 * i.err_deg:
        return SKIP
    # 围绕两站连线的中点整体旋转，γ 不变 ⟹ 覆盖判定必须不变
    mid = ((i.sites[0][0] + i.sites[1][0]) / 2,
           (i.sites[0][1] + i.sites[1][1]) / 2)
    ang = 0.7
    ca, sa = math.cos(ang), math.sin(ang)

    def rot(p):
        dx, dy = p[0] - mid[0], p[1] - mid[1]
        return (mid[0] + ca * dx - sa * dy, mid[1] + sa * dx + ca * dy)

    sites2 = [rot(p) for p in i.sites]
    bs2 = [(b + math.degrees(ang)) % 360.0 for b in i.bearings]
    j = Instance(sites=sites2, bearings=bs2, err_deg=i.err_deg, r_eff=i.r_eff)
    if not omega_certified_inactive(j):       # 旋转后 Ω 活化了 ⟹ 不在定理域内
        return SKIP
    r2 = geometry.localization_region(sites2, bs2, i.err_deg, r_eff=i.r_eff)
    if not r2.bounded or r2.omega_active:
        return SKIP
    if r2.diameter_circle_ok != r.diameter_circle_ok:
        return "整体旋转后覆盖判定翻转（γ、ε 都没变，两次 Ω 都不活化）"
    return None


def _n_generality(i: Instance) -> str | None:
    """命题 B：顶点数 m ≤ 2n（Ω 不活化）／m ≤ 2n + K（活化）。"""
    if i.n < 2:
        return SKIP
    r = region_of(i)
    if not r.bounded:
        return SKIP
    tested = 0
    for q in r.polygon:
        if near_site(q, i.sites):
            continue
        for s, b in zip(i.sites, i.bearings):
            tested += 1
            off = abs(geometry.norm180(
                math.degrees(math.atan2(q[1] - s[1], q[0] - s[0])) - b))
            if off > i.err_deg + _EPS_TOL:
                return f"n={i.n}：顶点 {q} 超出站 {s} 的楔形（偏差 {off:.6f}°）"
    if not tested:
        return SKIP
    bound = 2 * i.n if not r.omega_active else 2 * i.n + 360
    if r.m > bound:
        return f"n={i.n}：m={r.m} > {bound}（Ω {'活化' if r.omega_active else '不活化'}）"
    return None


def _no_silent_degenerate(i: Instance) -> str | None:
    """退化必须显式报告：bounded=False ⟹ degenerate 非空；𝒟=0 ⟹ 未声明有界。"""
    r = region_of(i)
    if not r.bounded and not r.degenerate:
        return "bounded=False 却没有退化说明（静默返回）"
    if r.diameter == 0.0 and r.bounded and r.m >= 2:
        return "𝒟=0 却声明有界"
    return None


def _interior_candidates(i: Instance) -> str | None:
    """筛出的候选点必须都是凸包顶点（无严格内点）。

    Γ 一般位置下 n=2 恰有 4 个候选点。不足 4 个的情形是**半直线**截断：
    `ray_intersection` 要求两条 t 都 > 0，当某个角落在某站**背后**时该
    交点不存在——这是契约行为，不是缺陷（实测全部出现在 γ≳177° 或 Ω 活化处）。
    故只在这两个条件都排除时要求 4 个。
    """
    if i.n != 2 or not geometry.wedge_bounded(i.bearings, i.err_deg):
        return SKIP
    r = region_of(i)
    if not r.bounded or len(r.corners_raw) < 2:
        return SKIP
    hull = geometry.convex_hull(r.corners_raw)
    for p in r.corners_raw:
        if p not in hull:
            return f"候选点 {p} 不是凸包顶点（严格内点）"
    g = raw_gamma(i.bearings[0], i.bearings[1])
    if not r.omega_active and 15.0 < g < 165.0 and len(r.corners_raw) != 4:
        return (f"γ={g:.3f}° 且 Ω 不活化，候选点却只有 "
                f"{len(r.corners_raw)} 个（应为 4）")
    return None


def _diameter_is_diagonal(i: Instance) -> str | None:
    """⚠ **已被证伪的猜想**：凸四边形的直径必落在一条对角线上。

    在**本题域**内成立（ε ≤ 2°、站源距比 ≤ 1.5 时 19687/19687），
    在**一般域**内不成立（全域 36330 中有 29 例落在边上）。故它是
    **题域限定命题**，不是一般定理——不得作为定理写进论文。
    """
    if i.n != 2 or not geometry.wedge_bounded(i.bearings, i.err_deg):
        return SKIP
    corners, why = geometry.uncertainty_corners(
        i.sites[0], i.bearings[0], i.sites[1], i.bearings[1], i.err_deg)
    if why:
        return SKIP
    hull = geometry.convex_hull(corners)
    if len(hull) != 4:
        return SKIP
    d, ends = geometry.max_pairwise_distance(hull)
    if ends is None:
        return SKIP
    ia, ib = _idx(hull, ends[0]), _idx(hull, ends[1])
    if ia is None or ib is None:
        return SKIP
    if {ia, ib} in ({0, 2}, {1, 3}):
        return None                                    # 对角线，命题成立
    # 排除并列（对角线同样长）的歧义情形
    second = max(geometry.dist(hull[p], hull[q])
                 for p in range(4) for q in range(p + 1, 4)
                 if {p, q} not in ({ia, ib},))
    if d - second <= 1e-9 * d:
        return SKIP
    return (f"直径落在**边** {ia}–{ib} 上：{d:.6f} vs 次长 {second:.6f}"
            f"（ε={i.err_deg}°，基线比 {max(math.dist(s, (0, 0)) for s in i.sites) / min(math.dist(s, (0, 0)) for s in i.sites):.2f}）")


def _clear_radius_consistency(i: Instance) -> str | None:
    """显式半径与 r_MEC 一致（**绝不** monkeypatch CLEAR_RADIUS）。"""
    r = region_of(i)
    if not r.bounded or len(r.polygon) < 2:
        return SKIP
    for R in (20.0, 50.0, 200.0):
        got = geometry.clearing_guaranteed(r.polygon, R)
        want = geometry.min_enclosing_circle(r.polygon).radius <= R
        if got != want:
            return f"R={R}: clearing_guaranteed={got} 但 r_MEC≤R 为 {want}"
    return None


# ── claim：冻结回归 ─────────────────────────────────────────
_RAY_CASES = [
    # (s1, θ1, s2, θ2, 期望有交点, 说明)      —— 全部可手算
    ((0.0, 0.0), 0.0, (5.0, -3.0), 90.0, True,
     "正向相交：(0,0)+t(1,0) 与 (5,−3)+s(0,1) 交于 (5,0)，t=5>0, s=3>0"),
    ((0.0, 0.0), 0.0, (5.0, -3.0), 270.0, False,
     "s<0：直线仍交于 (5,0)，但那在 S₂ 的**背后**"),
    ((0.0, 0.0), 180.0, (5.0, -3.0), 90.0, False,
     "t<0：交点在 S₁ 背后"),
    ((0.0, 0.0), 0.0, (0.0, 5.0), 0.0, False,
     "平行"),
]


def _ray_contract(i: Instance) -> str | None:
    """契约：`ray_intersection` 只接受**半直线**（两条 t 都 > 0），不是直线。

    ⚠ 为什么必须有这条 claim：`no_t_positive_filter` 这个 mutant（放开 t ≤ 0
    的守卫）在**区域层面是不可观测的**——实测 80000 个「反向交点」全部被楔形
    谓词筛掉（反向点相对该站的视角偏离 θ 达 180°−ε ≫ ε），所以任何看区域输出
    的 claim 都抓不到它，mutant 会**逃逸**（实测确实逃逸过）。
    唯一能抓住它的地方就是**函数契约本身**，故此处直接查契约。

    代价：本 claim 与实例无关，跑 267 遍是浪费。用 `key` 去重不管用（每个实例
    都不同），故在第一遍后自行短路——见 `_ray_contract_done`。
    """
    if _ray_contract_done[0]:
        return SKIP
    _ray_contract_done[0] = True
    for s1, t1, s2, t2, want, why in _RAY_CASES:
        got = geometry.ray_intersection(s1, t1, s2, t2)
        if (got is not None) != want:
            return (f"契约违反——期望{'有' if want else '无'}交点，"
                    f"实得 {got!r}（{why}）")
        if got is not None and geometry.dist(got[0], (5.0, 0.0)) > 1e-9:
            return f"交点位置错：{got[0]!r} 应为 (5.0, 0.0)"
    return None


_ray_contract_done = [False]


def _n2_regression_frozen(i: Instance) -> str | None:
    """清单#21：数据集 A/B 的 𝒟、r_MEC、覆盖必须与冻结常量一致。

    ⚠ 数据集 B 与 Ω 不相容（§7.1），故这里走 **Ω-free** 的 `localize()`，
    锚定的是与 GeoGebra 对照过的纯交会几何。
    """
    got = None
    for ds in (DS_A, DS_B):
        if i.sites[0] != ds["s1"] or i.sites[1] != ds["s2"]:
            continue
        got = ds
    if got is None:
        return SKIP
    loc = geometry.localize(got["s1"], got["th1"], got["s2"], got["th2"], 1.0)
    if not loc.bounded:
        return f"冻结数据集退化：{loc.degenerate}"
    want_d, want_r = ((D_A, R_A) if got is DS_A else (D_B, None))
    if abs(loc.diameter - want_d) > 1e-6:
        return f"𝒟 偏离冻结值：{loc.diameter!r} vs {want_d!r}"
    if want_r is not None and abs(loc.cover_r - want_r) > 1e-6:
        return f"r_MEC 偏离冻结值：{loc.cover_r!r} vs {want_r!r}"
    return None


# ── 注册表 ──────────────────────────────────────────────────
def q1_claims() -> list[Claim]:
    return [
        Claim("n2_bit_identical", "n=2 且 Ω 不活化时新实现与 localize() 逐位一致",
              _n2_bit_identical, kind="regression", source="清单#2"),
        Claim("corner_back_substitution", "每个顶点回代各站方位角 ≤ ε",
              _corner_back_substitution, source="清单#1"),
        Claim("convexity", "区域 CCW 凸且面积为正",
              _convexity, source="清单#3"),
        Claim("cover_iff_mec_eq_half_diameter", "覆盖 ⟺ r_MEC = 𝒟/2",
              _cover_iff_mec_eq_half_diameter, source="清单#4"),
        Claim("jung_bounds", "𝒟/2 ≤ r_MEC ≤ 𝒟/√3",
              _jung_bounds, source="命题 A + Jung"),
        Claim("mec_three_way", "covering_radius == MEC 孪生体 == Welzl",
              _mec_three_way, source="清单#4"),
        Claim("scale_arena_only", "只缩放场地：Ω 不活化时 𝒟 逐位不变",
              _scale_arena_only, source="清单#5"),
        Claim("scale_both", "场地与 r_eff 同时 ×100 ⟹ 𝒟 恰好 ×100",
              _scale_both, source="清单#5"),
        Claim("epsilon_monotone", "ε↑ ⟹ 𝒟 不减",
              _epsilon_monotone, source="清单#6"),
        Claim("reff_monotone", "r_eff↑ ⟹ 𝒟 不减",
              _reff_monotone, source="清单#7"),
        Claim("oracle_diameter_bracket", "0 ≤ 𝒟_真 − 𝒟_采样 ≤ 2δ",
              _oracle_diameter_bracket, source="oracle"),
        Claim("oracle_containment", "顶点被角度式谓词接受",
              _oracle_containment, source="oracle"),
        Claim("criterion_by_angles", "点积式 == 角度式覆盖判定",
              _criterion_by_angles, source="泰勒斯"),
        Claim("cover_band", "★ 窗口定理：Ω 不活化时不覆盖 ⟺ 90°<γ<90°+2ε",
              _cover_band, kind="theorem", source="本次新定理"),
        Claim("cover_band_omega_active",
              "窗口定理在 Ω 活化时也成立（**已被证伪**）",
              _cover_band_omega_active, kind="refuted", expect="refuted",
              source="Ω 修正"),
        Claim("cover_site_independent", "覆盖判定与站点位置/基线无关",
              _cover_site_independent, source="本次新定理"),
        Claim("n_generality", "任意 n：顶点满足全部楔形，且 m 符合层级界",
              _n_generality, source="命题 B"),
        Claim("no_silent_degenerate", "退化必须显式报告，不得静默给 𝒟=0",
              _no_silent_degenerate, source="踩坑"),
        Claim("interior_candidates", "候选点均为凸包顶点（无严格内点）",
              _interior_candidates, source="实测"),
        Claim("diameter_is_diagonal",
              "凸四边形直径必在对角线上（**已被证伪**）",
              _diameter_is_diagonal, kind="refuted", expect="refuted",
              source="引理 M2"),
        Claim("ray_contract", "ray_intersection 只接受半直线（两 t 均 > 0）",
              _ray_contract, kind="contract", source="契约"),
        Claim("clear_radius_consistency", "clearing_guaranteed 与 r_MEC 一致",
              _clear_radius_consistency, source="见风险"),
        Claim("n2_regression_frozen", "数据集 A/B 与冻结常量一致",
              _n2_regression_frozen, kind="regression", source="repro_cover_fail"),
    ]


def reset_run_state() -> None:
    """把与实例无关的一次性 claim 复位（同进程内重复跑 suite 时必须调）。"""
    _ray_contract_done[0] = False


LANES = {
    "physical": q1_instances,
    "refutation": q1_refutation_instances,
    "frozen": q1_frozen_instances,
}

_FROZEN_CLAIMS = {"n2_regression_frozen"}


def lane_of(claim: Claim) -> str:
    """泳道分配：冻结回归 → `frozen`；`expect="refuted"` → 一般域；其余 → 物理自洽域。"""
    if claim.name in _FROZEN_CLAIMS:
        return "frozen"
    return "refutation" if claim.expect == "refuted" else "physical"
