"""
B题交会定位几何内核（问题 1）。

给定两个检测点与两个示向度（各带 ±1° 误差），构造「定位不确定区域」，
并给出覆盖判定与清除可行性判定。**问题 2/3/4 的共同地基。**

坐标系（已用官方模拟器实测锁定，见 strategy_notes.md）：
    0° = +x 轴，逆时针为正
    θ = atan2(y_src - y_meas, x_src - x_meas) mod 360
**不是**罗盘方位角。搞反会让整个定位模块「静默失效」——能算出点，但在错误位置。

核心机制：每频道至多 1 个干扰源，故同频道在两个不同位置的示向度，
两条射线交会即该源位置。

参考 skill：.claude/skills/b-simulator/references/strategy_notes.md
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache

from config import CLEAR_RADIUS

# Windows 中文控制台默认 GBK，打印 ✓/✗ 会抛 UnicodeEncodeError 而**中断自检**。
# 必须放在任何 print 之前——否则检查结果根本来不及输出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

Point = tuple[float, float]

# 判定两条射线平行的叉积阈值。方向向量均为单位向量，故 |d1 × d2| = |sin γ|，
# 取 1e-9 对应交会角 γ ≈ 6e-8 度——远小于任何有意义的几何退化。
PARALLEL_EPS = 1e-9

# 浮点比较容差（坐标量级为 1e3 m，1e-9 相对误差足够）
_EPS = 1e-9


# ── 基础向量运算 ────────────────────────────────────────────
def unit_vector(theta_deg: float) -> Point:
    """示向度（度，数学坐标系）→ 单位方向向量 (cosθ, sinθ)。"""
    a = math.radians(theta_deg)
    return (math.cos(a), math.sin(a))


def cross(a: Point, b: Point) -> float:
    """二维叉积 a × b = a.x·b.y − a.y·b.x。"""
    return a[0] * b[1] - a[1] * b[0]


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm180(deg: float) -> float:
    """把角度差归一到 [0, 180]。"""
    d = abs(deg) % 360.0
    return 360.0 - d if d > 180.0 else d


# ── 射线交会 ────────────────────────────────────────────────
def ray_intersection(
    s1: Point, theta1: float, s2: Point, theta2: float
) -> tuple[Point, float, float] | None:
    """两条**射线**求交。

    射线参数式 P = S + t·(cosθ, sinθ)，要求 t > 0。
    返回 (交点, t1, t2)；平行、或交点在任一射线「身后」时返回 None。

    ⚠ 必须只取 t > 0 的解。若坐标系搞反（用罗盘方位角），
    两个 t 会同时变负——这正是实测中用来反证坐标系约定的判据。
    """
    d1 = unit_vector(theta1)
    d2 = unit_vector(theta2)
    den = cross(d1, d2)
    if abs(den) < PARALLEL_EPS:
        return None                                  # 平行（同向或反向）

    v = sub(s2, s1)
    t1 = cross(v, d2) / den
    t2 = cross(v, d1) / den
    if t1 <= 0.0 or t2 <= 0.0:
        return None                                  # 交点在某条射线身后

    return (s1[0] + t1 * d1[0], s1[1] + t1 * d1[1]), t1, t2


def crossing_angle(theta1: float, theta2: float) -> float:
    """交会角 γ：两条射线的夹角，规约到锐角侧 [0, 90]。

    ⚠ 定位误差 ∝ 1/sin γ。γ → 0 时不确定区域爆炸——这正是问题 2 的动机。
    """
    d = norm180(theta1 - theta2)
    return 180.0 - d if d > 90.0 else d


# ── 任意 n 的通用几何（2026-09-11 新增）─────────────────────
# 问题 1 的判据本身对任意检测点数成立；下面是把它从 n=2 推广到任意 n 所需的
# 基础设施。既有的 uncertainty_corners / localize / covering_radius 一字未改。
#
# ⚠ 角的用法：覆盖窗口定理用的是 norm180(θ1−θ2)（未折叠，值域 [0°,180°]），
#    **不是** crossing_angle（它折叠到 [0°,90°]，对 >90° 的角返回其补角）。
#    把 crossing_angle 的输出喂给覆盖判据是本项目最容易犯的错。


def wedge_ok(site: Point, theta: float, eps: float, p: Point) -> bool:
    """p 是否落在站点 site 的角域 [θ−ε, θ+ε] 内。

    楔形 = 两个半平面之交：
        cross(u, p−S) ≥ 0      （p 在 θ−ε 方向的逆时针侧）
        cross(v, p−S) ≤ 0      （p 在 θ+ε 方向的顺时针侧）
    其中 u = 单位向量(θ−ε)，v = 单位向量(θ+ε)。

    ⚠ 叉积的**次序**不能反。写成 cross(p−S, u) 会把楔形静默旋转 180°，
    变成一个「能算出点、但位置全错」的实现——本项目已踩过一次这个坑。
    """
    d = sub(p, site)
    return (
        cross(unit_vector(theta - eps), d) >= -_EPS
        and cross(unit_vector(theta + eps), d) <= _EPS
    )


@lru_cache(maxsize=512)
def _circle_polygon_cached(
    cx: float, cy: float, radius: float, segments: int
) -> tuple[Point, ...]:
    return tuple(
        (
            cx + radius * math.cos(2.0 * math.pi * k / segments),
            cy + radius * math.sin(2.0 * math.pi * k / segments),
        )
        for k in range(segments)
    )


def circle_polygon(center: Point, radius: float, segments: int = 360) -> list[Point]:
    """圆盘的**内接**正 segments 边形，逆时针，顶点严格落在圆上。

    内接（而非外切）保证多边形 ⊆ 圆盘，即用多边形近似 Ω 得到的是**保守**的
    子集——直径只会被低估而不会被高估。弓高 = R(1−cos(π/K))，
    K=360、R=1800 时约 0.0685 m；K=120 时 0.617 m，K=72 时 1.71 m。

    ⚠ 返回的是**副本**，调用方可随意改动；内部结果带 lru_cache（纯函数，
    同一 K 边形在一条路径上会被建三次）。这也是热路径上最贵的一步，故必须缓存。
    """
    if segments < 3:
        raise ValueError(f"segments 至少为 3，收到 {segments}")
    return list(_circle_polygon_cached(center[0], center[1], radius, segments))


def candidate_vertices(rays: list[tuple[Point, float]]) -> list[Point]:
    """把 2n 条边界半直线两两求交，返回候选顶点（枚举序）。

    rays 的顺序约定：逐站点成对追加 (S_i, θ_i−ε), (S_i, θ_i+ε)。

    ⚠ 枚举顺序**必须**保持「i<j 且逐站点成对」。n=2 时这让 ray_intersection
    的实参顺序与 uncertainty_corners 的嵌套循环逐位相同，从而保证 n=2 的
    浮点结果与既有实现一致（文档自洽性检查第 2 条）。

    同站点的两条射线不参与求交：它们的交点是站点自身（t=0），
    而 ray_intersection 的 t > 0 守卫本就会返回 None，跳过不改变结果。
    """
    out: list[Point] = []
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            if rays[i][0] == rays[j][0]:
                continue
            hit = ray_intersection(rays[i][0], rays[i][1], rays[j][0], rays[j][1])
            if hit is not None:
                out.append(hit[0])
    return out


def convex_hull(pts: list[Point]) -> list[Point]:
    """Andrew 单调链求凸包，逆时针，**去掉共线点**。少于 3 点时原样返回。"""
    p = sorted(set(pts))
    if len(p) <= 2:
        return p

    def half(seq: list[Point]) -> list[Point]:
        h: list[Point] = []
        for q in seq:
            while len(h) >= 2 and cross(sub(h[-1], h[-2]), sub(q, h[-2])) <= _EPS:
                h.pop()
            h.append(q)
        return h

    return half(p)[:-1] + half(p[::-1])[:-1]


def polar_order(pts: list[Point]) -> list[Point]:
    """绕质心按极角逆时针排序（只改顺序，不动坐标——位相同安全）。"""
    if len(pts) <= 2:
        return list(pts)
    cx = sum(q[0] for q in pts) / len(pts)
    cy = sum(q[1] for q in pts) / len(pts)
    return sorted(pts, key=lambda q: math.atan2(q[1] - cy, q[0] - cx))


def polygon_area(poly: list[Point]) -> float:
    """鞋带公式的有符号面积，逆时针为正。"""
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        s += a[0] * b[1] - b[0] * a[1]
    return s / 2.0


def is_convex_ccw(poly: list[Point]) -> bool:
    """多边形是否严格逆时针凸（相邻边叉积全 > 0）。"""
    n = len(poly)
    if n < 3:
        return False
    return all(
        cross(sub(poly[(i + 1) % n], poly[i]), sub(poly[(i + 2) % n], poly[(i + 1) % n]))
        > 0
        for i in range(n)
    )


def wedge_bounded(bearings: list[float], err_deg: float = 1.0) -> bool:
    """角域之交是否有界——**只依赖示向度**，与站点位置无关。

    把每个角域写成半平面：
        cross(u,p) ≥ c  ⟹ 外法向方向 θ−ε+90°
        cross(v,p) ≤ c  ⟹ 外法向方向 θ+ε−90°
    解集有界 ⟺ 这些外法向正张成 R² ⟺ 方向角排序后的**最大间隙 < 180°**。

    n=1（单个楔形）恒为 False。**n=2 时有精确的充要条件**：

        bounded ⟺ γ > 2ε

    理由：两楔形的方向范围分别是 [θ₁−ε, θ₁+ε] 与 [θ₂−ε, θ₂+ε]，二者有交
    ⟺ γ < 2ε ⟺ 存在一条射线同时落在两个楔形内 ⟹ 区域沿该方向无限延伸。
    γ = 2ε 时两范围**恰好相切**（仍共享一条边界射线），故依然无界——
    这也是判据用 `gap < 180 − 1e-9` 而非 `<= 180` 的原因。
    数值已验证 9/9（含 γ = 2ε 的临界点）。

    无界时必须先与 Ω 求交，否则直径无意义（tier 2b）。
    """
    angs: list[float] = []
    for th in bearings:
        angs.append((th - err_deg + 90.0) % 360.0)
        angs.append((th + err_deg - 90.0) % 360.0)
    if len(angs) < 2:
        return False
    angs.sort()
    gap = angs[0] + 360.0 - angs[-1]           # 环绕间隙
    for i in range(1, len(angs)):
        gap = max(gap, angs[i] - angs[i - 1])
    return gap < 180.0 - 1e-9


# ── 不确定区域 ──────────────────────────────────────────────
def uncertainty_corners(
    s1: Point, theta1: float, s2: Point, theta2: float, err_deg: float = 1.0
) -> tuple[list[Point], str]:
    """构造 ±err 不确定区域的顶点。

    从 S1 引 θ1±err 两条射线，从 S2 引 θ2±err 两条射线；
    四条射线两两相交（2×2）得到至多 4 个顶点，即区域四角。

    返回 (顶点列表, 退化原因)。退化原因非空表示区域**无界或退化**，
    此时直径无意义，必须在论文中单独讨论。
    """
    corners: list[Point] = []
    for a1 in (theta1 - err_deg, theta1 + err_deg):
        for a2 in (theta2 - err_deg, theta2 + err_deg):
            r = ray_intersection(s1, a1, s2, a2)
            if r is not None:
                corners.append(r[0])

    if len(corners) < 4:
        return corners, (
            f"仅 {len(corners)}/4 个顶点有效——两射线近似平行或交点在反方向，"
            f"不确定区域无界（交会角 γ={crossing_angle(theta1, theta2):.2f}°）"
        )
    return corners, ""


def max_pairwise_distance(pts: list[Point]) -> tuple[float, tuple[Point, Point] | None]:
    """点集的直径：最大欧氏距离及其两端点。O(n²)，n ≤ 4。"""
    best = 0.0
    ends: tuple[Point, Point] | None = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = dist(pts[i], pts[j])
            if d > best:
                best, ends = d, (pts[i], pts[j])
    return best, ends


# ── 泰勒斯定理与圆的覆盖 ────────────────────────────────────
def in_diameter_circle(a: Point, b: Point, p: Point) -> bool:
    """泰勒斯定理：p 是否在以 ab 为**直径**的圆内（含圆周）。

        ∠apb > 90° ⟺ 圆内
        ∠apb = 90° ⟺ 圆周上
        ∠apb < 90° ⟺ 圆外

    转化为点积判定：cos∠apb 的符号 = (a−p)·(b−p) 的符号。
    """
    return ((a[0] - p[0]) * (b[0] - p[0]) + (a[1] - p[1]) * (b[1] - p[1])) <= _EPS


def diameter_circle_covers(a: Point, b: Point, pts: list[Point]) -> bool:
    """以 ab 为直径的圆是否覆盖全部 pts。对应问题 1 的判定要求。"""
    return all(in_diameter_circle(a, b, p) for p in pts)


def circumcircle(a: Point, b: Point, c: Point) -> tuple[Point, float] | None:
    """三角形外接圆 (圆心, 半径)；三点共线时返回 None。"""
    d = 2.0 * (
        a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
    )
    if abs(d) < 1e-12:
        return None
    sa = a[0] ** 2 + a[1] ** 2
    sb = b[0] ** 2 + b[1] ** 2
    sc = c[0] ** 2 + c[1] ** 2
    cx = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
    cy = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
    ctr = (cx, cy)
    return ctr, dist(ctr, a)


def _covers_all(center: Point, r: float, pts: list[Point]) -> bool:
    return all(dist(center, p) <= r + 1e-9 for p in pts)


def covering_radius(pts: list[Point]) -> float:
    """点集的**最小包围圆**半径。

    最小包围圆必由 1/2/3 个点确定，故暴力枚举所有点对（直径圆）与
    三点组（外接圆），取能覆盖全集的最小半径。点数 ≤ 4，代价可忽略。

    ⚠ 不能直接用「直径/2」代替：直径圆覆盖只在四边形满足泰勒斯条件时才成立，
    否则最小包围圆比直径圆更小（Jung 定理给出 D/2 ≤ r ≤ D/√3）。
    """
    n = len(pts)
    if n <= 1:
        return 0.0

    best = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            ctr = ((pts[i][0] + pts[j][0]) / 2, (pts[i][1] + pts[j][1]) / 2)
            if _covers_all(ctr, dist(pts[i], pts[j]) / 2, pts):
                best = min(best, dist(pts[i], pts[j]) / 2)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cc = circumcircle(pts[i], pts[j], pts[k])
                if cc is not None and _covers_all(cc[0], cc[1], pts):
                    best = min(best, cc[1])

    if best == float("inf"):
        # 理论上不可达（点对枚举必有一个可行解），兜底保证不返回 inf
        best = max(dist(pts[0], p) for p in pts)
    return best


def clearing_guaranteed(pts: list[Point], radius: float | None = None) -> bool:
    """能否**保证**清除：存在一点，到区域任意点的距离都不超过 radius。

    等价于最小包围圆半径 ≤ radius。策略含义：定位不确定区域越大越难保证清除，
    但 `/clear` 只要摸到 20 m 内即可（且不受定向盲区限制）——
    所以**降低对定位精度的要求、提高对物理靠近的要求**是可行方向。

    ⚠ `radius` 默认从**调用时**的 `CLEAR_RADIUS` 解析（原为 `radius=CLEAR_RADIUS`，
    在 def 时求值一次）。那个写法会让重绑 `geometry.CLEAR_RADIUS` **改变**
    `Localization.clear_ok`（属性里是调用时全局查找）却**不改变**本函数——
    两个 API 静默不一致。测试台请**永远显式传半径**，不要靠重绑全局来注入。
    """
    if not pts:
        return False
    return covering_radius(pts) <= (CLEAR_RADIUS if radius is None else radius)


# ── 顶层入口 ────────────────────────────────────────────────
@dataclass
class Localization:
    """一次双点交会定位的完整结果。"""

    s1: Point
    theta1: float
    s2: Point
    theta2: float
    err_deg: float = 1.0

    center: Point | None = None                 # 无误差时的标称交点
    r1: float = 0.0                             # 标称交点到 S1 的距离
    r2: float = 0.0                             # 标称交点到 S2 的距离
    corners: list[Point] = field(default_factory=list)
    diameter: float = 0.0
    diameter_ends: tuple[Point, Point] | None = None
    diameter_circle_ok: bool = False            # 直径圆是否覆盖整个区域
    cover_r: float = 0.0                        # 最小包围圆半径
    cross_deg: float = 0.0
    bounded: bool = False
    degenerate: str = ""

    @property
    def clear_ok(self) -> bool:
        """能否保证进入 20 m 清除半径。"""
        return self.bounded and self.cover_r <= CLEAR_RADIUS

    def summary(self) -> str:
        lines = [
            f"S1={_fmt(self.s1)}  θ1={self.theta1:.2f}°",
            f"S2={_fmt(self.s2)}  θ2={self.theta2:.2f}°",
            f"标称交点 {_fmt(self.center)}   "
            f"r1={self.r1:.1f} m  r2={self.r2:.1f} m",
            f"交会角 γ = {self.cross_deg:.2f}°   "
            f"（误差放大 ~1/sinγ = "
            f"{1.0 / math.sin(math.radians(self.cross_deg)):.2f}×）"
            if 0 < self.cross_deg < 90 else f"交会角 γ = {self.cross_deg:.2f}°",
        ]
        if self.degenerate:
            lines.append(f"⚠ 区域退化：{self.degenerate}")
            return "\n".join(lines)
        lines += [
            f"不确定区域 直径 = {self.diameter:.1f} m   "
            f"最小包围圆半径 = {self.cover_r:.1f} m",
            f"直径圆覆盖整个区域？ {'是' if self.diameter_circle_ok else '否'}",
            f"可保证进入 {CLEAR_RADIUS:.0f} m 清除半径？ "
            f"{'是' if self.clear_ok else '否 —— 必须继续逼近'}",
        ]
        return "\n".join(lines)


def _fmt(p: Point | None) -> str:
    return "None" if p is None else f"({p[0]:.1f}, {p[1]:.1f})"


def localize(
    s1: Point,
    theta1: float,
    s2: Point,
    theta2: float,
    err_deg: float = 1.0,
) -> Localization:
    """双点交会定位：算出标称位置、不确定区域与全部派生量。

    这是问题 1 的核心；问题 2 用它评估候选 S2 的优劣，
    问题 3/4 用它把多组示向度变成源位置估计。
    """
    res = Localization(s1=s1, theta1=theta1, s2=s2, theta2=theta2, err_deg=err_deg)
    res.cross_deg = crossing_angle(theta1, theta2)

    hit = ray_intersection(s1, theta1, s2, theta2)
    if hit is None:
        res.degenerate = (
            f"标称两条射线不相交（交会角 γ={res.cross_deg:.2f}°，"
            f"平行或交点在反方向）"
        )
        return res
    res.center, res.r1, res.r2 = hit

    corners, why = uncertainty_corners(s1, theta1, s2, theta2, err_deg)
    res.corners = corners
    if why:
        res.degenerate = why
        return res

    res.bounded = True
    res.diameter, res.diameter_ends = max_pairwise_distance(corners)
    if res.diameter_ends is not None:
        a, b = res.diameter_ends
        res.diameter_circle_ok = diameter_circle_covers(a, b, corners)
    res.cover_r = covering_radius(corners)
    return res


# ── Ω：场地与有效半径约束（2026-09-11 新增）──────────────────
# Ω = D(O, arena_r) ∩ ⋂_i D(S_i, r_eff)
# 它把「两楔形近于平行时限位区域沿长轴逸出」截断。**远不是普遍不活化**：
# γ 小时的活化率高达 95%。下面把它做成可计算的证书，而不是一次性断言。

# 题设：$r_eff$ 是**干扰源的有效接收半径**（各源不同，接口不返回），∈[1000,1500] m。
# 因此对站点的约束是**单边**的 |S_i − G| ≤ r_eff——站点离源多近都测得到，
# 「需走出较远距离才能找到边界」说的是上界。**没有下界 1000**。
R_EFF_MIN = 1000.0
R_EFF_MAX = 1500.0
DEFAULT_R_EFF = R_EFF_MAX          # 取最宽，得到最大的 Ω（对区域范围保守）


def omega_inactive_margin(
    region_pts: list[Point],
    sites: list[Point],
    arena_r: float = 1800.0,
    r_eff: float = DEFAULT_R_EFF,
) -> float:
    """Ω 的**严格**不活化证书：>0 ⟹ Ω 的边界不与区域相交，可跳过裁剪。

    对每个约束圆盘 D(c,R) 算 `R − max_{p∈region} |p−c|`，取最小值。
    因为 region 是凸多边形、|p−c| 是凸函数，**最大值必在顶点取得**
    （极点原理），所以只查 region_pts 就够了——前提是它确实是顶点集。

    ⚠ 与 §0.5 文档里「减去 𝒟」的启发式余量不同：那是把 P₀ 当代表点再加
    一个上界估计；这里是**精确**的逐顶点最大值，>0 就是严格不活化，
    不存在「估计偏乐观」的空间。
    """
    if not region_pts:
        return float("-inf")
    # 场地约束：圆心为原点
    margins = [arena_r - max(math.hypot(p[0], p[1]) for p in region_pts)]
    # 站点约束
    for s in sites:
        margins.append(r_eff - max(dist(p, s) for p in region_pts))
    return min(margins)


def _clip_halfplane(poly: list[Point], a: Point, b: Point) -> list[Point]:
    """Sutherland–Hodgman：保留有向直线 a→b **左侧**（叉积 ≥ 0）的部分。"""
    if not poly:
        return []
    ab = sub(b, a)
    out: list[Point] = []
    m = len(poly)
    for i in range(m):
        p, q = poly[i], poly[(i + 1) % m]
        cp = cross(ab, sub(p, a))
        cq = cross(ab, sub(q, a))
        if cp >= -_EPS:
            out.append(p)
        if (cp > _EPS and cq < -_EPS) or (cp < -_EPS and cq > _EPS):
            t = cp / (cp - cq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def clip_polygon(subject: list[Point], clipper: list[Point]) -> list[Point]:
    """用**凸**多边形 clipper（逆时针）裁剪 subject。

    裁剪边的过滤：设边所在直线的一次式 f(p) = cross(d, p−a)（d = b−a），
    保留侧是 f ≥ 0。若 subject 的**包围盒**整个落在 f ≥ −_EPS 内，则该边
    切不到任何东西，直接跳过。

    跳过是**位精确**的：此时每条边都满足 cp ≥ −_EPS（顶点全被复制），
    且 cq ≥ fmin ≥ −_EPS 使 `cq < −_EPS` 恒假（不会产生交点），
    与不跳过的输出逐位相同。收益是实打实的：K=360 的圆盘被 4 个顶点的
    小多边形裁剪时，真正会切的边只有个位数，其余 350+ 次全被省掉。
    """
    if not subject:
        return []
    out = list(subject)

    def _bbox(poly: list[Point]) -> tuple[float, float, float, float]:
        return (
            min(p[0] for p in poly),
            max(p[0] for p in poly),
            min(p[1] for p in poly),
            max(p[1] for p in poly),
        )

    # 包围盒只在**真的被切过**之后才重算。绝大多数边切不到任何东西，
    # 于是 O(K) 的过滤总共只花 O(K)，而不是 O(K·|subject|)。
    x0, x1, y0, y1 = _bbox(out)
    m = len(clipper)
    for i in range(m):
        a, b = clipper[i], clipper[(i + 1) % m]
        # f(p) = aa·p.x + bb·p.y + cc，保留侧 f ≥ 0
        dx, dy = b[0] - a[0], b[1] - a[1]
        aa, bb = -dy, dx
        cc = dy * a[0] - dx * a[1]
        fmin = aa * (x0 if aa > 0 else x1) + bb * (y0 if bb > 0 else y1) + cc
        if fmin >= -_EPS:
            continue                       # 整块都在保留侧，此边切不到
        out = _clip_halfplane(out, a, b)
        if not out:
            return []
        x0, x1, y0, y1 = _bbox(out)
    return out


def omega_polygon(
    sites: list[Point],
    arena_r: float = 1800.0,
    r_eff: float = DEFAULT_R_EFF,
    segments: int = 360,
) -> list[Point]:
    """Ω 的**内接** K 边形近似（保守：结果 ⊆ 真实 Ω）。

    ⚠ **不要用这个函数去参与热路径**。它从场地 K 边形出发，被每个站点圆
    逐个裁剪，代价是 O(K²)——K=360 时单次约 **86 ms**，比整条 tier-1 路径
    慢 800 倍。`region_polygon` 走的是等价但廉价的顺序（把小多边形拿去被
    大 K 边形裁，O(K·m)，m 是个位数），本函数只留给画图与诊断。

    数学上两种顺序给出同一个交集（凸集之交可交换），差别纯粹在算术路径。
    """
    poly = circle_polygon((0.0, 0.0), arena_r, segments)
    for s in sites:
        poly = clip_polygon(poly, circle_polygon(s, r_eff, segments))
        if not poly:
            return []
    return poly


def region_corners(
    sites: list[Point], bearings: list[float], err_deg: float = 1.0
) -> list[Point]:
    """角域之交的顶点（**不含 Ω**），按 uncertainty_corners 的契约。

    n=2 时与 uncertainty_corners 返回同一**多重集**（顺序不同：
    这里过了一次凸包，那边保持枚举序）。
    """
    rays: list[tuple[Point, float]] = []
    for s, th in zip(sites, bearings):
        rays.append((s, th - err_deg))
        rays.append((s, th + err_deg))
    kept = [
        p
        for p in candidate_vertices(rays)
        if all(wedge_ok(s, th, err_deg, p) for s, th in zip(sites, bearings))
    ]
    return convex_hull(kept)


def region_polygon(
    sites: list[Point],
    bearings: list[float],
    err_deg: float = 1.0,
    arena_r: float = 1800.0,
    r_eff: float = DEFAULT_R_EFF,
    segments: int = 360,
    force_omega: bool = False,
) -> tuple[list[Point], bool, str]:
    """定位区域的多边形，返回 (顶点, Ω 是否活化, 退化原因)。

    三层同一入口：
      tier 1  楔形之交有界且 Ω 严格不活化 → 零误差的精确解
      tier 2a 楔形之交有界但 Ω 活化     → 与 Ω 的 K 边形求交
      tier 2b 楔形之交无界             → 以 Ω 为种子，被 2n 个半平面裁剪
    """
    if len(sites) != len(bearings):
        return [], False, f"站点数 {len(sites)} 与示向度 {len(bearings)} 不匹配"
    if not sites:
        return [], False, "没有检测点"

    closed = wedge_bounded(bearings, err_deg)
    poly: list[Point] = []
    if closed:
        poly = region_corners(sites, bearings, err_deg)

    # tier 1：只在楔形自封闭且有实际面积时才可能走精确解
    if closed and len(poly) >= 3 and not force_omega:
        if omega_inactive_margin(poly, sites, arena_r, r_eff) > 0.0:
            return poly, False, ""

    # tier 2：需要 Ω。**裁剪顺序决定代价**：始终让「小多边形」去被「大 K 边形」
    # 裁，代价 O(K·m)；反过来从场地 K 边形出发逐个裁，代价 O(K²)（86 ms/次）。
    # 凸集之交可交换，两种顺序结果相同，差别只在算术路径。
    arena_poly = circle_polygon((0.0, 0.0), arena_r, segments)
    if closed and len(poly) >= 3:
        out = clip_polygon(poly, arena_poly)                  # tier 2a：小 ∩ 大
    else:
        out = list(arena_poly)                                # tier 2b：只能从场地起
        for s, th in zip(sites, bearings):
            for sgn, keep_left in ((-1.0, True), (1.0, False)):
                # sgn=-1：要求 cross(u, p−S) ≥ 0  ⟹ 保留 u 向量的左侧
                # sgn=+1：要求 cross(v, p−S) ≤ 0  ⟹ 保留 v 向量的右侧
                u = unit_vector(th + sgn * err_deg)
                far = (s[0] + u[0], s[1] + u[1])
                out = (
                    _clip_halfplane(out, s, far)
                    if keep_left
                    else _clip_halfplane(out, far, s)
                )
            if not out:
                break
    # 站点圆盘：此时 out 已是被裁小的一方（阶数 ≤ 2n + 少许），仍走廉价方向
    for s in sites:
        if not out:
            break
        out = clip_polygon(out, circle_polygon(s, r_eff, segments))

    if len(out) < 3 or abs(polygon_area(out)) < 1e-9:
        return [], True, "区域为空（角域与 Ω 无交）"
    return out, True, ""


@dataclass(frozen=True)
class EnclosingCircle:
    """最小包围圆。`support` 是确定它的点下标（2 点=直径对，3 点=外接圆）。"""

    center: Point
    radius: float
    kind: str                                   # "2点(直径对)" | "3点(外接圆)" | "退化"
    support: tuple[int, ...] = ()


def min_enclosing_circle(pts: list[Point]) -> EnclosingCircle | None:
    """最小包围圆（圆心 + 半径 + 确定集 + 三角形分类）。

    ⚠ 这是 `covering_radius` 的**孪生体，不是重构**：逐字镜像它的循环结构
    （先点对、后三元组，同样的 `covers`/严格 `<`、同样的 `dist`/`circumcircle`/
    `_covers_all` 调用、同样 `< 1e-9` 容差），以保证两者半径**位精确相等**。
    `covering_radius` 只给半径、没有圆心与确定集，论文解释需要后者；
    改前者会让 14 项既有回归失去意义，故宁可留一份平行实现 + 一条严格 `==` 自检。
    """
    n = len(pts)
    if n == 0:
        return None
    if n == 1:
        return EnclosingCircle(pts[0], 0.0, "退化", (0,))

    best = float("inf")
    best_c: EnclosingCircle | None = None

    for i in range(n):
        for j in range(i + 1, n):
            ctr = ((pts[i][0] + pts[j][0]) / 2, (pts[i][1] + pts[j][1]) / 2)
            r = dist(pts[i], pts[j]) / 2
            if _covers_all(ctr, r, pts):
                if r < best:
                    best, best_c = r, EnclosingCircle(
                        ctr, r, "2点(直径对)", (i, j)
                    )

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                cc = circumcircle(pts[i], pts[j], pts[k])
                if cc is not None and _covers_all(cc[0], cc[1], pts):
                    if cc[1] < best:
                        # 分类：确定集三角形的最大内角
                        tri = (pts[i], pts[j], pts[k])
                        ang = max(
                            _angle_at(tri[0], tri[1], tri[2]),
                            _angle_at(tri[1], tri[0], tri[2]),
                            _angle_at(tri[2], tri[0], tri[1]),
                        )
                        kind = "3点(外接圆)"
                        if ang > 90.0 + 1e-9:
                            kind = "3点(钝角——不应出现)"
                        best, best_c = cc[1], EnclosingCircle(
                            cc[0], cc[1], kind, (i, j, k)
                        )

    if best_c is None:
        # 理论上不可达（点对枚举必有一个可行解），兜底保证不返回 None
        r = max(dist(pts[0], p) for p in pts)
        return EnclosingCircle(pts[0], r, "退化", (0,))
    return best_c


def _angle_at(vertex: Point, a: Point, b: Point) -> float:
    """∠a-vertex-b，单位度，取值 [0,180]。"""
    u, v = sub(a, vertex), sub(b, vertex)
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(c))


@dataclass
class Region:
    """任意 n 个检测点的定位区域及其全部派生量。"""

    sites: list[Point]
    bearings: list[float]
    err_deg: float = 1.0
    polygon: list[Point] = field(default_factory=list)      # 最终区域（含 Ω）
    corners_raw: list[Point] = field(default_factory=list)  # 角域之交的候选顶点
    m: int = 0
    area: float = 0.0
    bounded: bool = False
    convex: bool = False
    omega_active: bool = False
    omega_margin: float = 0.0
    nominal: Point | None = None
    diameter: float = 0.0
    diameter_ends: tuple[Point, Point] | None = None
    diameter_circle_ok: bool = False
    mec: EnclosingCircle | None = None
    degenerate: str = ""

    @property
    def cover_r(self) -> float:
        return 0.0 if self.mec is None else self.mec.radius

    @property
    def cover_margin(self) -> float:
        """直径圆对最远顶点的**余量**（正 = 覆盖）。"""
        if self.diameter_ends is None:
            return 0.0
        a, b = self.diameter_ends
        m = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        r = self.diameter / 2
        return r - max(dist(m, p) for p in self.polygon)

    @property
    def clear_ok(self) -> bool:
        return self.bounded and self.cover_r <= CLEAR_RADIUS


def localization_region(
    sites: list[Point],
    bearings: list[float],
    err_deg: float = 1.0,
    arena_r: float = 1800.0,
    r_eff: float = DEFAULT_R_EFF,
    segments: int = 360,
    force_omega: bool = False,
) -> Region:
    """问题 1 的一般 n 版本：算区域的直径、覆盖判定与最小包围圆。"""
    reg = Region(
        sites=list(sites), bearings=list(bearings), err_deg=err_deg
    )

    corners = region_corners(sites, bearings, err_deg)
    # Ω 余量必须在**裁剪前**算：裁剪后的多边形顶点本就落在 Ω 边界上，
    # 余量恒 ≈0，毫无诊断价值。这个数是「Ω 到底想不想咬」的证书。
    reg.omega_margin = (
        omega_inactive_margin(corners, sites, arena_r, r_eff)
        if len(corners) >= 3
        else float("-inf")
    )
    ray_corners: list[tuple[Point, float]] = []
    for s, th in zip(sites, bearings):
        ray_corners.append((s, th - err_deg))
        ray_corners.append((s, th + err_deg))
    reg.corners_raw = [
        p
        for p in candidate_vertices(ray_corners)
        if all(wedge_ok(s, th, err_deg, p) for s, th in zip(sites, bearings))
    ]

    poly, omega_active, why = region_polygon(
        sites, bearings, err_deg, arena_r, r_eff, segments, force_omega
    )
    reg.polygon = poly
    reg.omega_active = omega_active
    reg.m = len(poly)
    reg.area = abs(polygon_area(poly))
    reg.convex = is_convex_ccw(poly)
    if why:
        reg.degenerate = why
        return reg
    if not poly:
        reg.degenerate = "区域为空"
        return reg

    reg.bounded = True
    reg.diameter, reg.diameter_ends = max_pairwise_distance(poly)
    if reg.diameter_ends is not None:
        a, b = reg.diameter_ends
        reg.diameter_circle_ok = diameter_circle_covers(a, b, poly)
    reg.mec = min_enclosing_circle(poly)

    # 标称交点：两条中心射线（n=2 时可解；n>2 时取最小二乘意义下的常识近似——
    # 这里只做 n=2 的精确解，n>2 交给后续的定位器，故留 None）
    if len(sites) == 2:
        hit = ray_intersection(
            sites[0], bearings[0], sites[1], bearings[1]
        )
        if hit is not None:
            reg.nominal = hit[0]
    return reg


# ── 回归测试 ────────────────────────────────────────────────
# 数据来源：2026-09-10 官方模拟器「问题3演练」，同一频道 2 的两条示向度。
# 期望值取自 strategy_notes.md 中已人工核算过的结果。
_REAL_S1, _REAL_TH1 = (300.0, 400.0), 214.82
_REAL_S2, _REAL_TH2 = (300.0, 0.0), 192.75

# 这两条射线在数学坐标系下给出正距离；若改用罗盘方位角，两个距离会变成
# −234.9 m 与 −607.9 m —— 几何上不成立。这就是坐标系约定的反证依据。
_EXPECT_R1 = 1038.3
_EXPECT_R2 = 874.0
_EXPECT_CROSS = 22.07


def _self_test() -> bool:
    print("=" * 66)
    print("geometry.py 回归测试 —— 官方模拟器实测数据")
    print("=" * 66)

    loc = localize(_REAL_S1, _REAL_TH1, _REAL_S2, _REAL_TH2)
    print(loc.summary())
    print()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # 1. 坐标系判据：数学坐标系下两个距离都为正，且量级与实测一致
    check(
        "标称交点存在且两段距离为正",
        loc.center is not None and loc.r1 > 0 and loc.r2 > 0,
        f"r1={loc.r1:.1f} r2={loc.r2:.1f}",
    )
    check(
        f"r1 ≈ {_EXPECT_R1} m",
        abs(loc.r1 - _EXPECT_R1) < 3.0,
        f"实际 {loc.r1:.1f}",
    )
    check(
        f"r2 ≈ {_EXPECT_R2} m",
        abs(loc.r2 - _EXPECT_R2) < 3.0,
        f"实际 {loc.r2:.1f}",
    )
    check(
        f"交会角 γ ≈ {_EXPECT_CROSS}°",
        abs(loc.cross_deg - _EXPECT_CROSS) < 0.02,
        f"实际 {loc.cross_deg:.2f}",
    )

    # 2. 交点必须真的在两条射线的正前方——用角度回代自洽校验
    if loc.center is not None:
        cx, cy = loc.center
        back1 = math.degrees(math.atan2(cy - _REAL_S1[1], cx - _REAL_S1[0])) % 360
        back2 = math.degrees(math.atan2(cy - _REAL_S2[1], cx - _REAL_S2[0])) % 360
        check(
            "交点回代 S1 的方向角 = θ1",
            norm180(back1 - _REAL_TH1) < 0.01,
            f"{back1:.2f}°",
        )
        check(
            "交点回代 S2 的方向角 = θ2",
            norm180(back2 - _REAL_TH2) < 0.01,
            f"{back2:.2f}°",
        )

    # 3. 不确定区域
    check("不确定区域有界（4 个顶点）", loc.bounded, loc.degenerate)

    # 4. 实测结论：直径远大于清除半径，故远场交会不足以直接清除
    if loc.bounded:
        check(
            "直径 > 清除半径×2（远场交会不够，必须逼近）",
            loc.diameter > 2 * CLEAR_RADIUS,
            f"直径 {loc.diameter:.1f} m vs 清除直径 {2 * CLEAR_RADIUS:.0f} m",
        )
        # Jung 定理的**锐界**：D/2 ≤ r_MEC ≤ D/√3。
        # 原先这里写的是 r_MEC ≤ D 与 D ≥ r_MEC（同一条、且过松），
        # 放过了 r_MEC 偏小 2 倍的实现错误。
        jung_lo, jung_hi = loc.diameter / 2, loc.diameter / math.sqrt(3)
        check(
            "r_MEC ≥ D/2（Jung 下界，命题 A）",
            loc.cover_r >= jung_lo - 1e-6,
            f"r_MEC={loc.cover_r:.4f} ≥ D/2={jung_lo:.4f}",
        )
        check(
            "r_MEC ≤ D/√3（Jung 上界）",
            loc.cover_r <= jung_hi + 1e-6,
            f"r_MEC={loc.cover_r:.4f} ≤ D/√3={jung_hi:.4f}",
        )
        # 覆盖与 r_MEC 的等价命题：覆盖 ⟺ r_MEC = D/2（MEC 由直径对或直角三角形确定）
        eq = abs(loc.cover_r - jung_lo) < 1e-6
        check(
            "覆盖 ⟺ r_MEC = D/2（等价命题自洽）",
            loc.diameter_circle_ok == eq,
            f"覆盖={loc.diameter_circle_ok}  r_MEC−D/2={loc.cover_r - jung_lo:+.2e}",
        )

    # 5. 退化路径不应被误触发
    far = localize((0.0, 0.0), 45.0, (1000.0, 0.0), 45.0)
    check("近似平行射线被正确判为退化", not far.bounded and bool(far.degenerate),
          far.degenerate)

    # 6. 回归：一次真实的「直径圆**不**覆盖」算例（2026-09-11）
    #    见 repro_cover_fail.py。D/E 两站，交会角 ~91°，区域近矩形，
    #    I 点以 0.431 m 落在直径圆外 —— 验证算法能给出「否」，不是写死「是」。
    fail_c = localize(
        (1158.5208537908559, -801.3474829903406), 143.589584,
        (-211.7517570274545, 1109.3446610684468), 234.609399,
    )
    check(
        "回归：已知「不覆盖」算例仍判为不覆盖",
        fail_c.bounded and not fail_c.diameter_circle_ok,
        f"直径={fail_c.diameter:.4f} m  覆盖={fail_c.diameter_circle_ok}",
    )
    check(
        "回归：该算例 r_MEC > D/2（MEC 由锐角三角形确定）",
        fail_c.cover_r > fail_c.diameter / 2,
        f"r_MEC={fail_c.cover_r:.6f} > D/2={fail_c.diameter / 2:.6f}",
    )

    # ── 新增自检（2026-09-11，任意 n 的基础设施）──────────────
    # 7. wedge_ok：6 点手算对照。
    #    站点取原点、θ=0°、ε=1°，故楔形 = 方向角落在 [−1°, +1°] 的射线。
    #    坐标一律**手算写字面量**，绝不用 unit_vector 现推——否则测的
    #    是「同一段代码自洽」而不是「实现与外部的约定一致」。
    #    ⚠ 第 6 点是**镜像点**（方向 180°）：它是唯一能抓住「叉积次序写反
    #      ⟹ 楔形静默旋转 180°」那个 bug 的点。历史教训，不可删。
    #    内/外各取 0.9° 与 1.1°（离边界 0.1°），而非 0.5°/1.5°——因为手写的
    #    十进制字面量本身有 ~5e-8 的舍入误差，在 r=10 处折合约 5e-9 的
    #    法向偏移。**贴着边界写手算点是在测数据精度，不是在测代码**，
    #    所以「闭边界」这一条另立 check 13，用精确构造去测。
    _W6 = [
        ("内一点 方向 −0.9°", (9.9987663, -0.1570732), True),
        ("内一点 方向 +0.9°", (9.9987663, 0.1570732), True),
        ("外一点 方向 −1.1°", (9.9981569, -0.1919744), False),
        ("外一点 方向 +1.1°", (9.9981569, 0.1919744), False),
        ("外一点 方向 180°（正后方）", (-10.0, 0.0), False),
        ("镜像点 方向 179°（180° 旋转的另一支）", (-9.9984769, 0.1745241), False),
    ]
    wrong = [n for n, q, want in _W6 if wedge_ok((0.0, 0.0), 0.0, 1.0, q) != want]
    check(
        "wedge_ok：6 点手算对照（含 180° 镜像点）",
        not wrong,
        "全部一致" if not wrong else f"错判 {wrong}",
    )

    # 8. wedge_ok 与 uncertainty_corners 自洽：四角必须同时落在两站楔形内，
    #    且每站至少一角**恰好**贴在自己的边界上（否则 ε 没接上）。
    if loc.bounded:
        sites = ((_REAL_S1, _REAL_TH1), (_REAL_S2, _REAL_TH2))
        in_all = all(
            wedge_ok(s, th, 1.0, c) for c in loc.corners for s, th in sites
        )
        def _off_from(site: Point, th: float, q: Point) -> float:
            """q 相对 site 的示向度与 θ 的未折叠夹角。"""
            return abs(
                norm180(math.degrees(math.atan2(q[1] - site[1], q[0] - site[0])) - th)
            )

        on_edge = all(
            any(abs(_off_from(s, th, c) - 1.0) < 1e-9 for s, th in sites)
            for c in loc.corners
        )
        check(
            "wedge_ok 与 uncertainty_corners 自洽（四角在两楔形内）",
            in_all,
            f"{sum(wedge_ok(s, th, 1.0, c) for c in loc.corners for s, th in sites)}"
            f"/{len(loc.corners) * len(sites)} 通过",
        )
        check(
            "每个顶点都恰好压在某一站的边界上",
            on_edge,
            "ε 已在楔形判据中生效" if on_edge else "有顶点不在任何边界上",
        )

    # 9. ★ 候选顶点枚举序与 uncertainty_corners **逐位一致**（n=2）
    #    这是「只增不改」的硬约束：ray_intersection 收到同样的实参，
    #    就必须吐出同样的浮点结果。用 list ==（含顺序）比较，**不放宽容差**。
    rays2 = [
        (_REAL_S1, _REAL_TH1 - 1.0), (_REAL_S1, _REAL_TH1 + 1.0),
        (_REAL_S2, _REAL_TH2 - 1.0), (_REAL_S2, _REAL_TH2 + 1.0),
    ]
    cand2 = candidate_vertices(rays2)
    bit_eq = cand2 == loc.corners
    check(
        "n=2 候选顶点与 uncertainty_corners 逐位一致",
        bit_eq,
        "4/4 位精确" if bit_eq else f"获得 {len(cand2)} 个，期望 {len(loc.corners)} 个",
    )

    # 10. wedge_bounded：只依赖示向度
    check(
        "wedge_bounded：n=1 无界 / n=2 平行无界 / n=2 直角有界",
        (not wedge_bounded([0.0]))
        and (not wedge_bounded([0.0, 0.0]))
        and wedge_bounded([0.0, 90.0]),
        f"1站={wedge_bounded([0.0])} 平行={wedge_bounded([0.0, 0.0])} "
        f"90°={wedge_bounded([0.0, 90.0])}",
    )

    # 11. 凸包 / 面积 / 逆时针自洽
    if loc.bounded:
        hull = convex_hull(loc.corners)
        check(
            "凸包：CCW、面积等于极角序面积、四角全在内",
            is_convex_ccw(hull)
            and abs(polygon_area(hull) - abs(polygon_area(polar_order(loc.corners)))) < 1e-9
            and len(hull) == 4
            and abs(polygon_area(hull)) > 0.0,
            f"m={len(hull)} 面积={abs(polygon_area(hull)):.1f} m² "
            f"凸={is_convex_ccw(hull)}",
        )

    # 12. circle_polygon：顶点严格在圆上，且多边形 ⊆ 圆盘（内接，保守）
    cp = circle_polygon((0.0, 0.0), 1800.0, 360)
    check(
        "circle_polygon：顶点全在圆上且严格内接（保守）",
        len(cp) == 360
        and max(abs(dist(q, (0.0, 0.0)) - 1800.0) for q in cp) < 1e-9
        and all(dist(q, (0.0, 0.0)) <= 1800.0 + 1e-9 for q in cp),
        f"K={len(cp)} 弓高={1800.0 * (1 - math.cos(math.pi / 360)):.4f} m",
    )

    # 13. 闭边界：楔形含其两条边界射线。
    #     这条是 uncertainty_corners 赖以成立的前提——四角恰在边界上，
    #     若判据把它们判为「外」，整个区域就会被裁成空集。
    #     用 unit_vector 精确构造（此处**必须**如此：要测的契约就是
    #     「凡按本模块约定沿 θ±ε 走出的点都被接受」，手写小数字面量反而不精确）。
    edge_bad = []
    for _th in (0.0, 30.0, 123.4, 214.82, 359.0):
        for _eps in (0.5, 1.0, 2.0):
            for _t in (1.0, 10.0, 1000.0):
                for _sgn in (-1.0, 1.0):
                    _u = unit_vector(_th + _sgn * _eps)
                    _q = (_t * _u[0], _t * _u[1])
                    if not wedge_ok((0.0, 0.0), _th, _eps, _q):
                        edge_bad.append((_th, _eps, _t, _sgn))
    check(
        "闭边界：沿 θ±ε 精确构造的点全部被接受",
        not edge_bad,
        "60/60 通过" if not edge_bad else f"漏 {len(edge_bad)} 个，如 {edge_bad[0]}",
    )

    # ── 新增自检 · B 部分（Ω / Region / MEC 孪生体）──────────
    # 14. ★ 一般 n 入口在 n=2 时与 localize() **逐位一致**。
    #     这是「只增不改」的第二条硬约束：新路径若漂 1 ULP，就是发现，不是容差问题。
    if loc.bounded:
        reg = localization_region([_REAL_S1, _REAL_S2], [_REAL_TH1, _REAL_TH2])
        same = (
            reg.diameter == loc.diameter
            and reg.cover_r == loc.cover_r
            and reg.diameter_circle_ok == loc.diameter_circle_ok
            and reg.m == len(loc.corners)
            and reg.nominal == loc.center
        )
        check(
            "★ n=2：localization_region 与 localize() 逐位一致",
            same,
            f"𝒟 {reg.diameter!r} vs {loc.diameter!r}｜"
            f"r_MEC {reg.cover_r!r} vs {loc.cover_r!r}｜m {reg.m} vs {len(loc.corners)}",
        )
        check(
            "n=2 走 tier 1（Ω 不活化，零误差）",
            not reg.omega_active and reg.omega_margin > 0.0,
            f"Ω活化={reg.omega_active} 余量={reg.omega_margin:.3f} m",
        )

    # 15. MEC 孪生体：半径必须与 covering_radius **严格相等**（位精确）。
    #     两边都是「点对 → 三元组」的同序枚举 + 同样的 `<` 比较，故可以要求 `==`；
    #     一旦不等，说明孪生体漂移了，必须查明而不是放宽容差。
    for _name, _pts in (
        ("数据集 A 四角", loc.corners),
        ("D/E 四角", fail_c.corners),
        ("正三角形", [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0 * math.sqrt(3))]),
        ("两点", [(0.0, 0.0), (7.0, 24.0)]),
        ("单点", [(3.0, 4.0)]),
    ):
        _me = min_enclosing_circle(_pts)
        _cr = covering_radius(_pts)
        check(
            f"MEC 孪生体与 covering_radius 严格相等（{_name}）",
            _me is not None and _me.radius == _cr,
            f"孪生体={None if _me is None else _me.radius!r} vs {_cr!r}",
        )

    # 16. 正三角形的 MEC 应 = 边长/√3 且由 3 点确定；矩形应由 2 点确定
    _tri = [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0 * math.sqrt(3))]
    _mec_tri = min_enclosing_circle(_tri)
    _rect = [(-5.0, 0.0), (5.0, 0.0), (5.0, 0.0 + 6.0), (-5.0, 6.0)]
    _mec_rect = min_enclosing_circle(_rect)
    check(
        "MEC 分类：正三角形由 3 点确定且 r=边长/√3；矩形由 2 点确定",
        _mec_tri is not None
        and len(_mec_tri.support) == 3
        and abs(_mec_tri.radius - 10.0 / math.sqrt(3.0)) < 1e-9
        and _mec_rect is not None
        and len(_mec_rect.support) == 2
        and abs(_mec_rect.radius - dist((-5.0, 0.0), (5.0, 6.0)) / 2) < 1e-9,
        f"三角 r={_mec_tri.radius:.6f} (期望 {10 / math.sqrt(3):.6f}) 支撑{len(_mec_tri.support)}点｜"
        f"矩形 r={_mec_rect.radius:.6f} 支撑{len(_mec_rect.support)}点",
    )

    # 17. Ω 缩放不变性：**两种缩放都要测**。
    #     只放大 arena_r 而站点圆不动，一个站点圆的 bug 能蒙混过关。
    if loc.bounded:
        _a = localization_region([_REAL_S1, _REAL_S2], [_REAL_TH1, _REAL_TH2])
        _b = localization_region(
            [_REAL_S1, _REAL_S2], [_REAL_TH1, _REAL_TH2], arena_r=1800.0 * 100
        )
        _c = localization_region(
            [_REAL_S1, _REAL_S2],
            [_REAL_TH1, _REAL_TH2],
            arena_r=1800.0 * 100,
            r_eff=1500.0 * 100,
        )
        check(
            "Ω 缩放不变性（arena_r×100 单独）",
            abs(_b.diameter - _a.diameter) / _a.diameter < 1e-6,
            f"Δ𝒟/𝒟={(abs(_b.diameter - _a.diameter) / _a.diameter):.2e}",
        )
        check(
            "Ω 缩放不变性（arena_r 与 r_eff 同时 ×100）",
            abs(_c.diameter - _a.diameter) / _a.diameter < 1e-6,
            f"Δ𝒟/𝒟={(abs(_c.diameter - _a.diameter) / _a.diameter):.2e}",
        )

    # 18. 强制 Ω 裁剪必须**开得动**且结果保守（子集 ⟹ 直径不增）。
    #     只声明「K 边形近似」却不演练这条路径，等于没实现。
    if loc.bounded:
        _f = localization_region(
            [_REAL_S1, _REAL_S2], [_REAL_TH1, _REAL_TH2], force_omega=True
        )
        # 保守性直接验：裁剪后的每个顶点都必须真的落在 Ω 的每个约束圆盘内。
        _in_omega = all(
            math.hypot(q[0], q[1]) <= 1800.0 + 1e-9
            and dist(q, _REAL_S1) <= 1500.0 + 1e-9
            and dist(q, _REAL_S2) <= 1500.0 + 1e-9
            for q in _f.polygon
        )
        check(
            "强制 Ω 裁剪：顶点全在 Ω 内、直径不增、覆盖判定仍自洽",
            _f.omega_active
            and _f.bounded
            and _in_omega
            and _f.diameter <= loc.diameter + 1e-9
            and _f.diameter_circle_ok
            == (abs(_f.cover_r - _f.diameter / 2) <= 1e-6),
            f"m={_f.m} 𝒟={_f.diameter:.6f} vs 精确 {loc.diameter:.6f} "
            f"（低估 {loc.diameter - _f.diameter:.6f} m）",
        )

    # 19. clearing_guaranteed 的半径必须**显式传入**才可靠
    if loc.bounded:
        check(
            "clearing_guaranteed：显式半径与 r_MEC 一致",
            clearing_guaranteed(loc.corners, loc.cover_r)
            and not clearing_guaranteed(loc.corners, loc.cover_r - 1.0)
            and clearing_guaranteed(loc.corners) == (loc.cover_r <= CLEAR_RADIUS),
            f"r_MEC={loc.cover_r:.4f} 默认={clearing_guaranteed(loc.corners)}",
        )

    # 20. n=2 的「有界 ⟺ γ > 2ε」律——把 tier 分派的判据本身钉死。
    #     γ = 2ε 恰在临界：两楔形的方向范围相切，仍共享一条边界射线 ⟹ 仍无界。
    _law_bad = [
        gam
        for gam in (0.5, 1.9, 1.99, 2.0, 2.01, 2.5, 5.0, 22.07, 90.0)
        if wedge_bounded([0.0, gam]) != (gam > 2.0)
    ]
    check(
        "n=2「有界 ⟺ γ > 2ε」律（γ=2ε 相切处仍无界）",
        not _law_bad,
        "9/9 一致" if not _law_bad else f"违例 {_law_bad}",
    )

    # 21/22. tier 2a 与 tier 2b —— 三条路径都必须被**真的走到**。
    #     均匀随机抽样几乎抽不到 γ≈2ε 的窄带，故这里手工构造。
    #     构造法：源置于 1500 m 环上（|S_i−G|∈[1000,1500] 的边界），
    #     两站对称张开 γ，则 γ 越小基线越短、区域越长。
    def _narrow(gamma_deg: float, r: float = 1499.0):
        d = 2.0 * r * math.tan(math.radians(gamma_deg) / 2.0)
        s1, s2 = (0.0, 0.0), (d, 0.0)
        g = (d / 2.0, r)
        b = [
            math.degrees(math.atan2(g[1] - s[1], g[0] - s[0])) % 360
            for s in (s1, s2)
        ]
        return s1, s2, b

    def _inside_omega(poly: list[Point], ss: list[Point], ar: float, re: float):
        return all(
            math.hypot(q[0], q[1]) <= ar + 1e-9
            and all(dist(q, s) <= re + 1e-9 for s in ss)
            for q in poly
        )

    # 21. tier 2a：γ=3°>2ε ⟹ 楔形之交有界，但细长区域远超场地 ⟹ Ω 活化
    _s1a, _s2a, _ba = _narrow(3.0)
    _t2a = localization_region([_s1a, _s2a], _ba)
    check(
        "tier 2a：楔形有界 + Ω 活化 → SH 裁剪生效且结果 ⊆ Ω",
        wedge_bounded(_ba)
        and _t2a.omega_active
        and _t2a.bounded
        and _t2a.omega_margin < 0.0
        and _inside_omega(_t2a.polygon, [_s1a, _s2a], 1800.0, 1500.0),
        f"γ={norm180(_ba[0] - _ba[1]):.3f}° 裁剪前余量={_t2a.omega_margin:.0f} m "
        f"𝒟={_t2a.diameter:.2f} m m={_t2a.m}",
    )

    # 22. tier 2b：γ=1.911°<2ε ⟹ 楔形之交**无界**，必须以 Ω 为种子反向裁剪
    _s1b, _s2b, _bb = _narrow(1.911)
    _t2b = localization_region([_s1b, _s2b], _bb)
    check(
        "tier 2b：楔形无界 → 以 Ω 为种子裁剪，结果有界且 ⊆ Ω",
        (not wedge_bounded(_bb))
        and _t2b.omega_active
        and _t2b.bounded
        and _t2b.m >= 3
        and _inside_omega(_t2b.polygon, [_s1b, _s2b], 1800.0, 1500.0),
        f"γ={norm180(_bb[0] - _bb[1]):.3f}° 𝒟={_t2b.diameter:.2f} m "
        f"面积={_t2b.area:.0f} m² m={_t2b.m}",
    )

    # 23. 无 Ω 的老接口在无界情形下**必须**说「退化」，不得静默返回直径 0
    _fb = localize(_s1b, _bb[0], _s2b, _bb[1])
    check(
        "无界情形：localize() 报退化而非静默给 D=0",
        (not _fb.bounded) and bool(_fb.degenerate) and _fb.diameter == 0.0,
        _fb.degenerate[:48] or "（未报退化——静默失败！）",
    )

    # 24. Ω 余量必须取自**裁剪前**：裁剪后的顶点落在 Ω 边界上，余量恒 ≈0。
    _post = omega_inactive_margin(_t2a.polygon, [_s1a, _s2a], 1800.0, 1500.0)
    check(
        "Ω 余量取自裁剪前（裁剪后它恒 ≈0，无诊断价值）",
        _t2a.omega_margin < -100.0 and abs(_post) < 1.0,
        f"裁剪前={_t2a.omega_margin:.0f} m  裁剪后={_post:+.3f} m",
    )

    # 25. clip_polygon 的包围盒过滤必须**位精确**（它是纯加速，不是近似）。
    #     这是热路径上唯一一处「跳过计算」，跳过就得逐位可证明。
    import random as _rnd

    _r = _rnd.Random(7)

    def _ref_clip(subject: list[Point], clipper: list[Point]) -> list[Point]:
        out = list(subject)
        for _i in range(len(clipper)):
            out = _clip_halfplane(
                out, clipper[_i], clipper[(_i + 1) % len(clipper)]
            )
            if not out:
                return []
        return out

    _diff = 0
    _trials = 0
    while _trials < 600:
        _subj = convex_hull(
            [(_r.uniform(-900, 900), _r.uniform(-900, 900)) for _ in range(_r.randint(3, 7))]
        )
        if len(_subj) < 3:
            continue
        _cl = circle_polygon(
            (_r.uniform(-300, 300), _r.uniform(-300, 300)),
            _r.uniform(400.0, 1200.0),
            _r.choice((36, 72, 120)),
        )
        _trials += 1
        if clip_polygon(_subj, _cl) != _ref_clip(_subj, _cl):
            _diff += 1
    check(
        "clip_polygon 的包围盒过滤位精确（纯加速，非近似）",
        _diff == 0,
        f"{_trials} 组对照，{_diff} 处差异",
    )

    # ── 输出 ────────────────────────────────────────────────
    print("-" * 66)
    passed = 0
    for name, ok, detail in checks:
        passed += ok
        print(f"  {'✓' if ok else '✗'} {name}" + (f"   [{detail}]" if detail else ""))
    print("-" * 66)
    print(f"{passed}/{len(checks)} 通过")
    print("=" * 66)
    return passed == len(checks)


if __name__ == "__main__":
    import sys

    sys.exit(0 if _self_test() else 1)
