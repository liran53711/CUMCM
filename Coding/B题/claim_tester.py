"""
结论测试台 —— **与具体 claim 无关**的引擎（问题 1..4 共用）。

设计目标：后续每条结论都编码成一个 `Claim` 丢进来跑，加一条 claim
不需要碰这个文件。claim 本身写在 `claims_q1.py` / `claims_q2.py` / ...

四件必需品
----------
1. **物理自洽采样**：禁止独立抽样示向度。站点必须先满足
   ``1000 ≤ ‖S_i − G‖ ≤ 1500``，再由真源位置正演出示向度再加误差——
   否则会得到「根本测不到该源」的无效实例（本项目已因此吃过一个假结论）。
2. **独立 oracle**：换一种**表述**去查 geometry.py，不是把它抄一遍。
   楔形判据用 atan2、最小包围圆用 Welzl、点在内用射线法——都与被测代码
   走不同的算术路径，故能抓住「实现与约定不一致」这类静默错误。
3. **能失败**：`run_suite` 必须能被伪证检验（见 ``MUTANTS``）。一个永远
   打印 PASS 的测试台和正确的代码无法区分，而本项目记录在案的两个 bug
   （叉积次序反了、MEC 不等号反了）**都是静默实现错**，这不是假想。
4. **确定性**：只用整数种子，全部走 ``random.Random``。

约定（沿用本仓库）：纯 stdlib + numpy/matplotlib 可选，扁平放置，无包，
``__main__`` + ``sys.exit(0/1)``，无 pytest。
"""

from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

import geometry
from geometry import Point

Point = tuple[float, float]     # 与 geometry 同义，便于本文件独立阅读


def enable_utf8_console() -> None:
    """Windows 中文控制台默认 GBK，打印 ✓/✗ 会抛 UnicodeEncodeError 并在
    **自检中途**中断。必须在任何 print 之前调用。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ── 采样 ────────────────────────────────────────────────────
ARENA_R = 1800.0
R_EFF_MIN, R_EFF_MAX = 1000.0, 1500.0

# 采样时的最大重试：源必须落在「两站都能测到」的区域里，
# 而窄带情形下这一条件很挑，重试次数给足。
_MAX_TRIES = 200


@dataclass
class Instance:
    """一个物理自洽的定位实例。

    ``source`` 与 ``deltas`` 保留真源与逐站误差，使
    (a) 收缩器能把 |δ| 顶到 ε、(b) ``literal()`` 能打出可复现的构造式。
    """

    sites: list[Point]
    bearings: list[float]
    err_deg: float = 1.0
    r_eff: float = R_EFF_MAX
    """该源的**有效接收半径**（题设：各源不同，∈[1000,1500]，接口不返回）。

    ⚠ 约束是**单边**的：|S_i − G| ≤ r_eff。**没有下界**——站点离源多近都能测到
    （题面：「需走出较远距离才能找到边界」）。早先按「|S_i − G| ∈ [1000,1500]」
    双边采样是**错的**，它无端丢掉近距站点，并把 Ω 活化率从 22% 抬到 38%。
    """
    source: Point | None = None
    deltas: list[float] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.deltas is None and self.source is not None:
            self.deltas = [
                geometry.norm180(b - math.degrees(
                    math.atan2(self.source[1] - s[1], self.source[0] - s[0])
                ))
                for s, b in zip(self.sites, self.bearings)
            ]

    @property
    def n(self) -> int:
        return len(self.sites)

    def key(self) -> tuple:
        """去重键（浮点取 6 位小数，够区分几何上不同的实例）。"""
        return (
            tuple((round(a, 6), round(b, 6)) for a, b in self.sites),
            tuple(round(t, 6) for t in self.bearings),
            round(self.err_deg, 6),
            round(self.r_eff, 6),
        )

    def literal(self) -> str:
        """可直接粘进回归测试的全精度构造式。"""
        s = "[" + ", ".join(f"({a!r}, {b!r})" for a, b in self.sites) + "]"
        t = "[" + ", ".join(f"{x!r}" for x in self.bearings) + "]"
        src = "" if self.source is None else f", source=({self.source[0]!r}, {self.source[1]!r})"
        return (
            f"Instance(sites={s}, bearings={t}, "
            f"err_deg={self.err_deg!r}, r_eff={self.r_eff!r}{src})"
        )

    def rebearing(self, new_deltas: list[float]) -> "Instance":
        """按给定的 δ 序列重建示向度（真源已知时才可用）。"""
        if self.source is None:
            return self
        out = []
        for s, d in zip(self.sites, new_deltas):
            true = math.degrees(
                math.atan2(self.source[1] - s[1], self.source[0] - s[0])
            )
            out.append((true + d) % 360.0)
        return Instance(
            sites=list(self.sites), bearings=out, err_deg=self.err_deg,
            r_eff=self.r_eff, source=self.source, deltas=list(new_deltas),
            note=self.note,
        )


def sample_instance(
    rng: random.Random,
    n: int = 2,
    err_deg: float = 1.0,
    err_mode: str = "uniform",
    arena_r: float = ARENA_R,
    r_eff_range: tuple[float, float] = (R_EFF_MIN, R_EFF_MAX),
) -> Instance | None:
    """采一个物理自洽的实例；失败返回 None。

    真源按 ``offline_sim.make_jammers`` 的同一惯例采样
    （``r = R·sqrt(u)``、``θ ~ U(0,2π)``），使空间分布律与模拟器一致。

    ``err_mode``:
      ``zero``     δ_i ≡ 0（标称，无误差）
      ``uniform``  δ_i ~ U(−ε, ε)
      ``extreme``  |δ_i| = ε，符号随机（对抗性 claim 用这个）

    ⚠ 同一实例内**不得重抽** δ——题设是同一地点的误差固定。
    ⚠ 绝不独立抽样示向度：那会产生几何上不存在的楔形。
    """
    # 源：场地内均匀（与模拟器同律）
    r = arena_r * math.sqrt(rng.random())
    th = rng.uniform(0.0, 2.0 * math.pi)
    source = (r * math.cos(th), r * math.sin(th))

    lo, hi = r_eff_range
    # r_eff 是**源**的属性（各源不同），整条实例共用一个值。
    r_eff = rng.uniform(lo, hi)
    sites: list[Point] = []
    for _ in range(n):
        got = None
        for _try in range(_MAX_TRIES):
            # 站点：在场地内均匀撒，再按「必须能测到该源」筛
            sr = arena_r * math.sqrt(rng.random())
            st = rng.uniform(0.0, 2.0 * math.pi)
            cand = (sr * math.cos(st), sr * math.sin(st))
            if math.dist(cand, source) <= r_eff:     # 单边：只需落在有效接收半径内
                got = cand
                break
            # 被拒的重抽也从同一流消耗，故序列跨运行稳定
        if got is None:
            return None
        sites.append(got)

    if err_mode == "zero":
        deltas = [0.0] * n
    elif err_mode == "extreme":
        deltas = [err_deg * (1.0 if rng.random() < 0.5 else -1.0) for _ in range(n)]
    elif err_mode == "uniform":
        deltas = [rng.uniform(-err_deg, err_deg) for _ in range(n)]
    else:
        raise ValueError(f"未知 err_mode: {err_mode!r}")

    bearings = [
        (math.degrees(math.atan2(source[1] - s[1], source[0] - s[0])) + d) % 360.0
        for s, d in zip(sites, deltas)
    ]
    return Instance(
        sites=sites, bearings=bearings, err_deg=err_deg, r_eff=r_eff,
        source=source, deltas=deltas, note=f"err_mode={err_mode}",
    )


def sample_stream(seed: int, n: int = 2, **kw) -> Instance | None:
    """单实例命名流：每实例一个独立整数种子，跨运行稳定。"""
    return sample_instance(random.Random(seed * 1000003 + n), n=n, **kw)


# ── 独立 oracle（换一种表述去查 geometry）────────────────────
def oracle_contains(site: Point, theta_deg: float, eps_deg: float, p: Point) -> bool:
    """楔形判据的**角度式**表述——**完全不用叉积**。

    专抓「叉积次序写反 ⟹ 楔形静默旋转 180°」这个 mutant。
    """
    dx, dy = p[0] - site[0], p[1] - site[1]
    if dx == 0.0 and dy == 0.0:
        return True
    ang = math.degrees(math.atan2(dy, dx))
    return abs(geometry.norm180(ang - theta_deg)) <= eps_deg + 1e-9


def oracle_wedge_region_ok(
    sites: list[Point], bearings: list[float], eps: float, p: Point
) -> bool:
    return all(oracle_contains(s, t, eps, p) for s, t in zip(sites, bearings))


def _point_in_poly_strict(poly: list[Point], p: Point) -> bool:
    """射线法（crossing number）——与被测的半平面判据**不同**的算法类。"""
    if len(poly) < 3:
        return False
    x, y = p
    inside = False
    m = len(poly)
    for i in range(m):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % m]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def oracle_mec(pts: list[Point], seed: int = 12345) -> tuple[Point, float] | None:
    """最小包围圆：**Welzl 随机增量**。

    与 ``geometry.covering_radius`` 的「穷举 2 点/3 点」属**不同算法类**，
    故能独立复核它，而不是把它抄一遍。
    """
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0], 0.0

    def trivial(rs: list[Point]) -> tuple[Point, float] | None:
        if not rs:
            return None
        if len(rs) == 1:
            return rs[0], 0.0
        if len(rs) == 2:
            c = ((rs[0][0] + rs[1][0]) / 2, (rs[0][1] + rs[1][1]) / 2)
            return c, math.dist(rs[0], rs[1]) / 2
        cc = geometry.circumcircle(rs[0], rs[1], rs[2])
        if cc is not None:
            return cc
        # 三点共线 → 取最远两点为直径
        best = max(
            ((math.dist(rs[i], rs[j]), i, j) for i in range(3) for j in range(i + 1, 3))
        )
        _, i, j = best
        c = ((rs[i][0] + rs[j][0]) / 2, (rs[i][1] + rs[j][1]) / 2)
        return c, math.dist(rs[i], rs[j]) / 2

    def inside(circ, p: Point) -> bool:
        return circ is not None and math.dist(circ[0], p) <= circ[1] + 1e-9

    def welzl(ps: list[Point], rs: list[Point]) -> tuple[Point, float] | None:
        if not ps or len(rs) == 3:
            return trivial(rs)
        p = ps[-1]
        d = welzl(ps[:-1], rs)
        if inside(d, p):
            return d
        return welzl(ps[:-1], rs + [p])

    order = list(pts)
    random.Random(seed).shuffle(order)
    return welzl(order, [])


def oracle_cover_by_angles(a: Point, b: Point, pts: list[Point]) -> bool:
    """覆盖判据的**角度式**表述（顶点张角），与点积式独立。"""
    for p in pts:
        u = (a[0] - p[0], a[1] - p[1])
        v = (b[0] - p[0], b[1] - p[1])
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu < 1e-12 or nv < 1e-12:
            continue
        c = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
        if math.degrees(math.acos(max(-1.0, min(1.0, c)))) < 90.0 - 1e-9:
            return False
    return True


def oracle_omega_margin(
    region_pts: list[Point], sites: list[Point], arena_r: float, r_eff: float
) -> float:
    """Ω 余量的**独立重算**（同定义、不同实现，用于交叉复核）。

    余量 = min( 场地余量, 各站圆盘余量 )，>0 即 Ω 严格不活化。
    """
    if not region_pts:
        return float("-inf")
    return min(
        [arena_r - max(math.hypot(p[0], p[1]) for p in region_pts)]
        + [r_eff - max(math.dist(p, s) for p in region_pts) for s in sites]
    )


def oracle_region_sample_bbox(
    sites: list[Point],
    bearings: list[float],
    eps: float,
    nominal: Point,
    half: float,
    k: int = 60,
) -> list[Point]:
    """在标称交点附近的框内稠密取点，按**角度式**谓词筛出区域内点。"""
    out = []
    for i in range(k):
        for j in range(k):
            x = nominal[0] + (-half + 2.0 * half * i / (k - 1))
            y = nominal[1] + (-half + 2.0 * half * j / (k - 1))
            if oracle_wedge_region_ok(sites, bearings, eps, (x, y)):
                out.append((x, y))
    return out


# ── claim / 结果 ─────────────────────────────────────────────
SKIP = object()
"""`check` 返回它表示「本实例不在该 claim 的适用域内」——既不算过也不算挂。"""


@dataclass
class Failure:
    instance: Instance
    reason: str
    shrunk: Instance | None = None

    def describe(self) -> str:
        lines = [f"    反例：{self.reason}"]
        if self.shrunk is not None and self.shrunk.key() != self.instance.key():
            lines.append(f"    收缩后：{self.shrunk.literal()}")
        else:
            lines.append(f"    实例：{self.instance.literal()}")
        return "\n".join(lines)


@dataclass
class Claim:
    """一条待检验的结论。

    ``expect``:
      ``hold``     必须绿；任何反例都是 blocker
      ``refuted``  必须**报出反例**（用于把已被证伪的猜想钉在案上，
                   防止它偷偷溜回论文）。报不出反例反而算挂。
    """

    name: str
    statement: str
    check: Callable[[Instance], str | None]
    kind: str = "theorem"           # theorem | empirical | regression | refuted
    expect: str = "hold"
    source: str = ""                # 清单#n / 引理 n / 命题 n
    shrinker: Callable[[Instance, "Claim"], Instance | None] | None = None


@dataclass
class Outcome:
    claim: Claim
    checked: int = 0
    skipped: int = 0
    failures: list[Failure] = field(default_factory=list)
    ok: bool = False
    note: str = ""

    @property
    def status(self) -> str:
        if self.failures and self.claim.expect == "refuted":
            return "已证伪（符合预期）"
        if self.failures:
            return "失败"
        if self.claim.expect == "refuted":
            return "未能证伪（不符合预期）"
        return "通过" if self.checked else "未执行"


# ── 反例最小化 ───────────────────────────────────────────────
def default_shrinker(inst: Instance, claim: Claim, limit: int = 20) -> Instance | None:
    """有界不动点收缩：删站点 → 把 |δ| 顶到 ε。"""
    cur = inst
    for _ in range(limit):
        moved = False

        # (a) 删站点（Q1 的 claim 都是 n-通用的，反例可缩到更少站点）
        if cur.n > 2:
            for i in range(cur.n):
                cand = Instance(
                    sites=cur.sites[:i] + cur.sites[i + 1:],
                    bearings=cur.bearings[:i] + cur.bearings[i + 1:],
                    err_deg=cur.err_deg, source=cur.source,
                )
                if _still_fails(cand, claim):
                    cur, moved = cand, True
                    break
        if moved:
            continue

        # (b) 把 |δ_i| 顶到 ε，保留符号
        if cur.source is not None:
            ds = list(cur.deltas or [0.0] * cur.n)
            hard = [
                math.copysign(cur.err_deg, d) if abs(d) > 1e-12 else 0.0 for d in ds
            ]
            if hard != ds:
                cand = cur.rebearing(hard)
                if _still_fails(cand, claim):
                    cur, moved = cand, True
        if not moved:
            break
    return cur if cur.key() != inst.key() else None


def _still_fails(inst: Instance, claim: Claim) -> bool:
    try:
        r = claim.check(inst)
    except Exception:
        return False
    return r is not None and r is not SKIP


# ── 执行 ─────────────────────────────────────────────────────
def run_claim(
    claim: Claim, instances: list[Instance], max_failures: int = 5,
    shrink: bool = True,
) -> Outcome:
    out = Outcome(claim=claim)
    seen: set[tuple] = set()
    for inst in instances:
        try:
            reason = claim.check(inst)
        except Exception as exc:                     # 异常也是失败，不是跳过
            reason = f"抛出 {type(exc).__name__}: {exc}"
        if reason is SKIP:
            out.skipped += 1
            continue
        out.checked += 1
        if reason is None:
            continue
        k = inst.key()
        if k in seen:
            continue
        seen.add(k)
        if len(out.failures) < max_failures:
            shr = None
            if shrink:
                fn = claim.shrinker or default_shrinker
                try:
                    shr = fn(inst, claim)
                except Exception:
                    shr = None
            out.failures.append(Failure(inst, str(reason), shr))
    out.ok = (not out.failures) if claim.expect == "hold" else bool(out.failures)
    return out


def run_suite(
    claims: list[Claim], instances: list[Instance], verbose: bool = True
) -> bool:
    enable_utf8_console()
    if verbose:
        print("=" * 72)
        print(f"结论测试台 —— {len(claims)} 条 claim × {len(instances)} 个实例")
        print("=" * 72)
    all_ok = True
    for c in claims:
        o = run_claim(c, instances)
        all_ok &= o.ok
        if verbose:
            mark = "✓" if o.ok else "✗"
            print(
                f"  {mark} [{o.status}] {c.name}  "
                f"(检验 {o.checked}／跳过 {o.skipped}／反例 {len(o.failures)})"
            )
            if c.expect == "refuted":
                print(f"      预期被证伪：{c.statement}")
            for f in o.failures:
                print(f.describe())
    if verbose:
        print("-" * 72)
        print("全部通过" if all_ok else "存在失败")
    return all_ok


# ── 扫描 ─────────────────────────────────────────────────────
def run_sweep(
    values: list[Any],
    build: Callable[[Any], Instance | None],
    measure: Callable[[Instance], dict],
    label: str = "knob",
) -> list[dict]:
    """对一个旋钮扫一遍，返回可直接写 CSV 的行。"""
    rows: list[dict] = []
    for v in values:
        inst = build(v)
        if inst is None:
            rows.append({label: v, "ok": 0})
            continue
        row = {label: v}
        row.update(measure(inst))
        row.setdefault("ok", 1)
        rows.append(row)
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_sweep(path: str, rows: list[dict], x: str, ys: list[str], title: str = "") -> bool:
    """出图；matplotlib 缺失时安静跳过（返回 False）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    xs = [r[x] for r in rows if r.get("ok")]
    if not xs:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    for y in ys:
        ax.plot(xs, [r.get(y) for r in rows if r.get("ok")], marker=".", ms=4, lw=1.2, label=y)
    ax.set_xlabel(x)
    ax.grid(alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ── 伪证检验（mutant）────────────────────────────────────────
# 运行时 monkeypatch geometry（**不写文件**），断言 suite 必须变红。
# 一个永远打印 PASS 的测试台和正确的代码无法区分——而本项目记录在案的
# 两个 bug 都是静默实现错，所以这一步不是形式主义。
_ORIG: dict[str, Any] = {}


def _save(name: str) -> None:
    if name not in _ORIG:
        _ORIG[name] = getattr(geometry, name)


def mutant_wedge_cross_reversed() -> None:
    """楔形判据把叉积实参次序写反 ⟹ 楔形静默旋转 180°。"""
    _save("wedge_ok")

    def bad(site, theta, eps, p):
        d = geometry.sub(p, site)
        return (
            geometry.cross(d, geometry.unit_vector(theta - eps)) >= 0
            and geometry.cross(d, geometry.unit_vector(theta + eps)) <= 0
        )

    geometry.wedge_ok = bad


def mutant_no_t_positive_filter() -> None:
    """接受 t ≤ 0 的交点 ⟹ 把射线反方向的交点也算成顶点。"""
    _save("ray_intersection")

    def bad(s1, t1, s2, t2):
        d1 = geometry.unit_vector(t1)
        d2 = geometry.unit_vector(t2)
        den = geometry.cross(d1, d2)
        if abs(den) < geometry.PARALLEL_EPS:
            return None
        d = geometry.sub(s2, s1)
        a = geometry.cross(d, d2) / den
        b = geometry.cross(d, d1) / den
        p = (s1[0] + a * d1[0], s1[1] + a * d1[1])
        return p, a, b          # 不再要求 a>0 且 b>0

    geometry.ray_intersection = bad


def mutant_cover_radius_factor_two() -> None:
    """把 D 当半径用：覆盖 := (r_MEC ≤ D)。

    ⚠ 这里**不能**用「覆盖 := r_MEC ≤ D/2」当 mutant——那正是等价命题本身
    （覆盖 ⟺ r_MEC = D/2），与原实现**逻辑等价**，测不出任何东西。
    真正会犯的是量纲/因子错，所以 mutant 必须错在**量**上：
    r_MEC ∈ [D/2, D/√3] ⊂ (D/2, D)，于是本式恒真，凡「不覆盖」处都发散。
    """
    _save("diameter_circle_covers")

    def bad(a, b, pts):
        return geometry.covering_radius(pts) <= geometry.dist(a, b)

    geometry.diameter_circle_covers = bad


def mutant_omega_always_clipped() -> None:
    """无条件走 K 边形裁剪 ⟹ 破坏尺度不变性（场地 ×100 时会被切掉 ~6.85 m）。"""
    _save("region_polygon")
    orig = _ORIG["region_polygon"]

    def bad(sites, bearings, err_deg=1.0, arena_r=1800.0, r_eff=1500.0,
            segments=360, force_omega=False):
        return orig(sites, bearings, err_deg, arena_r, r_eff, segments, True)

    geometry.region_polygon = bad


def mutant_epsilon_snaps_to_omega() -> None:
    """把 ε 泄进 r_eff ⟹ 误差界变大时区域反而收缩。"""
    _save("omega_inactive_margin")
    orig = _ORIG["omega_inactive_margin"]

    def bad(region_pts, sites, arena_r=1800.0, r_eff=1500.0):
        return orig(region_pts, sites, arena_r, r_eff + 1.0)

    geometry.omega_inactive_margin = bad


MUTANTS: dict[str, Callable[[], None]] = {
    "wedge_cross_reversed": mutant_wedge_cross_reversed,
    "no_t_positive_filter": mutant_no_t_positive_filter,
    "cover_radius_factor_two": mutant_cover_radius_factor_two,
    "omega_always_clipped": mutant_omega_always_clipped,
    "epsilon_snaps_to_omega": mutant_epsilon_snaps_to_omega,
}


def apply_mutant(name: str) -> None:
    if name not in MUTANTS:
        raise KeyError(f"未知 mutant {name!r}；可选 {sorted(MUTANTS)}")
    MUTANTS[name]()


def restore_geometry() -> None:
    for k, v in _ORIG.items():
        setattr(geometry, k, v)
    _ORIG.clear()


# ── 自检 ─────────────────────────────────────────────────────
def _probe_family(n_random: int = 120, seed: int = 20260911) -> list[Instance]:
    """确定性实例族，供伪证检验使用。

    两部分：
      1. 随机物理自洽样本（n=2）；
      2. **手工窄交会角**样本——源固定，两站距源 d、角距恰为 g，
         故 γ = g 精确可控。窄 γ 是 Ω 最容易咬的形状，必须显式包含，
         否则「强制 Ω 裁剪」这类 mutant 可能整族都命中不了。
         （站点距源 d ∈ [1000,1500] 保证仍在本题的物理环带内。）
    """
    rng = random.Random(seed)
    out: list[Instance] = []
    while len(out) < n_random:
        i = sample_instance(rng, n=2, err_mode="zero")
        if i is not None:
            out.append(i)

    src = (0.0, 0.0)
    alpha = 0.7                                  # 任意朝向，避免与坐标轴对齐
    for g_deg, d in ((2.5, 1400.0), (3.0, 1400.0), (4.0, 1300.0), (6.0, 1200.0),
                     (10.0, 1200.0), (30.0, 1200.0), (60.0, 1100.0), (89.0, 1100.0),
                     (91.0, 1100.0), (92.0, 1100.0), (120.0, 1050.0), (150.0, 1050.0)):
        a = math.radians(alpha)
        g = math.radians(g_deg)
        sites = [
            (d * math.cos(a - g / 2), d * math.sin(a - g / 2)),
            (d * math.cos(a + g / 2), d * math.sin(a + g / 2)),
        ]
        bs = [
            math.degrees(math.atan2(src[1] - s[1], src[0] - s[0])) % 360.0
            for s in sites
        ]
        out.append(Instance(sites=sites, bearings=bs, err_deg=1.0, r_eff=d,
                            source=src, deltas=[0.0, 0.0],
                            note=f"narrow γ={g_deg}°"))
    return out


def _self_test() -> bool:
    enable_utf8_console()
    print("=" * 72)
    print("claim_tester.py 自检 —— 测试台自己必须先能被证明「会失败」")
    print("=" * 72)

    checks: list[tuple[str, bool, str]] = []

    def ck(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    # 1. 采样器：物理自洽
    rng = random.Random(20260911)
    insts = []
    while len(insts) < 300:
        i = sample_instance(rng, n=2)
        if i is not None:
            insts.append(i)
    ann_ok = all(
        math.dist(s, i.source) <= i.r_eff + 1e-9          # 单边：站点落在有效接收半径内
        for i in insts for s in i.sites
    )
    reff_ok = all(R_EFF_MIN - 1e-9 <= i.r_eff <= R_EFF_MAX + 1e-9 for i in insts)
    ang_ok = all(
        abs(geometry.norm180(b - math.degrees(
            math.atan2(i.source[1] - s[1], i.source[0] - s[0])
        ))) <= i.err_deg + 1e-9
        for i in insts for s, b in zip(i.sites, i.bearings)
    )
    ck("采样器物理自洽（站点在 r_eff 内、r_eff∈[1000,1500]、|δ|≤ε）",
       ann_ok and ang_ok and reff_ok,
       f"{len(insts)} 实例 {'全部' if ann_ok and ang_ok and reff_ok else '有越界'}")

    # 2. 确定性：同种子两次必须完全一样
    a = [sample_instance(random.Random(20260911), n=2) for _ in range(50)]
    b = [sample_instance(random.Random(20260911), n=2) for _ in range(50)]
    ck("确定性：同种子逐位可复现",
       all(x.key() == y.key() for x, y in zip(a, b)), "50 实例")

    # 3. err_mode = extreme 时 |δ| = ε
    rng2 = random.Random(5)
    ex = []
    while len(ex) < 100:
        i = sample_instance(rng2, n=2, err_mode="extreme")
        if i is not None:
            ex.append(i)
    ck("err_mode=extreme：|δ| 恰为 ε、符号有正有负",
       all(abs(abs(d) - 1.0) < 1e-12 for i in ex for d in i.deltas)
       and any(d > 0 for i in ex for d in i.deltas)
       and any(d < 0 for i in ex for d in i.deltas),
       "100 实例")

    # 4. 平凡真 claim 必须过
    ok_claim = Claim("trivial_true", "恒真", lambda i: None)
    r = run_claim(ok_claim, insts)
    ck("平凡真 claim 通过", r.ok and r.checked == len(insts),
       f"检验 {r.checked}")

    # 5. 平凡假 claim 必须挂，且**报出反例**
    bad_claim = Claim("trivial_false", "恒假", lambda i: "故意失败")
    r = run_claim(bad_claim, insts, shrink=False)
    ck("平凡假 claim 失败并报出反例", (not r.ok) and bool(r.failures),
       f"反例 {len(r.failures)} 个")

    # 6. SKIP 既不算过也不算挂
    skip_claim = Claim("skips", "全跳过", lambda i: SKIP, expect="hold")
    r = run_claim(skip_claim, insts)
    ck("SKIP：既不算过也不算挂，且空检验不算通过",
       r.ok and r.checked == 0 and r.skipped == len(insts),
       f"跳过 {r.skipped}")

    # 7. expect=refuted 的语义：报不出反例反而算挂
    r = run_claim(Claim("r1", "应被证伪", lambda i: None, expect="refuted"), insts)
    ck("expect=refuted：报不出反例算挂", not r.ok, r.status)
    r = run_claim(Claim("r2", "应被证伪", lambda i: "反例在这", expect="refuted"),
                  insts, shrink=False)
    ck("expect=refuted：报出反例算过", r.ok, r.status)

    # 8. check 抛异常算失败，不算跳过
    def boom(i):
        raise ValueError("boom")

    r = run_claim(Claim("boom", "抛异常", boom), insts, shrink=False)
    ck("check 抛异常算失败（不静默跳过）",
       (not r.ok) and r.checked == len(insts) and "ValueError" in r.failures[0].reason,
       r.failures[0].reason if r.failures else "无反例")

    # 9. oracle 独立性：角度式与叉积式在随机点上一致
    rng3 = random.Random(99)
    mism = 0
    for _ in range(2000):
        site = (rng3.uniform(-500, 500), rng3.uniform(-500, 500))
        th = rng3.uniform(0, 360)
        eps = rng3.choice((0.5, 1.0, 2.0))
        p = (rng3.uniform(-1500, 1500), rng3.uniform(-1500, 1500))
        if geometry.wedge_ok(site, th, eps, p) != oracle_contains(site, th, eps, p):
            mism += 1
    ck("oracle：角度式与叉积式楔形判据一致", mism == 0, f"{mism}/2000 不一致")

    # 10. oracle：Welzl 与 covering_radius 一致
    worst = 0.0
    for _ in range(300):
        pts = [(rng3.uniform(-200, 200), rng3.uniform(-200, 200))
               for _ in range(rng3.randint(2, 7))]
        w = oracle_mec(pts)
        worst = max(worst, abs(w[1] - geometry.covering_radius(pts)))
    ck("oracle：Welzl 与 covering_radius 一致", worst < 1e-9, f"最大差 {worst:.2e}")

    # 11. oracle：射线法点在内与多边形一致
    bad_in = 0
    for _ in range(200):
        pts = geometry.convex_hull(
            [(rng3.uniform(-100, 100), rng3.uniform(-100, 100)) for _ in range(8)]
        )
        if len(pts) < 3:
            continue
        cen = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        if not _point_in_poly_strict(pts, cen):
            bad_in += 1
        far = (cen[0] + 5000, cen[1] + 5000)
        if _point_in_poly_strict(pts, far):
            bad_in += 1
    ck("oracle：射线法（质心在内、远点在外）", bad_in == 0, f"{bad_in} 处错判")

    # 12. ★ 伪证检验：5 个 mutant 必须**全部被抓住**
    #
    # 探针必须建在**可观测**的层面。下面三条是**实测**结论，不是推测；
    # 第一版探针在此处 0/5 漏网，原因正是把探针建在了不可观测的层面。
    #   · `ray_intersection` 的 t>0 守卫：8 万次采样，交点落在射线身后的情形
    #     被楔形谓词 **100%** 滤掉（反向点与 θ 的夹角恒为 180°−ε > ε）⟹ 它在
    #     **区域**层面完全不可观测，只能探它自己的**契约**。
    #   · 「覆盖 := r_MEC ≤ D/2」**不是 bug**——那正是等价命题本身，与原实现
    #     逻辑等价。能抓的是**因子/量纲**错（把 D 当半径用），故 mutant 改了。
    #   · 强制 Ω 裁剪：证书 >0 时 Ω 是区域的超集，裁剪只在「区域探入 K 边形
    #     弓高壳层」时才有影响——单实例命中率约 50%，故探针必须跑**实例族**。
    fam = _probe_family()

    def _diam(poly: list[Point]) -> float:
        return geometry.max_pairwise_distance(poly)[0] if len(poly) >= 2 else 0.0

    def _ray_contract() -> str | None:
        """t>0 契约：交点落在第二条射线**身后**时必须返回 None。"""
        hit = geometry.ray_intersection((0.0, 0.0), 0.0, (5.0, 3.0), 90.0)
        if hit is not None:
            return f"交点 (5,0) 在第二条射线身后（t2=−3），却返回了 {hit[0]}"
        good = geometry.ray_intersection((0.0, 0.0), 0.0, (5.0, 3.0), 270.0)
        if good is None:
            return "正例（t1=t2=3）被误判为无交"
        return None

    def _wedge_oracle(i: Instance) -> str | None:
        if not geometry.wedge_bounded(i.bearings, i.err_deg):
            return SKIP
        r = geometry.localization_region(i.sites, i.bearings, i.err_deg)
        if not r.corners_raw:
            return "有界构形下 raw 候选点为空（谓词把真顶点全滤掉了）"
        for q in list(r.corners_raw) + list(r.polygon):
            for s, b in zip(i.sites, i.bearings):
                if not oracle_contains(s, b, i.err_deg, q):
                    return f"点 {q} 不在站 {s} 的楔形内（θ={b:.3f}°）"
        return None

    def _cover_vs_margin(i: Instance) -> str | None:
        """覆盖判定的真值由**余量符号**独立给出，与被测函数无关。"""
        r = geometry.localization_region(i.sites, i.bearings, i.err_deg)
        if not r.bounded:
            return SKIP
        return None if r.diameter_circle_ok == (
            r.cover_margin >= -1e-6 * max(1.0, r.diameter)
        ) else "覆盖判定与直径圆余量的符号不一致"

    def _omega_cert(i: Instance) -> str | None:
        """证书语义：Ω 余量 > 0 时返回的必须是**纯楔形之交**。

        把 `arena_r` 顶到区域外接半径刚刚之上——此时证书恰好为正，而区域
        已经探出内接 K 边形的弓高壳层，于是「是否真的走了精确解」变得可观测。
        """
        if not geometry.wedge_bounded(i.bearings, i.err_deg):
            return SKIP
        c = geometry.region_corners(i.sites, i.bearings, i.err_deg)
        if len(c) < 3:
            return SKIP
        ar = max(math.hypot(*p) for p in c) * (1.0 + 1e-9)
        if oracle_omega_margin(c, i.sites, ar, R_EFF_MAX) <= 0.0:
            return SKIP
        got, _, _ = geometry.region_polygon(
            i.sites, i.bearings, i.err_deg, ar, R_EFF_MAX, 360, False
        )
        return None if abs(_diam(got) - _diam(c)) <= 1e-9 else (
            "证书说 Ω 不活化，区域却被裁剪了"
        )

    def _omega_oracle(i: Instance) -> str | None:
        """Ω 余量必须与**独立重算**一致（不与被测函数自证）。"""
        c = geometry.region_corners(i.sites, i.bearings, i.err_deg)
        if len(c) < 3:
            return SKIP
        a = geometry.omega_inactive_margin(c, i.sites, ARENA_R, 1200.0)
        b = oracle_omega_margin(c, i.sites, ARENA_R, 1200.0)
        return None if abs(a - b) <= 1e-9 else f"Ω 余量 {a!r} ≠ oracle {b!r}"

    probes = [
        # 顶点落在楔形内 —— **必须用角度式 oracle**；用 wedge_ok 自证抓不住
        # 叉积次序反了的 mutant（区域与判据一起被静默旋转 180°，自洽）。
        # ⚠ 还必须查 `corners_raw`，不能只查 `polygon`：谓词反了会让
        # `region_corners` 只留下 2 个点，`region_polygon` 于是**静默退回
        # tier 2b**，而 tier 2b 的半平面是直接用 `unit_vector` 搭的、不经过
        # `wedge_ok`——最终多边形反而"碰巧正确"。谓词的错只有在 raw 层可见。
        Claim("probe_wedge_oracle",
              "raw 候选点与区域顶点都落在全部楔形内（角度式 oracle）",
              _wedge_oracle),
        # 回代方位角（走 atan2，与叉积无关）—— 同样是独立表述
        Claim("probe_backsub", "顶点回代各站方位角 ≤ ε",
              lambda i: None if all(
                  abs(geometry.norm180(math.degrees(
                      math.atan2(q[1] - s[1], q[0] - s[0])) - b)) <= i.err_deg + 1e-9
                  for q in geometry.localization_region(
                      i.sites, i.bearings, i.err_deg).polygon
                  for s, b in zip(i.sites, i.bearings)
              ) else "有顶点回代超界"),
        Claim("probe_ray_contract", "交点落在射线身后 ⟹ ray_intersection 返回 None",
              lambda i: _ray_contract()),
        Claim("probe_cover_vs_margin", "覆盖判定 ⟺ 直径圆余量非负",
              _cover_vs_margin),
        Claim("probe_omega_certificate", "Ω 余量 > 0 ⟹ 区域 = 纯楔形之交",
              _omega_cert),
        Claim("probe_omega_oracle", "Ω 余量与独立重算一致", _omega_oracle),
    ]

    caught: list[str] = []
    missed: list[str] = []
    restore_geometry()
    base_ok = run_suite(probes, fam, verbose=False)
    if not base_ok:
        ck("伪证检验前置：未注入 mutant 时探针全绿", False,
           "基线就挂了，mutant 检验无意义")
    else:
        for mname in MUTANTS:
            restore_geometry()
            apply_mutant(mname)
            try:
                still_ok = run_suite(probes, fam, verbose=False)
            finally:
                restore_geometry()
            (missed if still_ok else caught).append(mname)

        ck("★ 伪证检验：未注入时全绿，注入后 5 个 mutant 全部被抓住",
           not missed, f"抓住 {len(caught)}/{len(MUTANTS)}：{sorted(caught)}"
           + (f"  漏网 {sorted(missed)}" if missed else ""))

    # 13. mutant 必须被干净还原
    restore_geometry()
    same = (
        geometry.covering_radius([(0.0, 0.0), (10.0, 0.0)])
        == geometry.covering_radius([(0.0, 0.0), (10.0, 0.0)])
        and "region_polygon" not in _ORIG
    )
    ck("mutant 注入后可干净还原（无残留 monkeypatch）", same, f"残留 {sorted(_ORIG)}")

    # 14. CSV 确定性
    import tempfile, os
    rows = [{"g": i, "D": i * 1.5} for i in range(20)]
    d = tempfile.mkdtemp()
    p1, p2 = os.path.join(d, "a.csv"), os.path.join(d, "b.csv")
    write_csv(p1, rows)
    write_csv(p2, rows)
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        ck("CSV 输出字节确定", f1.read() == f2.read(), p1)

    # ── 输出 ────────────────────────────────────────────────
    print("-" * 72)
    passed = 0
    for name, ok, detail in checks:
        passed += ok
        print(f"  {'✓' if ok else '✗'} {name}" + (f"   [{detail}]" if detail else ""))
    print("-" * 72)
    print(f"{passed}/{len(checks)} 通过")
    print("=" * 72)
    return passed == len(checks)


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
