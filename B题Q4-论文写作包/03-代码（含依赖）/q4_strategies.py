r"""
问题4 —— **定向源下的检测与清除策略，以及 Q3 策略的失效模式**

    python q4_strategies.py --seeds 8 --n 13 --dir 0.5

## Q4 与 Q3 的唯一本质差别

问题4：目标区域中**既有全向干扰源，又有定向干扰源**，总数 10–16，
**定向源的个数与定向方向均未知**。其余条件与问题3 相同。

定向源在定向方向两侧各 $90°$ 内辐射，**之外没有信号**。这一条同时击穿两件事：

| | Q3 的依据 | 定向源下 |
| --- | --- | --- |
| **探测** | 点在 $r_{\text{eff}}$ 内 ⟹ 必测到信号 | ✗ 还要在扇区内 |
| **证空** | 覆盖全场的点全返回 `no_signal` ⟹ 必空 | ✗ **扇区外的点也返回 `no_signal`** |

所以 `q4_criteria.covers_arena_dir`（距离覆盖 **∧** 角度覆盖）既是新的证空判据，
**也自动给出了探测保证** —— 同一个几何条件同时管两件事。

## 本脚本的两个臂

| 臂 | 说明 |
| --- | --- |
| `q3` | 直接搬 Q3 的 `run_shared`（7 哨兵 + 共享测量）**不换判据** —— 预期会**误判真源为空** |
| `q4` | 新布局（方格 + **场地边界环**）+ 新判据 `covers_arena_dir` |

`q3` 臂的意义不是「再优化一版」，而是**把失效模式量化出来**：
它到底漏掉几个源、漏在什么位置。这是论文里「为什么必须换判据」的实证。

## 布局的两条硬性要求（数值实测得来）

1. **边界环半径必须恰好等于 $R=1800$。** 取 1795 时，源落在 1797 处的最近环点
   在**源身后**（方位 $180°$），绕过 $0°$ 的间隔变成 $182°>180°$ ⟹ 判据失效。
   **差 5 m 就从达标变成不达标。**
2. **必须同时有内层点。** 只有边界环时，半径 275 m 处的源在 $1000$ m 内
   **一个点都没有**（环上最近点距它 $1525$ m）。粗方格提供距离覆盖。
"""

from __future__ import annotations

import argparse
import math
import random
import statistics as st
import sys

sys.path.insert(0, r"D:\Contest\数学建模大赛\CUMCM\Coding\B题")
sys.path.insert(0, r"D:\Contest\数学建模大赛\CUMCM\q3\02-代码")
import offline_sim as OS  # noqa: E402
from claim_tester import enable_utf8_console  # noqa: E402

import q4_criteria as C  # noqa: E402
import q3_strategies as Q  # noqa: E402
from q3_harness import LocalSim, Robot  # noqa: E402

ARENA = C.ARENA


def grid_route(step: float = 300.0):
    """方格的蛇形（boustrophedon）巡回 —— 比最近邻紧凑得多。"""
    rows: dict[float, list] = {}
    for x, y in C.square_grid(step):
        rows.setdefault(round(y, 6), []).append(x)
    out, left = [], True
    for y in sorted(rows):
        xs = sorted(rows[y])
        if not left:
            xs = xs[::-1]
        out += [(x, y) for x in xs]
        left = not left
    return out



# ⚠ 试过把清源排序从「纯最近邻」换成「滚动 TSP（NN + 2-opt）」：
#   **完全没变化**（895.2 vs 892.4 s/个，请求数 394 对 394，6/6 全清）。
#   Q3 里碎片化路由是真实损失，Q4 里不是 —— 因为 Q4 的清源本来就被
#   33 个扫描点切成很多轮、每轮只有 1–2 个源可清，TSP 无从发挥。
#   （这是我第 5 次「归因指出问题 → 直接改 → 无效或更差」。）


def tsp_route(pts, start=(0.0, 0.0), passes: int = 60):
    """对**同一个点集**求一条更短的巡回：最近邻起步 → 2-opt 收敛。

    ## 为什么这条有用，而「清源顺序换 TSP」没用

    两件事完全不同：
    - **清源顺序**：Q4 的清源被 33 个扫描点切成很多轮、每轮只有 1–2 个源可清，
      TSP 无从发挥 —— 实测 895.2 vs 892.4，**确实没变化**（那个结论是对的）。
    - **扫描路线**：33 个点是**一次性**走完的，顺序是**真自由变量**。
      同一批点，蛇形（`grid_route` + `ring_route`）= 30079 m，
      NN+2-opt = 21132 m，**差 8947 m = 1789 s**。

    实测（n=14，定向 0.6，6 局配对）：**892.4 → 800.4 s/个（−10.3%）**，两臂都全清。
    ⚠ seed 3 会变差 3.6 s —— 这是统计性收益，不是单调的。
    """
    left, cur, out = list(pts), start, []
    while left:
        k = min(range(len(left)), key=lambda i: math.dist(cur, left[i]))
        cur = left.pop(k)
        out.append(cur)

    def length(o):
        d, c0 = 0.0, start
        for p in o:
            d += math.dist(c0, p)
            c0 = p
        return d

    for _ in range(passes):
        improved = False
        for i in range(len(out) - 1):
            for j in range(i + 1, len(out)):
                cand = out[:i] + out[i:j + 1][::-1] + out[j + 1:]
                if length(cand) < length(out) - 1e-9:
                    out, improved = cand, True
        if not improved:
            break
    return out


def interior_route():
    """**极简内层结构**：中心 + 环500×4 + 环1000×8（13 点）。

    来自 `优化-问题4的时间下界与400秒目标的可行性.md` §2 的穷举：
    这是满足「距离覆盖 ∧ 角度覆盖」的**最少内层点数**，
    取代原先的 800 m 方格蛇形（21 点、20565 m 赶路）。

    ⚠ 内层的职责只有两件：
      1. 距离覆盖 —— 边界环只覆盖 $r\ge800$ 的外圈，内盘需要内层点；
      2. 内层源的角度覆盖 —— 半径 275 m 处的源在 $1000$ m 内**只能看到中心一个点**
         （环上最近点距它 $1525$ m），必须有点在它周围。
    """
    pts = [(0.0, 0.0)] + C.ring(4, 500.0) + C.ring(8, 1000.0)
    # 最近邻成路（13 点，贪心足够）
    left, cur, out = list(pts), (0.0, 0.0), []
    while left:
        k = min(range(len(left)), key=lambda i: math.dist(cur, left[i]))
        cur = left.pop(k)
        out.append(cur)
    return out


def ring_route(k: int, rho: float, start=(0.0, 0.0)):
    """边界环：从离当前位置最近的一点起，沿圆周走一圈。"""
    pts = C.ring(k, rho)
    j0 = min(range(k), key=lambda j: math.dist(start, pts[j]))
    return pts[j0:] + pts[:j0]


def capture_fine(robot, kb, ch, budget: int = 20, step: float = 30.0,
                 reach: float = 1500.0, remeasure: int = 3) -> bool:
    """**定步长沿射线推进**：每步直接 `/clear` —— 不二分、不过冲。

    ## 对治的是什么（2026-09-13）

    生产用的 `q3_strategies.capture` 是**几何步长 + 二分**：

        cur_t = 200.0;  while ...: cur_t *= 1.6        # 步长按 1.6 涨
        if alpha > 90: crossed = True                  # 冲过源
        ...7 次二分在最后一段里走回来

    源在距离 D 处时，探针位置是 200 → 520 → 1032 → …, **必然冲过源**
    （最远到 ~1.6D），再二分回走 ≈ 0.6D ⟹ **总路程 ≈ 2.2 D**，
    而必要路程只有 D。官方 10 局实测：离点测量段的**多余走位 385 m/段**，
    15 段/局，合计 5777 m/局。

    ## 为什么定步长必然命中

    源**就在射线上**（示向度准，`BEARING_ERR_DEG = 1°`），所以

      · 沿轴：最近探针距源 ≤ step/2 = 15 m
      · 横向：偏差 = 已走距离 × tan(1°)。每 `remeasure` 步重测并把射线
        **刷新到更近的点**，故横向偏差只按「两次重测之间走的距离」算：
        90 m × 0.01745 = 1.6 m
      · 最坏距离 = sqrt(15² + 1.6²) = **15.1 m < 20 m（`CLEAR_RADIUS`）**

    ⟹ **必然命中，且不会冲过源**（在冲过去之前就已经清掉了），故**无振荡风险**
    —— 生产版那个二分正是为了治振荡才加的（见其 docstring）。

    ## ⚠ 不依赖 `near`

    `NEAR_RADIUS = 5 m`，且要求「在扇区内」，对定向源太严，指望不上。
    本原语每步直接发 `/clear`（3 s，与方向无关）。

    ## 代价

    步数变多（D/step 步），每步 `移动 + /clear`。但每步**不比生产版贵**：
    生产版每步是 `移动 + /measure`（5 s），本原语每步是 `移动 + /clear`（3 s），
    只是每 `remeasure` 步才补一次测量。
    """
    if not kb.bearings[ch]:
        return False
    px, py, th = kb.bearings[ch][-1]
    u = Q.unit(th)
    t = 0.0
    k = 0
    for _ in range(int(reach / step) + 12):
        t += step
        q = Q.clamp_to_arena(px + t * u[0], py + t * u[1])
        if math.dist(q, (px, py)) < 1.0:          # 已顶到场地边界
            break
        robot.move_to(*q)
        if Q._try_clear(robot, kb, ch):
            return True
        k += 1
        if k % remeasure == 0:
            res, svd = robot.measure(ch)
            kb.record(ch, q[0], q[1], res, svd)
            if res == "direction":                # 刷新射线到更近的点
                px, py, th = q[0], q[1], svd
                u = Q.unit(th)
                t = 0.0
    return False


# ══════════════════════════════════════════════════════════════════════
# ★★★ 2026-09-13【圆心 no_signal ⟹ 本地铺网】`clear_mode="localmesh"`
#
# ## 实测签名（12 局离线，n=14 定向 1.0，共 168 个源）
#
#   89% 首次 /clear 即成功；11% 失败。失败者里：
#     · `go_clear_mec` 循环内出现过 **`no_signal`** 的 → **1 个，F = 7413 s**
#     · 没出现过的 → 18 个，中位 F = **88 s**
#     · F ≤ 1000 s 的 16 个里，出现 `no_signal` 的 **0 个**
#
# ⟹ **`no_signal` 是灾难的签名**，而 MEC 半径 R 不是
#   （对 F 做最小二乘 F = −186 + 7.47R，**R² = 0.094**，毫无解释力）。
#
# ## 机理
#
# 机器人走到 MEC 圆心（离真源仅约 30 m），却因**在扇区外**收到 `no_signal`；
# 现行代码 `break` ⟹ 掉进 `capture`（信号驱动、每次 ~1941 m）⟹ `corridor_sweep`
# （60 m 步，实测 97% 失败率）⟹ 尾段网。
# **人已站在 30 m 处，却因为"听不见"而放弃。**
#
# ## 改法
#
# `no_signal` 时不 `break`，改用**已知的 MEC 半径 R** 在圆心附近铺一张
# 间距 20 m 的网 —— 20 m 间距保证任意点有探点落在 14.1 m ≤ 20 m 清除半径内，
# 而真源**必在** $\mathrm{disc}(\hat S, R)$ 里 ⟹ **必然命中**。
#
# 代价 $T(R)$（蛇形，20 m 间距，实测速度/清除时长）：R=40→158 s，R=80→474 s，
# R=120→1151 s。对比实测灾难 7413 s。
#
# ⚠ 两点保留：
#   1. `no_signal` 那个类**只有 1 个样本**（机理清楚，统计很薄）。
#   2. `R_local_max` 之外仍走原链（网太大不划算，$T(250)\approx4600$ s 已接近灾难量级）。
#
# ## 配对实验结果（`q4_localmesh_paired.py`，全部零漏清，共 80 配对局）
#
# 数值为**加上蛇形修复之后**的最终结果：
#
# | 配置 | 局数 | 配对差 | 配对 t | 移动差 | 更快局数 |
# | --- | --- | --- | --- | --- | --- |
# | n=10 dir=1.0 | 16 | **−477 s/局 = −47.7 s/个** | −3.21 | −8.3% | 10/16 |
# | n=12 dir=0.6 | 16 | **−342 s/局 = −28.5 s/个** | −4.37 | −6.7% | 13/16 |
# | n=14 dir=1.0 | 32 | **−431 s/局 = −30.8 s/个** | −6.35 | −7.4% | 28/32 |
# | n=16 dir=1.0 | 16 | **−349 s/局 = −21.8 s/个** | −3.37 | −5.3% | 11/16 |
#
# **样本外检验**（n=14：`R_LOCAL_MAX` 在 seed 1–16 上选，17–32 为样本外）：
#   样本内 −308 s/局（t=−3.34）｜ **样本外 −274 s/局（t=−3.58，14/16 更快）**
#   ⟹ 样本外是样本内的 89%，**没有过拟合**。
#   （该检验跑在蛇形修复之前；蛇形是全局改动、两臂同享，不影响配对的内部效度。）
#
# ⚠ 全部是本机离线结果，**官方零验证**；采纳前必须官方复跑。
# ══════════════════════════════════════════════════════════════════════
#
# ## 上限 `R_LOCAL_MAX` 的定法（**不要随便调大**）
#
# 判据是期望代价：**铺网 iff $T(R) < F$**，$F$ 为兜底代价。
# 实测 $F$ 中位 88 s、**均值 625 s**（重尾，最大 7413 s）；按均值定价 ⟹
# $T(R)=625$ 解得 $R\approx88$ m。
#
# 16 局参数扫描（n=14 定向 1.0，同批种子，**有选择偏差**）：
#
# | R_LOCAL_MAX | 配对差 | t | 移动差 |
# | --- | --- | --- | --- |
# | 40 | +6 | +0.41 | −14 |
# | **60** | **−225** | **−3.81** | −1445 |
# | **80** | **−308** | **−3.34** | −1903 |
# | 100 | −250 | −2.20 | −1732 |
# | 150 | −164 | −1.76 | −1481 |
# | 250 | −24 | −0.10 | −1125 |
#
# ⟹ 单峰，最优 60–80，**与理论预测的 88 一致**。
# ⚠ 早期版本取 250 时效应几乎为零（−24 s）——**上限定大等于没改**。
R_LOCAL_MAX = 80.0

# ★★ Q4 专用：`q3_mec_clear.go_clear_mec` 的 `early_rmax`（见 `run_q4` 里的推导）。
#    Q3 的共用默认值是 40；Q4 实测 p(R_c) 撑到 ~100 才落到盈亏平衡 1/3 以下。
Q4_EARLY_RMAX = 100.0


def go_clear_mec_v2(robot, kb, ch, max_probe: int = 5,
                    early_rmax: float | None = None,
                    r_local_max: float = R_LOCAL_MAX,
                    on_nosignal: str = "ray",
                    defer_fail: int = 0) -> bool:
    """`q3_mec_clear.go_clear_mec` 的修补版：圆心处 `no_signal` 改走本地网。

    除 `no_signal` 分支外，**逐行照抄**生产版（含共线检测、`clear_ok`/`early`
    两级清源判据、`near` 分支）。仅把
        `if res == "no_signal": break`
    换成
        `if res == "no_signal": 在圆心铺半径 R+20 的网（20 m 间距）`
    """
    import q3_mec_clear as MC

    if early_rmax is None:
        early_rmax = MC._EARLY_CLEAR_RMAX
    if not getattr(kb, "_patho", None):
        kb._patho = set()
    if ch not in kb._patho and len(kb.bearings[ch]) >= 2:
        if MC.fix_collinear(robot, kb, ch):
            kb._patho.add(ch)

    for _ in range(max_probe):
        reg = MC.mec_of(kb, ch)
        if reg is None:
            break
        cx, cy = reg.mec.center
        x, y = Q.clamp_to_arena(cx, cy)
        robot.move_to(x, y)
        if reg.clear_ok:                                 # R_c <= 20 ⟹ 必然清除
            if robot.clear(ch) == "success":
                kb.cleared.add(ch)
                kb.refresh_empty()
                return True
        elif 0.0 < reg.cover_r <= early_rmax:            # 圆心略偏也先试一次
            if robot.clear(ch) == "success":
                kb.cleared.add(ch)
                kb.refresh_empty()
                return True
        res, svd = robot.measure(ch)                     # 中心补测
        kb.record(ch, x, y, res, svd)
        if res == "near" and robot.clear(ch) == "success":
            kb.cleared.add(ch)
            kb.refresh_empty()
            return True
        if res == "no_signal":
            # ★★★ 2026-09-13【`on_nosignal="prod"` = 逐字退回生产版】
            #
            # 存在的唯一目的是让 `defer_fail` 实验**只差一个变量**。
            # 实测教训：`defer_fail>0` 那版最初直接把 `Q.go_clear` 换成
            # `go_clear_mec_v2`，而 v2 默认 `on_nosignal="ray"` ⟹ 顺带把
            # 已否决的 `raymesh` 也打开了 ⟹ 两臂差了两处，结果无法归因
            # （12 局 B 快 210 s/局，但说不清是谁的功劳）。
            if on_nosignal == "prod":
                break                                    # ← 与 `MC.go_clear_mec` 逐字相同
            # ★★★ 修复点（唯一改动）
            if on_nosignal == "ray" and kb.bearings[ch]:
                # ── 沿【最后一条示向度】的射线搜 ──────────────────────
                # 与 `mesh` 版的根本差别在**保证的来源**：
                #   · `mesh`：真源在 MEC 圆盘内 —— **依赖 `err_deg=1.0` 这个误差模型**
                #   · `ray` ：真源在「最后一条示向度所在射线」上（±1°、≤1500 m）
                #             —— **是测量直接给的，不依赖任何模型**
                # 已用 2186 条官方样本验证后者：测到示向度时的最大距离 1478 m ≤ 1500 ✓
                # 生产链在 no_signal 之后走的 `capture`/`corridor_sweep` 也是沿射线的，
                # 只是步长（60 m）保证不了命中（横向偏差 d·tan1° 在 1500 m 处达 26 m > 20 m）。
                # `ray_mesh_clear` 是同一思路的**保证版**（20 m 步长 + 锥形半宽）。
                _px, _py, _th = kb.bearings[ch][-1]
                if ray_mesh_clear(robot, kb, ch, _px, _py, _th):
                    return True
                break
            R = float(getattr(reg.mec, "radius", 0.0) or 0.0)
            if R <= r_local_max:
                rad = R + 20.0
                lim = 4 * (int(rad / 20.0) + 1) ** 2 + 8      # ⚠ 必须给足 limit
                if mesh_clear(robot, kb, ch, (cx, cy),
                              radius=rad, step=20.0, limit=lim):
                    return True
            break

    # ★★★ 2026-09-13【失败后延后，而不是立刻掉进昂贵兜底】`defer_fail = N`
    #
    # ## 依据：$R_c$ 随示向度条数快速衰减（6 局实测）
    #
    # | 条数 | 2（现行停测点）| 3 | 4 | 5 |
    # | --- | --- | --- | --- | --- |
    # | $R_c$ 中位 | **37.9 m** | 22.8 | **15.8** | 20.0 |
    # | $R_c\le20$（必然清除）占比 | **5%** | 35% | **64%** | **100%** |
    #
    # ⟹ 在 2 条示向度处清源，只有 5% 是"必然成功"的；**多攒 2 条就到 64%~100%**。
    # 而多攒的代价只是「在常规停点继续测它」（4.87 s/次）。
    #
    # 所以：**第一次失败不烧 `capture`（~1941 m）**，让这个频道留在
    # 「已定位未清除」状态、继续在常规停点被测量；$R_c$ 掉下来后再试。
    # 试满 `N` 次仍未成才允许掉进 `capture`/`corridor_sweep` —— **正确性不受影响**
    # （尾段 `clear_located(force=True)` 仍是最后兜底）。
    if defer_fail > 0:
        _cnt = kb.__dict__.setdefault("_defer_cnt", {})
        _cnt[ch] = _cnt.get(ch, 0) + 1
        if _cnt[ch] <= defer_fail:
            return False
    if Q.capture(robot, kb, ch):
        return True
    return Q.corridor_sweep(robot, kb, ch)


def run_q4(robot, kb, step: float = 300.0, k_ring: int = 180,
           max_pts: int = 4000, angular: bool = True,
           ring_eps: float = 200.0, early_stop: bool = True,
           drop_outer: bool = False, policy: str = "pick",
           capture_mode: str = "prod", clear_mode: str = "prod",
           mid_stops: int = 0, defer_fail: int = 0,
           meas_uncleared: bool = False, try_cap: int | None = None,
           meas_due: bool = False, scan_current_first: bool = True):
    """Q4 策略：**可靠扫描布局 + 定向判据**，边扫边清。

    ⚠ 清扫顺序是「先方格、后边界环」，但**判据只有在整条布局走完后才可能成立**
    （角度覆盖要求点集把每个源都围住）。所以扫描阶段基本不能提前判定，
    代价换来的是**判据的可靠性** —— 这正是 Q4 与 Q3 的取舍。
    """
    import q3_mec_clear as MC
    # ★★★ 2026-09-13【early_rmax 40 → 100】—— 本轮最大的单项收益
    #
    # `early_rmax` 是「MEC 圆心略偏时也先在圆心试一次 `/clear`」的门槛。
    # 现行 40 是 Q3 上取的保守值；**Q4 的剂量-响应曲线完全不同**：
    #
    # | $R_c$ m | ≤35 | 35–40 | 40–50 | 50–70 | 70–100 | 100–200 | >200 |
    # | --- | --- | --- | --- | --- | --- | --- | --- |
    # | **Q4 实测 $p$** | 1.00 | 0.83 | 0.91 | 0.75 | **0.60** | 0.17 | 0.20 |
    # | Q3（旧注释）| 1.00 | 0.48 | — | 0.50 | 0.23 | 0.00 | 0.00 |
    #
    # 盈亏平衡 $p^*=1/3$（试一次成功能省 6 s，白试亏 3 s：$6p=3(1-p)$）
    # ⟹ **Q4 上阈值应取到 ~100**，而不是 40。40–100 那一段占了 **39% 的源**，
    # 原来一次都没试过清就掉进了昂贵的兜底链。
    #
    # 实测（12 配对局，n=14 定向 1.0，policy=roll，零漏清）：
    #     8713 → 8070 s/局 = **−46 s/个**，$t=-7.87$；`capture` 调用 44 → 25 次/12局
    #
    # ⚠ 不直接改 `q3_mec_clear._EARLY_CLEAR_RMAX`（那是 Q3/Q4 共用模块）——
    #   在这里包一层传参，保持 Q4 改动全部落在 Q4 自己的文件里。
    Q.go_clear = (lambda rb, kb, ch, *a, **k:
                  MC.go_clear_mec(rb, kb, ch, *a, early_rmax=Q4_EARLY_RMAX, **k))
    # ★★ 2026-09-13 清源原语可换（见 `go_clear_mec_v2` 的说明）。
    #    `clear_mode="localmesh"` → 圆心处 `no_signal` 不再掉进 `capture`，
    #    改用在**已知 MEC 半径**下铺一张必然命中的本地网。
    if clear_mode == "prodv2":
        # ★★★ 2026-09-13【仅为「可证伪」而存在的一个模式】
        #
        # `defer_fail>0` 必须把 `Q.go_clear` 换成 `go_clear_mec_v2`
        # （生产版 `MC.go_clear_mec` 在**内部**就烧掉了 `capture`，外面拦不住）。
        # 于是「延后」实验的两臂天然差了**两个**东西：原语 + 延后。
        # 本模式 = v2 但 `on_nosignal="prod"` + 不延后 ⟹
        # **`prodv2` 与 `prod` 必须逐位相同**，否则延后实验不可归因。
        # `q4_defer_paired.py` 的锚点 2 就在验这一条。
        Q.go_clear = (lambda rb, kb_, ch, *a, **k:
                      go_clear_mec_v2(rb, kb_, ch, *a, on_nosignal="prod",
                                      early_rmax=Q4_EARLY_RMAX, **k))
    elif clear_mode == "raymesh":
        # 沿【最后一条示向度的射线】搜（`on_nosignal="ray"`，v2 的默认）
        Q.go_clear = go_clear_mec_v2
    elif clear_mode == "localmesh":
        # 绕【MEC 圆心】铺方格网（依赖误差模型；官方未复现，留作对照）
        Q.go_clear = (lambda rb, kb, ch, *a, **k:
                      go_clear_mec_v2(rb, kb, ch, *a, on_nosignal="mesh", **k))
    # ══════════════════════════════════════════════════════════════════════
    # ★★★ 2026-09-13【定向延后清源】`defer_fail = N > 0`
    #
    # 用户提案：「是不是我们太早开始清理了，导致每个源的位置还不是很精确，
    # 所以需要这么多兜底机制？如果能在某个时刻把源的范围框得比较小了再开始清理…」
    #
    # ## 与【已被否定的全局提阈值】的区别（这是关键，否则会重复劳动）
    #
    # 已测过「把所有频道的停测阈值从 2 提到 4」：−164 s/局，$t=-1.24$，
    # **样本外 +18 s/局 —— 平局**。原因是它让**每一个**频道都多测 2 次
    # （无论它本来会不会清成功），付了成本却没换来收益。
    #
    # 本改动**只作用在本来就清失败的频道上**：
    #   · 清成功 ⟹ 进 `kb.cleared` ⟹ `kb.dead` ⟹ 立即不再被测量（零成本）
    #   · 清失败 ⟹ **不烧 `capture`（~1941 m）**，留在「已定位未清除」状态，
    #     在后续常规停点继续攒示向度（$R_c$ 中位 37.9 → 15.8 m），再试
    #   · 试满 `N` 次仍不成 ⟹ 才允许掉进 `capture`/`corridor_sweep`（正确性不变）
    #
    # ## 三处必须同时改，缺一不可
    #
    # | # | 改什么 | 为什么 |
    # | --- | --- | --- |
    # | 1 | `Q.go_clear` 带 `defer_fail=N` | 失败后**返回 False 而不是烧兜底** |
    # | 2 | `scan_at` 改 `stop_two=False` | 否则 ≥2 条的频道被跳过，$R_c$ 永远不动 |
    # | 3 | `tries` 上限 `2 → 2+N` | 否则第 2 次失败后它被踢出清源目标集 |
    #
    # ⚠ 第 2 条**不会**把请求数打回 1776–2522 那种量级：`kb.dead = cleared|empty`，
    #   已证空/已清除的频道本就在 `dead` 里被 `scan_at` 跳过；真正被额外测到的
    #   只有「已定位未清除」的那 0~2 个频道，且攒到 `_BEAR_MAX` 条即停。
    # ══════════════════════════════════════════════════════════════════════
    _BEAR_MAX = 5        # 5 条时 p(R_c≤20)=100%（见 go_clear_mec_v2 尾部的表）
    if defer_fail > 0:
        _df = int(defer_fail)
        # ⚠⚠ **必须显式传 `early_rmax=Q4_EARLY_RMAX`**：`go_clear_mec_v2` 的默认是
        #    `None` ⟹ 回落到 `MC._EARLY_CLEAR_RMAX`（Q3 的 **40**），
        #    会把本轮最大的单项收益（40→100，−46 s/个）**悄悄丢掉**。
        #    （顺带记一笔：`clear_mode="raymesh"/"localmesh"` 两条已否决的路径
        #      就是这个写法 —— 它们当年是拿一个**更弱的 prod 基线**比出来的，
        #      结论方向不受影响，但复跑时要知道这一点。）
        # ⚠⚠ **必须带 `on_nosignal="prod"`**：否则 `no_signal` 分支会走
        #    `ray_mesh_clear`（即已否决的 `raymesh` 方案），两臂就差了**两个**变量，
        #    结果无法归因。实测踩过这个坑（见 `go_clear_mec_v2` 里的同一条注释）。
        Q.go_clear = (lambda rb, kb_, ch, *a, **k:
                      go_clear_mec_v2(rb, kb_, ch, *a, defer_fail=_df,
                                      on_nosignal="prod",
                                      early_rmax=Q4_EARLY_RMAX, **k))
    # 清源尝试次数的上限：`tries[ch]` 超过它就被踢出「清源目标集」，交给收尾。
    # `defer_fail>0` 时放宽，否则延后的那几次根本没机会被执行到。
    # `try_cap` 可用命令行/参数覆盖，用于把「放宽重试次数」这个变量**单独隔离**出来
    # （否则 D 臂相对 B 臂差了「延后 + 次数」两个东西，又是一次不可归因的对比）。
    _CAP = int(try_cap) if try_cap is not None else 2 + max(0, int(defer_fail))
    # 【两个开关**完全正交**】`defer_fail` 与 `meas_uncleared` 必须能独立开关，
    # 否则无法归因。
    #
    # 实测教训（2026-09-13，两次踩同一类坑）：
    #   1. 最初把 `_MEAS_ON` 写成 `meas or defer>0` ⟹ 2×2 里的
    #      「只延后」和「测量+延后」两臂**逐位相同**（12/12 局），白跑一轮。
    #   2. 更早那版把 defer 与 v2 的 `on_nosignal="ray"` 绑在一起，
    #      两臂差了两个变量，−210 s/局 说不清是谁的功劳。
    # ⟹ **做对照实验时，臂之间只能差一个布尔量。**
    _MEAS_ON = bool(meas_uncleared) or bool(meas_due)
    # ★★★ 2026-09-13【`meas_due`：只补测「即将轮到清」的频道】—— 对 `meas_uncleared` 的收窄
    #
    # ## 为什么收窄（`meas_uncleared` 已实测清楚，见 `q4_defer_paired.py` 的说明）
    #
    # `meas_uncleared` 的作用链**已直接测量**（在 `MC.go_clear_mec` 上挂探针，12 局）：
    #
    # | | `go_clear` 内部移动次数 | 内部走路/局 |
    # | --- | --- | --- |
    # | A 生产 | **2.8** | 11186 m |
    # | B 继续测量 | **2.1** | 9608 m |
    #
    # ⟹ 继续测量让位置估计在 `go_clear` **被调用之前**就收敛，于是第一次探测
    #    就满足 `reg.clear_ok`，不必再"走到圆心→测→重算圆心→再走"。
    #    **省的是 `go_clear` 内部那几轮探测游走的走路（−1578 m/局，占总走路降幅 84%）**，
    #    跟 `capture` 兜底基本无关（`capture` 本来只有 2.1 次/局）。
    #
    # ## 但它的成本付得太早、太宽
    #
    # `meas_uncleared` 从「拿到 2 条示向度」起就在每个停点测，平均 **+43 次/局
    # ≈ +210 s**；而走路只省 −377 s ⟹ 净 −140 s/局，**典型局反而 +108 s**
    # （频道定位太晚时，补测付了钱却来不及把走路省回来）。
    #
    # ## 收窄判据
    #
    # 只在「当前点就是剩余巡回点里离它最近的那个」时补测 —— **与
    # `clear_located_pick` 用的是同一个判据**。于是补测的正好是**接下来就要清的
    # 那几个频道**，攒到的示向度立刻被 `go_clear` 用掉，不会白测。
    _DUE = bool(meas_due) and not bool(meas_uncleared)
    # ★★ 清源兜底原语可换（2026-09-13）。
    #    `capture_mode="fine"` → 用本模块的 `capture_fine`（定步长、无过冲、无二分），
    #    替掉 `q3_strategies.capture`（几何步长 ×1.6 + 二分，实测 ≈2.2D）。
    #    ⚠ 只打补丁、**不改 `q3_strategies.py` 一个字** —— 该文件是 Q3/Q4 共用命名，
    #      改它会同时污染第三题。这与 `Q.go_clear = MC.go_clear_mec` 是同一套做法。
    #    ⚠ 挂载是**全局状态**，配对实验必须在两臂之间存/恢复（见 `q4_capture_paired.py`）。
    if capture_mode == "fine":
        Q.capture = capture_fine
    # ★★ 关掉 `capture` / `corridor_sweep`：它们都是**信号驱动**的
    #   （沿示向度射线逼近、遇 `no_signal` 退出/绕行），而定向源的困难恰恰是
    #   「走到距真源 7 m 却因在扇区外而收到 no_signal」——
    #   实测这两个函数在 Q4 里白烧 **7539 m**（占总位移 14.6%）。
    #   收尾改由 `mesh_clear` 承担（按几何覆盖，与方向无关）。
    # ⚠ 试过把 `capture`/`corridor_sweep` 彻底关掉（归因说它们烧了 7539 m）：
    #   **结果更差，1127 → 1211 s/个**。那 7539 m 不是浪费 —— 它们在**实际清源**，
    #   关掉后 `mesh_clear`（81 点 × 9 s ≈ 730 s/源）要多跑很多次。
    #   ⟹ 归因只能指出「哪里花得多」，**不能推出「哪里该砍」**。
    # ★ `angular=False` 用于**隔离实验**：同一布局只关掉角度判据，
    #   量出「可靠性」到底值多少秒。（原先 `run_q4` 无条件设 True，
    #   导致外部打补丁无效 —— 那个实验因此作废。）
    kb.dir_aware = bool(angular)
    # ⚠ 不设 `required_points` ⟹ 走【增量判定】（`covers_arena_dir_fast`）。
    #   原先是「整版走完才判空」，那会让每个空频道在每个扫描点都被测一次，
    #   实测请求数 1776–2522 次 —— 官方真实时间预算只有 20 分钟，扛不住。
    #   增量判定让空频道一旦被证明就停止测量。
    tries: dict[int, int] = {}

    def located():
        return [c for c in Q.CHANNELS
                if c not in kb.cleared and c not in kb.empty
                and kb.estimate(c) and tries.get(c, 0) < _CAP]

    def clear_located(force: bool = False):
        todo = [c for c in Q.CHANNELS if kb.estimate(c)]
        if not force:
            todo = [c for c in todo if tries.get(c, 0) < _CAP]
        todo = [c for c in todo if c not in kb.cleared and c not in kb.empty]
        for ch in Q._order_nn(kb, todo, (robot.x, robot.y)):
            if ch in kb.cleared or ch in kb.empty:
                continue
            tries[ch] = tries.get(ch, 0) + 1
            if not Q.go_clear(robot, kb, ch) and force:
                e = kb.estimate(ch)
                if e:
                    x, y = Q.clamp_to_arena(e[0], e[1])
                    robot.move_to(x, y)
                    if robot.clear(ch) == "success":
                        kb.cleared.add(ch)

    # ★★★ 「顺路才清」（2026-09-12 晚，独立验证 −95.9 s/个，12/12 局一致，配对 t=−11.76）
    #
    # ## 问题
    #
    # 原版在「**第一次**能定位到源」的巡回点就清源，不管当前点离源多远。
    # 而源被定位的位置取决于它被看到两次的位置，**可能离源上千 m** ——
    # 巡回后面往往还有更靠近它的点。在那个远处点上清源，绕路就长。
    #
    # ## 改法
    #
    # 只在「当前点就是**剩余巡回点里离它最近的那个**」时才清。
    #
    # ## 为什么安全（三条独立论证）
    #
    # 1. **不增加任何一次测量**：`scan_at(..., stop_two=True)` 会跳过已有 ≥2 条
    #    示向度的频道，而 `estimate()` 非空 ⟹ 至少有 2 条 ⟹ **已定位的源本来
    #    就不会再被扫**。推迟清源与检测次数完全解耦。
    # 2. **不会漏清**：剩余点里没有更近的 ⟹ 当场清；走到队尾仍未清的，
    #    由收尾既有的 `clear_located(force=True)` + 细网盲清兜住。
    # 3. **不影响证空判据**：判据是**点集**的性质（诊断 §2：任何真子集都不满足），
    #    与清源发生在哪一点无关。
    #
    # ## 实测（独立实现 + 保真锚点逐位相同）
    #
    # | | A 现行 | B 顺路才清 |
    # | --- | --- | --- |
    # | 中位虚拟时（n=14, dir=0.6, 12 配对局）| 9479 s | **8177 s** |
    # | s/个 | 695.7 | **599.7** |
    # | 移动 | 36930 m | 30212 m（**−18.2%**）|
    # | 全清 | 12/12 | **12/12** |
    # | 配对 t | — | **−11.76**，**12/12 局全部变快** |
    #
    # 复跑：`python q4_pick_paired.py --seeds 12 --n 14 --dir 0.6`
    def clear_located_pick(route, idx):
        """只清「当前点就是剩余巡回点里离它最近的那个」的已定位源。"""
        now = route[idx]
        rest = route[idx + 1:]
        todo = [c for c in Q.CHANNELS if kb.estimate(c)
                and c not in kb.cleared and c not in kb.empty
                and tries.get(c, 0) < _CAP]
        if not rest:
            pick = todo
        else:
            pick = []
            for c in todo:
                e = kb.estimate(c)
                q_ = (e[0], e[1])          # ⚠ estimate 返回 (x, y, 残差) 三元组
                d_now = math.dist(now, q_)
                d_best = min(math.dist(p, q_) for p in rest)
                if d_now <= d_best + 1e-9:
                    pick.append(c)
        if not pick:
            return
        for ch in Q._order_nn(kb, pick, (robot.x, robot.y)):
            if ch in kb.cleared or ch in kb.empty:
                continue
            tries[ch] = tries.get(ch, 0) + 1
            Q.go_clear(robot, kb, ch)

    def _due_chans(now, rest):
        """「即将轮到清」的已定位频道。

        ⚠ **判据必须与 `clear_located_pick` 逐字相同**（那边是清、这边是补测），
          否则会去测一堆"马上不清"的频道，白付测量钱 —— 那正是
          `meas_uncleared` 的问题所在。
        """
        out = []
        for c in Q.CHANNELS:
            if c in kb.dead or len(kb.bearings[c]) < 2:
                continue
            e = kb.estimate(c)
            if not e:
                continue
            q_ = (e[0], e[1])
            if not rest or math.dist(now, q_) <= min(math.dist(p, q_)
                                                     for p in rest) + 1e-9:
                out.append(c)
        return out

    def _scan_order(now=None, rest=None):
        """★ 2026-09-13【当前频道优先】`scan_current_first`（**默认开**）

        ## 改什么

        每个测站依次测 k 个频道。原来按 `Q.CHANNELS` 的固定顺序测，
        于是**每站至少要多付一次换频**（首个频道与机器当前频道不同）。
        改成从**当前频道**起轮转之后，测站内的第一个频道就是机器已经待着的那个。

        **测量集合完全不变**，只改同站内的顺序。

        ## 依据：与队友独立实现的隔离配对对照（`_iso_compare/`）

        4 配置 × 20 局 × 5 臂 = 400 次配对（n=12/14/16 定向 1.0 + n=14 定向 0.7）：

        | | 跨配置 Δ | 跨配置 t | 方向 | 逐局 |
        | --- | --- | --- | --- | --- |
        | **本改动** | **−23 s/局** | **−18.76** | **4/4 配置为负** | **每一配置 20/20 局更快** |
        | 队友 21 站布局 | −101 | −0.22 | 3/4 负 | ⚠ n=16 漏清 1 |
        | 混合加密 41 点 | +2873 | +13.04 | 0/4 负 | — |

        ⟹ 幅度小（−1.2 ~ −2.0 s/个 ≈ −0.2%），但**效应近乎确定性**
          （t 极大是因为每局省的都是同一次换频），**零漏清、零风险**。
          同批实验里"看起来更大的改动"（换布局）反而符号不稳且引入漏清。

        ## 为什么默认开

        实验要测的应当是**实际出厂的那个配置**。默认关会制造
        「测的 A、发的 B」这类脚枪。要复现历史基线请显式传
        `scan_current_first=False`（详见 `Q4-08` §3）。
        """
        chans = _meas_chans(now, rest)
        if scan_current_first:
            cur = getattr(robot, "channel", None)
            if cur in chans:
                _j = chans.index(cur)
                chans = chans[_j:] + chans[:_j]
        return chans

    def _meas_chans(now=None, rest=None):
        """常规停点的待测频道表（只在 `defer_fail>0` 时不同于原行为）。

        原行为 = `scan_at(..., stop_two=True)` ⟹ 拿到 ≥2 条示向度就永久跳过它。
        `defer_fail>0` 时**已定位但未清除的频道继续测**，直到攒够 `_BEAR_MAX` 条。

        ## 为什么这一条是必须的（否则整个「定向延后」方案失效）

        一个频道在 2 条示向度处清失败后，如果它**不再被测量**，$R_c$ 就永远
        停在 37.9 m，第 2 次尝试与第 1 次是**同一个问题**，延后毫无意义。
        实测（`mid_stops` 那轮的机制隔离）已经撞过一次这个坑：中间补测点的
        $R_c$ 与不补测**逐位相同**，就是因为过滤器把频道挡在了外面。

        ## 为什么不会把请求数打回去（`defer_fail` 的请求预算是封顶的）

        `kb.dead = kb.cleared | kb.empty`，而 `scan_at` 先跳 `dead` ⟹
        已清除/已证空的频道**本来就一次都不测**。真正多出来的只有
        「已定位、未清除、未证空」的频道 —— 正常情况下 0~2 个，
        而且攒到 `_BEAR_MAX` 条就停 ⟹ 额外测量 ≤ 2 个 × 3 次 × 4.87 s ≈ 29 s/局。
        """
        if not _MEAS_ON:
            return list(Q.CHANNELS)
        extra = None if not _DUE else set(_due_chans(now, rest))
        out = []
        for c in Q.CHANNELS:
            if c in kb.dead:
                continue
            n = len(kb.bearings[c])
            if n >= _BEAR_MAX:                # 攒够还清不掉 ⟹ 再测无益，交给收尾
                continue
            if n >= 2 and extra is not None and c not in extra:
                continue                      # `meas_due`：没轮到清就不补测
            out.append(c)
        return out

    n_scan = 0
    # ⚠ 试过极简内层（中心+环500×4+环1000×8，13 点）替代方格：**净效果为零**
    #   —— 赶路省 7052 m，但角度判据成立得更晚 ⟹ 空频道存活更久 ⟹ 检测数
    #   959→1038 次、清源赶路 11777→14246 m。总时 15995→16190 s（略差）。
    #   **这是本方案的第一个真实的此消彼长，说明已在局部最优附近。**
    # ★ 扫描路线用 TSP 而非「方格蛇形 + 绕环」：同一点集，只改顺序，
    #   实测 892.4 → 800.4 s/个（−10.3%），6/6 全清。见 `tsp_route` 的说明。
    _P = C.square_grid(step) + C.ring(k_ring, ARENA + ring_eps)
    if drop_outer:
        # ★★ 去掉方格最外圈的 8 个点（r≈1789）。
        #
        # 依据（2026-09-13，32 局配对）：
        #   · 这 8 个点在【判据】上是**纯冗余**：去掉后角度覆盖 170.02°、距离覆盖
        #     572 m 全部不变（逐点边际 = 0.00°）。机理：外推环在 ρ=1875，
        #     只比它们远 86 m，**在角度上完全支配了它们**。
        #   · 但它们同时是【巡回的踏脚石】：去掉后单跳变大。
        #     故 `tsp_route` 在 28 点实例上会陷入差解（19523 m）——
        #     这里用**多初值**修到 19139 m（只比 36 点的 18919 长 1.2%）。
        #
        # 实测 32 局配对：**−22.8 s/个**（t=−1.69，p≈0.10，12/16 与 14/16 两批
        # 独立种子一致负向），两臂全清 32/32。**未达显著但期望为正**：
        # 省检测 8 点×6 频道×6 s = 288 s/局是确定的，赔移动约 88 s/局。
        # ⚠ 这是一个**边际改进**，若官方验证不如预期应回退（`drop_outer=False`）。
        _P = [p for p in _P if abs(math.hypot(*p) - 1789.0) >= 1.0]
        best, best_len = None, float("inf")
        for _t in range(8):
            pp = list(_P)
            random.Random(_t).shuffle(pp)
            r = tsp_route(pp, (robot.x, robot.y), passes=200)
            L = C.tour_length(r)
            if L < best_len:
                best, best_len = r, L
        route = best
    else:
        route = tsp_route(_P, (robot.x, robot.y))
    # ══════════════════════════════════════════════════════════════════════
    # ★★ 2026-09-13【滚动重规划】`policy="roll"` —— **已从「否定」翻转为「候选」**
    #
    # ## ⚠⚠ 本条曾给出**错误的否定判决**，务必看完整段再引用
    #
    # 最初测得「合并 68 配对局 t=−0.15，不显著；且漏清 1 次」⟹ 判否。
    # **该判决建立在两个缺陷上，都由用户质疑后发现：**
    #
    # ### 缺陷 1：两臂的 TSP 求解强度不同（测试不公平，偏袒 A 臂）
    #   · A 臂（`pick`）初始巡回：**8 初值 × 200 遍**
    #   · B 臂（`roll`）每次重解：**6 初值 × 120 遍**  ← 被喂了个更弱的求解器
    #   拉平后：均值 −57 → **−254 s/局**，移动差 −916 → −1624 m/局。
    #
    # ### 缺陷 2：`_patho` 频道的「大网兜底」根本铺不开（真 bug）
    #   `mesh_clear` 用 `limit=400` 计数截断，而大网梯子半径到 2400 m、
    #   步长 40 m（需 14641 探点）⟹ **半径 ≥600 的三层只扫了 400 点就退出，
    #   且枚举从 `-n` 起，截断时搜的是区域边角而非中心**。
    #   实测 n=12 seed12 频道1（估计偏差 **1742 m**）：白烧 17600 s **且漏清**。
    #   修法见收尾段注释：`_patho` ⟹ 改用 `ray_mesh_clear` 沿**射线**搜
    #   （估计不可信但示向度可信，搜索空间本是 1-D）。
    #   修后该局：**31268 → 11653 s，11/12 → 12/12**。
    #   ⚠ 该 bug **只在滚动路径下被触发**——固定巡回绕开了它，所以藏了很久。
    #
    # ## 最终判决（合并 68 配对局，5 个配置，缺陷全部修复后）
    #
    # | 量 | 值 |
    # | --- | --- |
    # | 均值配对差 | **−542 s/局 = −37.4 s/个** |
    # | 配对 t | **−2.85（显著）** |
    # | B 更快的局数 | 49/68 |
    # | 移动差 | **−2776 m/局（−8.9%）** |
    # | 漏清 | **A=0 B=0** |
    #
    # ⚠ **仍未官方验证**；采纳前必须官方复跑。
    #
    # ---
    #
    # ## 历史记录（**错误判决**，留档以免重蹈）
    #
    # | 配置 | A 固定巡回 | B 滚动 | 差 |
    # | --- | --- | --- | --- |
    # | n=10 dir=1.0 | 8966 | 8708 | −257 |
    # | n=12 dir=1.0 | 9220 | 10960 | **+1740** |
    # | n=14 dir=1.0 | 9259 | 8633 | −626 |
    # | n=16 dir=1.0 | 10976 | 10615 | −361 |
    # | n=14 dir=0.6 | 8022 | 7532 | −490 |
    #
    # **合并：均值 −57 s/局、中位 −480 s、`t = −0.15`、46/68 更快 ⟹ 不显著。**
    #
    # ## 两个必须记住的教训
    #
    # 1. **单配置的显著性会骗人。** 先只跑 n=14 dir=0.6 的 20 局，得到
    #    **−490 s/局、t=−2.98、15/20 更快**，看起来板上钉钉；扩到 5 个配置
    #    合并后掉到 t=−0.15。**单配置 20 局不足以定论跨配置的结论。**
    #    （同 `对-28点布局` 那次在 8 个种子上误判的教训 —— 这是第二次。）
    # 2. **它引入漏清。** n=12 seed12：频道 1 拿到 **6 条示向度**（定位充分）
    #    却整局未清。机理：滚动把源排进计划后，`go_clear` 失败使 `tries` 到 2，
    #    于是被 `_roll_build` 的 `tries.get(c,0) >= 2` **永久排除**，而共用的
    #    收尾路径没能兜住它。**漏清 = 判负**，不是降级。
    #    A 臂 0/68，B 臂 1/68。
    #
    # ## 但这次调查产出的东西是有价值的（见下方归因）
    #
    # 成本模型与「清源走位占全部走位 62%」的定位是**独立的结论**，
    # 与滚动方案是否采纳无关，可直接写进论文。
    # ══════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════
    # 【滚动重规划】`policy="roll"` 的实现（**已否定，仅作证据留存**）
    #
    # ## 动机（官方 10 局日志的实测归因）
    #
    # 用 `(移动, 检测, 换频, 清除)` 对虚拟时间做最小二乘，**成本模型精确**
    # （10 局，R²=0.9999985，最大残差 2.1 s）：
    #     移动 0.200585 s/m（≈4.985 m/s）｜检测 4.871 s｜换频 1.073 s｜清除 3.110 s
    # ⟹ **走位占 76.5%**、检测 18.2%、切换+清除 5.0%。
    #
    # 再把走位按「扫点之间」与「清源往返」拆开（判定：两端是否都落在布局停点 30 m 内）：
    #     扫点间 10688 m/局（占走位 38%） ｜ **清源 17420 m/局（占走位 62%）**
    # 清源走位**比整条扫描巡回（~18800 m）还长**，单项吃掉约 **47% 的虚拟时间**。
    #
    # 而用日志里「/clear 成功」的位置反推真源坐标（成功 ⟹ 发令点距源 ≤20 m），
    # 求「28 停点 ∪ 全部真源」的最优巡回：
    #     理想 **21444 m/局**  vs  实测 **28108 m/局**  ⟹ **可省 6664 m/局 ≈ 1336 s/局**
    # 理想巡回本身就**包含访问并清除全部源**，故这是可达的（非空中楼阁）。
    #
    # ## 现行为什么浪费
    #
    # 现行把【停点】排成一条固定巡回，清源只是「插进去再退回来」：
    # `clear_located_pick` 只在「当前停点已是剩余停点中离源最近的那个」时清。
    # 但「离源最近的停点」**未必是插入代价最小的那个位置** —— 它不看下一站去哪。
    # 源被摘出去单独往返，等于在巡回里挂了一条**不受优化的支路**。
    #
    # ## 改法
    #
    # 把「未访问停点 ∪ 已定位未清除的源」**放在同一个点集里**求 TSP，整条执行：
    #   · 目标集**变大**（新源被定位）时才重解，其余时间按既定顺序走
    #   · 停点目标 → `scan_at`；源目标 → `go_clear`
    #
    # ⚠ 「每步都重解、只执行第一个目标」是**错的**，Q3 已实测过
    #   （见 `q3_rolling_tsp.run_rolling` 的 docstring）：会在场地两侧来回折返，
    #   实测出现 2375 m 的单次大跳。**价值在「解一次、执行整条」，不在「重解」。**
    #
    # ⚠ 收尾段（下方 `kb.refresh_empty(force=True)` 到 `ray_mesh_clear`）
    #   两臂**共用**，不随 policy 分叉 —— 保证配对对比只动「扫描+清源调度」这一项。
    # ══════════════════════════════════════════════════════════════════════
    if policy == "roll":
        _stops_left = list(_P)
        _plan: list = []
        _planned: frozenset = frozenset()

        def _roll_build():
            """对「未访问停点 ∪ 已定位未清除源」求一次 TSP（自当前位置出发）。"""
            cur = (robot.x, robot.y)
            tg, meta = [], []
            for p in _stops_left:
                tg.append(p)
                meta.append(("stop", p, None))
            loc = set()
            for c in Q.CHANNELS:
                if c in kb.cleared or c in kb.empty:
                    continue
                e = kb.estimate(c)
                if not e or tries.get(c, 0) >= _CAP:
                    continue
                loc.add(c)
                tg.append((e[0], e[1]))
                meta.append(("src", (e[0], e[1]), c))
            if not tg:
                return [], frozenset()
            # ⚠⚠ 2026-09-13 公平性修正：原来这里只用 6 初值 × 120 遍，
            #   而 A 臂（`pick`）的初始巡回用的是 **8 初值 × 200 遍** ——
            #   **B 臂被喂了个更弱的求解器**，对比不公平（用户指出）。
            #   改成与 A 臂同强度。
            best, bl = None, float("inf")
            for _t in range(8):
                pp = random.Random(_t).sample(tg, len(tg))
                r_ = tsp_route(pp, cur, passes=200)
                L_ = C.tour_length(r_)
                if L_ < bl:
                    best, bl = r_, L_
            byid = {id(t): i for i, t in enumerate(tg)}
            return [meta[byid[id(t)]] for t in best], frozenset(loc)

        for _ in range(4 * max_pts):
            kb.refresh_empty()
            if not [c for c in Q.CHANNELS if c not in kb.dead]:
                break
            # ★★ 题面上界驱动的提前停机（同 `pick` 臂，逐字相同）。
            if early_stop and sum(1 for c in Q.CHANNELS if kb.bearings[c]) >= 16:
                print("  [提前停机] 16 个频道已给出示向度 = 题面总数上界，停止扫描",
                      flush=True)
                break
            if not _plan:
                _plan, _planned = _roll_build()
                if not _plan:
                    break
            _kind, _p, _ch = _plan.pop(0)
            if _kind == "stop":
                if _p not in _stops_left:          # 重解前的残留项，已消费过
                    continue
                Q.scan_at(robot, kb, _p[0], _p[1],
                          _scan_order(_p, [s for s in _stops_left if s != _p]),
                          stop_two=not _MEAS_ON)
                n_scan += 1
                _stops_left.remove(_p)
            else:
                if _ch in kb.cleared or _ch in kb.empty:
                    continue
                tries[_ch] = tries.get(_ch, 0) + 1
                Q.go_clear(robot, kb, _ch)
            kb.refresh_empty()
            # ★ 只在**目标集变大**（有新源被定位）时重解 —— Q3 的教训
            _now = frozenset(c for c in Q.CHANNELS
                             if c not in kb.cleared and c not in kb.empty
                             and kb.estimate(c) and tries.get(c, 0) < _CAP)
            if _now - _planned:
                _plan, _planned = _roll_build()
    else:
        for _i, (x, y) in enumerate(route[:max_pts]):
            kb.refresh_empty()
            if not [c for c in Q.CHANNELS if c not in kb.dead]:
                break
            # ★★ 题面上界驱动的提前停机（用户提出；用 `early_stop` 开关做配对对照）。
            #    题面「总数在 10-16 个之间」⟹ 一旦 **16 个不同频道**给出过示向度，
            #    总数就确定为 16，其余频道必空 —— 不需要任何覆盖证明。**这是可靠的**
            #    （用尽题面给出的上界），不是启发式。
            #
            #    ⚠ 但对抗评审在 **Q3** 上实测同类做法为 **−217.7 s/局（t=−1.89，负收益）**：
            #      删掉一个停点会改变其后【每一次接近源的方向】，触发清源失败级联
            #      （实测 seed 1 只跳过 1 次访问却多走 6338 m）。Q4 的形态不同
            #      （跳过的是扫描点、不是哨兵），**必须单独配对实测再决定去留**。
            if early_stop and sum(1 for c in Q.CHANNELS if kb.bearings[c]) >= 16:
                print(f"  [提前停机] 16 个频道已给出示向度 = 题面总数上界，停止扫描", flush=True)
                break
            Q.scan_at(robot, kb, x, y, _scan_order((x, y), route[_i + 1:]),
                      stop_two=not _MEAS_ON)
            n_scan += 1
            # ★★★ 2026-09-13【段内补测】`mid_stops > 0` 时，去下一站的路上插
            #   `mid_stops` 个补测点，**只测「已定位未清除」的频道**。
            #
            # ## 为什么只测那些频道（这才是成本可控的关键）
            #
            # · **证空判据只需要「覆盖集」** —— 现有 28 个停点已经是覆盖集了
            #   ⟹ 新插的点**不需要**测空频道（否则一个点要 ~20 频道 × 4.87 = 97 s）
            # · 补测点只服务「把 $R_c$ 压小」，所以只针对待复清的频道，通常 0~1 个
            # ⟹ **每个补测点约 4.87 s**
            #
            # ## 走路几乎不增（点插在路径上，三角形不等式 ⟹ 绕行 ≈ 0）
            #
            # ## 依据：定位不确定度 ∝ 1/√k（同一跨度上加密）
            #   跨度 1600 m、源垂距 600 m 的解析计算：
            #     测点间距 800 → 400 → 200 → 100 m
            #     不确定度 15.4 → 11.1 → 8.4 → 6.1 m（0.72× → 0.54× → 0.40×）
            if mid_stops > 0 and _i + 1 < len(route[:max_pts]):
                _nx, _ny = route[_i + 1]
                for _m in range(1, mid_stops + 1):
                    _t = _m / (mid_stops + 1.0)
                    _px, _py = x + (_nx - x) * _t, y + (_ny - y) * _t
                    _pend = [c for c in Q.CHANNELS
                             if c not in kb.cleared and c not in kb.empty
                             and kb.estimate(c) and tries.get(c, 0) < _CAP]
                    if not _pend:
                        break
                    Q.scan_at(robot, kb, _px, _py, _pend, stop_two=False)
                    n_scan += 1
                kb.refresh_empty()
            kb.refresh_empty()
            if located():
                # ★★★ 「顺路才清」：只在「当前点就是剩余巡回点里离它最近的那个」时才清。
                #     原版是无条件 `clear_located()`（第一次能定位就清，不管多远）。
                #     独立验证 −95.9 s/个、12/12 局一致、t=−11.76。详见 `clear_located_pick`。
                clear_located_pick(route, _i)
                kb.refresh_empty()

    # ★★ 扫描结束后【强制复检一次】判据。
    #    `refresh_empty` 有性能节流（测点数每 +25 才复检），33 个点的布局下
    #    最后一次检查会停在 25/33 —— 而判据需要完整布局才成立，于是
    #    **空频道永远不会被判定**，「可证明终止」形同虚设（实测局末 6 个频道
    #    no_signal 满 33/33、判据成立却不在 kb.empty 里）。
    kb.refresh_empty(force=True)
    # ★ `defer_fail` 的计数器在收尾前**清零**：代价是——若某频道在扫描段里
    #   从头到尾没被试过（`_defer_cnt` 仍是 0），收尾这次也会被"延后"，
    #   于是掉不进 `capture`，只能靠 `clear_located(force=True)` 里那个
    #   「移到估计点直接 `/clear`」的弱回退 —— 那比 `capture` 弱得多。
    #   清零 ⟹ 收尾**永远是完整链**（`capture` → `corridor_sweep` → 网梯子），
    #   延后策略只作用于扫描段。**这是正确性的边界，不是优化。**
    #   ⚠⚠ **不是 `pop`（清零）——那是错的，实测漏清过。** 清零会让收尾的
    #   **第一次**尝试也被延后（`_cnt[ch] = 0+1 = 1 ≤ N`），于是掉不进 `capture`，
    #   只能落到 `clear_located(force=True)` 里那个「移到估计点直接 `/clear`」
    #   的弱回退 —— 而定向源的估计点可以偏几百米（共线时距离不可观测）。
    #   实测（n=16 定向 1.0，12 局）C/D 两臂各**漏清 1 个源**，A/B 两臂 0 漏。
    #   推高到阈值之上 ⟹ 收尾第一次起就是完整链，**正确性与不加 defer 时相同**。
    _cnt = kb.__dict__.setdefault("_defer_cnt", {})
    for _c in Q.CHANNELS:
        _cnt[_c] = _cnt.get(_c, 0) + defer_fail + 10 ** 6
    clear_located(force=True)
    # ⚠ 跳过 `capture` / `corridor_sweep`：它们是**信号驱动**的，
    #   而定向源的困难恰恰在于「走到 7 m 却因在扇区外而收到 no_signal」
    #   （见 `优化-问题4的时间下界…md` §1 的 `fix` 桶 7539 m 纯浪费）。
    #   直接走下面的细网盲清 —— 它按**几何**覆盖，与方向无关。
    # ★★ 定向源收尾：以估计位置为中心的细网盲清。
    #    定向源的估计偏差可以很大（示向度只来自扇区内那一小段），
    #    `capture` 的 24 m 局部网够不到；而 /clear 与方向无关，细网必然命中。
    # ★★ 多尺度递进盲清 —— 单一半径不够，这是实测逼出来的。
    #
    # 实测（n=15 dir=0.8 seed6，频道 1）：6 条示向度**全部自洽**（残差 ≤0.94°），
    # 但 6 个测点**几乎共线** —— 它们全在指向源的同一条射线上（`corridor_sweep`
    # 沿路走出来的）。**方位交会要求测点对源有张角；共线时距离不可观测**，
    # LS 估计因此偏了 **177.3 m**，且偏差恰好沿射线方向。
    # 固定 ±80 m 的网够不到；而机器人沿射线已经走到 22.0 m —— **只差 2 m**。
    #
    # 所以：第一层失败就放大半径重试。代价可控（外层只在前面全部失败时才跑）。
    _MESH_LADDER = ((80.0, 20.0), (200.0, 25.0), (420.0, 30.0))

    # ★★★ 病态源专用大网（2026-09-13）。触发条件见 `q3_mec_clear.fix_collinear`：
    #   `go_clear_mec` 在扫描期检测到「估计不稳定」（近共线交会）时会给该频道
    #   打上 `kb._patho` 标记。此时**估计位置不可信**，±420 m 的网可能整个铺偏 ——
    #   官方演练就出过：一个源 `/clear` **失败 807 次**，整局 1108 s/个还漏清。
    #
    #   大网的层数与半径按「不稳定度」的量级定：病态实测分布是
    #   `>3000 m（99 分位）～20000 m（最大）`，故末层给到 2400 m。
    #   步长 40 m（≤20 m 是清除半径的要求，但网只要**有一点**落在 20 m 内即可 ——
    #   步长越大越省，40 m 时每行每列仍保证有一次落入 20 m）。
    #
    # ⚠ 代价：大网只在**被判病态**的源上跑（实测触发率 ~2–4%），
    #   而它替掉的是「807 次失败 = 2421 s」那种灾难 ⟹ 期望值划算。
    _MESH_LADDER_BIG = ((200.0, 40.0), (600.0, 40.0), (1200.0, 40.0), (2400.0, 40.0))

    def mesh_all(center, big: bool = False):
        for rad, stp in (_MESH_LADDER_BIG if big else _MESH_LADDER):
            if mesh_clear(robot, kb, ch, center, radius=rad, step=stp):
                return True
        return False

    _patho = getattr(kb, "_patho", set())

    for ch in list(Q.CHANNELS):
        if ch in kb.cleared:
            continue
        e = kb.estimate(ch)
        if e:
            # ★★★ 2026-09-13 修正（**漏清的真因**）：
            #   `_patho` 的含义是「交会不良 ⟹ **位置估计**不可信」，
            #   但**示向度本身是准的**（±1°）。所以正确的一维搜索空间是
            #   「最后一条示向度所在的那条射线」—— 必然命中；
            #   **不是**估计点周围的二维网（网心本身就是错的）。
            #
            #   原实现拿偏差上千 m 的估计当网心、再铺一张根本铺不开的大网
            #   ⟹ 实测（n=12 seed12 频道1，定向1.0）估计偏差 **1742 m**：
            #     白烧 4 层 × 400 探点 ≈ **17600 s**，**且仍然漏清**。
            #   这是滚动重规划那批实验里唯一的漏清，也是 n=12 那档
            #   +1740 s 异常值的成因 —— **它同时是漏清源和时间黑洞**。
            if ch in _patho and kb.bearings[ch]:
                _px, _py, _th = kb.bearings[ch][-1]
                if not ray_mesh_clear(robot, kb, ch, _px, _py, _th):
                    mesh_all((e[0], e[1]))
            else:
                mesh_all((e[0], e[1]))
        elif kb.bearings[ch]:
            # ★★ 只有 **1 条示向度**的频道 —— 这个分支是血的教训。
            #
            # `fit_position` 需要 ≥2 条示向度，所以这类频道 `estimate()` 返回 None，
            # 于是 `clear_located` 和上面的 `mesh_clear` **两条路径都会跳过它**。
            # 而它**已经给过 direction，是真源**（`refresh_empty` 也正确地不会判它为空）
            # ⟹ **它从缝里漏下去，永远清不掉**。
            #
            # 实测（2026-09-12 官方 Q4 演练第 2 局）：频道 2 只在全部 33 个扫描点中的
            # 1 个点上落入扇区（拿到 1 条示向度），之后全是 no_signal；日志里它对
            # **/clear 的调用次数为 0**，整局清了 12 个却漏了它。
            #
            # 机理：角度判据只保证「至少 1 个测点落在扇区内」（探测），
            # **不保证 2 个**（定位）。定向源扇区只有 180°，这是必然会出现的情形。
            #
            # `capture` / `corridor_sweep` 正是为「已知在一条射线上、但不知道多远」
            # 设计的 —— 沿射线推进。这里必须保留它们。
            if not Q.capture(robot, kb, ch, budget=20):
                Q.corridor_sweep(robot, kb, ch)
            # ★★ `capture`/`corridor_sweep` 会沿射线**补测**，可能把示向度从
            #    1 条增加到 ≥2 条 —— 于是现在**有了估计位置**。但上面的 `if e:`
            #    已经判过（那时 e 还是 None），必须**回头看一次**。
            #
            #    实测（n=14 seed3 dir=1.0）：频道 13 进循环时 1 条示向度（est=None）
            #    → 走 elif 分支 → 沿射线测到 4 条示向度、`capture`/`corridor_sweep`
            #    跑了 3 轮却始终没进 20 m → 循环已过、无人回头 → **漏清**。
            #    而它的估计偏差只有 25 m，`mesh_clear`（±80 m、20 m 步长）里
            #    距真源最近的点只有 9.5 m —— **只要回头补一次就必然清掉**。
            e2 = kb.estimate(ch)
            if e2 and ch not in kb.cleared:
                # 同样是「病态 ⟹ 估计不可信、射线可信」：先沿射线搜（见上）。
                if ch in _patho:
                    _px2, _py2, _th2 = kb.bearings[ch][-1]
                    if not ray_mesh_clear(robot, kb, ch, _px2, _py2, _th2):
                        mesh_all((e2[0], e2[1]))
                else:
                    mesh_all((e2[0], e2[1]))
            elif ch not in kb.cleared:
                # ★★ 仍无估计位置 ⟹ `capture`/`corridor_sweep` **一条新示向度都没补出来**
                #    （它们是信号驱动的；定向源「走到 7 m 却在扇区外收 no_signal」正是死穴）。
                #    按附录2(8)，**根本不需要第 2 条示向度** —— 源就在这条射线上，
                #    沿射线铺 20 m 间距的网同样必然命中。这是**最后一道兜底**，
                #    防的是「静默漏清」（硬约束失败），不是省时间。
                _px, _py, _th = kb.bearings[ch][-1]
                ray_mesh_clear(robot, kb, ch, _px, _py, _th)
    # ★★ 失效签名自检 —— 把「静默漏清」变成「可检测的漏清」。
    #    一个频道给过 direction 就确知是真源；若它最终仍未清除，就是硬失败。
    #    实测踩过：2026-09-12 官方演练第 2 局，频道 2 拿到 1 条示向度后
    #    被所有收尾路径跳过，整局清了 12 个却漏了它，而程序「正常结束」。
    missed = [c for c in Q.CHANNELS if c not in kb.cleared and kb.bearings[c]]
    if missed:
        print(f"  ⚠⚠ 【漏清告警】{len(missed)} 个频道拿到过示向度（= 真源）却未清除："
              f"{missed}（各自的示向度数 "
              f"{[len(kb.bearings[c]) for c in missed]}）", flush=True)
    return kb, n_scan


# ★ 走网顺序（2026-09-13）：`"prob"` = 由内向外（按后验概率，命中前的探针数省 56~65%）；
#   `"serp"` = 原来的行优先蛇形（留作对照）。**运行时读取，便于做配对实验切换。**
MESH_ORDER = "prob"


def mesh_clear(robot, kb, ch, center, radius: float = 80.0,
               step: float = 20.0, limit: int = 400,
               order: str | None = None) -> bool:
    """**定向源的收尾杀手锏：以估计位置为中心的细网盲清。**

    ## 为什么必须有这一步

    实测的定向源失败模式（seed 1 频道 9）：机器人走到距真源 **7.0 m**
    却测到 `no_signal`（在扇区外），**无从知道自己在 7 米处**；
    而 `capture` 的「3×3 局部网」中心是**偏差 31 m 的估计**、半径只到 24 m，
    **够不到真源周围那个 20 m 球**。结果 `/clear` 打了 150 次全部落空。

    ## 依据

    附录2(8)：**`/clear` 只要求 $\le20$ m，与信号、与定向方向都无关。**
    所以只要在估计位置周围铺一张**间距 $\le20$ m** 的网，**必然有一次落在真源 20 m 内**
    —— 前提是网撒得够大（估计偏差 + 20 m 才算够）。

    代价：$(2R/\Delta+1)^2$ 次「走 20 m + 清一次」$\approx9$ s，
    取 $R=80,\Delta=20$ 时 $9\times9=81$ 次 $\approx730$ s，**完全可接受。**
    """
    # ★★ 2026-09-13 修正：**层半径必须按预算封顶**。
    #
    # 原实现是 `n = int(radius/step)` 后再用 `limit` 计数截断 —— 于是
    # 大网梯子 `(200,40) (600,40) (1200,40) (2400,40)` 里半径 ≥600 的三层
    # **只跑了 400 个探点就退出**，而 `limit=400`、步长 40 m 只够铺 ±400 m；
    # 更糟的是枚举从 `-n` 往上数，**截断时搜的是那个区域的边角，不是中心**。
    #
    # 实测代价（n=12 seed12 频道 1，定向 1.0）：估计偏差 1742 m ⟹
    # 大网四层白烧 1600 个探点 ≈ 17600 s，**且仍然漏清**。
    # ⟹ 「网的半径 ≥ 不稳定度 ⟹ 铺网必然命中」这个前提是假的。
    #
    # 现在按预算反解半径：`(2n+1)² ≤ limit`，保证**层半径与实际覆盖一致**，
    # 宁可少铺也不假装铺了。
    n = min(int(radius / step), max(0, int((math.sqrt(max(1, limit)) - 1) / 2)))
    # ★★ 2026-09-13【遍历顺序：改蛇形】原实现是**行优先**——
    #    `for i: for j: j 从 -n 到 +n`，于是每走完一行都要**横穿整行宽度回到左端**
    #    才能开始下一行。多走 $2n\cdot 2ns$（约一倍的行间路程）。
    #
    # ## 蛇形对这个网格是【严格最优】的（可证）
    #
    #    网格上任意两点最小距离为 step ⟹ 访问全部 N=(2n+1)² 个点的路径
    #    有 N−1 条边、每条 ≥ step ⟹ 长度下界 (N−1)·step。
    #    蛇形（行内相邻、行间一步跨）**每条边恰好 = step** ⟹ 达到下界 ⟹ 最优。
    #
    # 实测（R=80, step=20 ⟹ n=4，行宽 160 m）：
    #    行优先：9×160 + 8×160 = 2720 m
    #    蛇形  ：9×160 + 8×20  = 1600 m   ⟹ **省 1120 m（41%）**
    #
    # ⚠ 这与 `ray_mesh_clear` 早已采用的写法一致（那边写的是 `if row % 2: ks.reverse()`）
    #   —— 方形网这边当初漏了。**只改次序，网格点集不变 ⟹ "必然命中"的保证不受影响。**
    # ★★ 2026-09-13【进网位置：从离机器人最近的那个角进】
    #
    # 蛇形只保证了**网格内部**最优（(N−1)·step，已证为下界）。但机器人要
    # **从外面进网**，固定从 `(-n,-n)` 进的话，这段进场路程没被优化。
    #
    # 代价取决于机器人离网心多远，两类网差别很大：
    #   · **局部网**（`go_clear_mec_v2` 用）：机器人**本来就在网心**（MEC 圆心），
    #     进场 = n·step·√2 ≈ 5.6% ⟹ 端到端不到 1 s/个，**可忽略**。
    #   · **尾段大网**：机器人从任意位置过来，可能上千 m。
    #     420 m 层的网跨度 840 m，**从远角进 vs 从近角进最多差 840√2 ≈ 1188 m
    #     = 238 s** ⟹ 这才是"和外界连接"该优化的地方。
    #
    # ⚠ **出网不用管**：`/clear` 一成功就当场停，不存在"走完再出来"。
    # ★★ 2026-09-13【遍历顺序：由内向外（按后验概率）】—— 本轮最大的一处
    #
    # ## 为什么
    #
    # 网格内的**后验分布不是均匀的**：真源满足的是若干条「方位条带」的交
    # ⟹ 最密处在**中心（最小二乘估计）**，越靠边越稀。
    # 而原实现从**角落**起步蛇形 ⟹ 命中位置在遍历序里近似**均匀随机**。
    #
    # ## 收益（模拟：源按 σ=R_c/2.5 的高斯采样、截断在 R_c 内）
    #
    # | $R_c$ m | 格 | 行优先蛇形 | **中心向外** | 省 |
    # | --- | --- | --- | --- | --- |
    # | 40 | 5×5 | 12.2 | **4.3** | 65% |
    # | 80 | 9×9 | 39.6 | **15.6** | 61% |
    # | 120 | 13×13 | 83.6 | **36.7** | 56% |
    #
    # 探针数直接正比于时间（每根 = 移动 + `/clear`）。**遍历仍走完（保证不变），
    # 只是命中通常发生在头几个探针，后面根本不用走。**
    #
    # ⚠ 这是**纯重排**：搜索空间、覆盖集、分支触发全都不变。
    # 按 `Q4-06 §4.3` 的判据属于**几何类 ⟹ 离线可信**。
    _order = MESH_ORDER if order is None else order
    if _order == "prob":
        # 由内向外；同半径按方位角排序，保证相邻探针仍挨着（不增走路）
        _cells = sorted(((i, j) for i in range(-n, n + 1) for j in range(-n, n + 1)),
                        key=lambda ij: (ij[0] ** 2 + ij[1] ** 2,
                                        math.atan2(ij[1], ij[0])))
    else:
        _i0, _i1, _di = ((-n, n + 1, 1) if robot.x <= center[0]
                         else (n, -n - 1, -1))
        _rev0 = robot.y > center[1]              # 首行从 y 近端起
        _cells = []
        for _ii, i in enumerate(range(_i0, _i1, _di)):
            _js = list(range(-n, n + 1))
            if bool(_ii % 2) ^ _rev0:            # 蛇形：隔行反向，且首行由 _rev0 定侧
                _js.reverse()
            _cells.extend((i, j) for j in _js)
    for i, j in _cells:
        if limit <= 0:
            return False
        limit -= 1
        x, y = Q.clamp_to_arena(center[0] + i * step, center[1] + j * step)
        robot.move_to(x, y)
        if robot.clear(ch) == "success":
            kb.cleared.add(ch)
            kb.refresh_empty()
            return True
    return False


RAY_STATS = {"trigger": 0, "hit": 0}


def ray_mesh_clear(robot, kb, ch, px: float, py: float, th_deg: float,
                   r_max: float = 1500.0, step: float = 20.0,
                   budget: int = 1200) -> bool:
    """**沿一条示向度射线铺网盲清** —— 「1 条示向度」频道的最后一道兜底。

    ## 它堵的是哪个洞

    收尾循环里 `elif kb.bearings[ch]`（只有 1 条示向度）那个分支：

        if not capture(...): corridor_sweep(...)
        e2 = kb.estimate(ch)
        if e2 and ch not in kb.cleared: mesh_all((e2[0], e2[1]), big=(ch in _patho))

    **`capture`/`corridor_sweep` 都是信号驱动的**（沿射线逼近、遇 `no_signal` 退出）。
    定向源的困难恰恰是「走到 7 m 却在扇区外、收到 `no_signal`」——
    于是它们可能**一条新示向度都补不出来** ⟹ `e2 is None`
    ⟹ **`mesh_all` 一次都不铺，这个频道静默漏掉。**

    ## 依据（与 `mesh_clear` 同源，但更强）

    附录2(8)：`/clear` **只要求 ≤20 m，与信号、与定向方向都无关**。

    所以**根本不需要第 2 条示向度**：源已知在「以最后一次测到示向度的点为起点的射线」
    上（方位 ±1°、距离 ≤$r_{\\text{eff}}$ ≤1500 m），
    **沿这条射线铺 20 m 间距的网，就同样必然有一次落在真源 20 m 内。**

    ## 形状与代价

    | 量 | 值 |
    | --- | --- |
    | 沿射线步长 | 20 m（清除半径）|
    | 横向列数 | $\\max\\big(0,\\ \\lceil (d\\sin1° - LIM)/step \\rceil\\big)$ 列每侧，$LIM=\\sqrt{20^2-(step/2)^2}$ ⟹ **$d\\le992$ 只铺 1 列**（2026-09-13 修正，原式过宽 3×）|
    | 射线长度 | $\\le1500$ m（$r_{\\text{eff}}$ 上界）|
    | 上界点数 | ≈ 75 步 × 至多 5 列 ≈ **300 点** |
    | 每点代价 | 移动 20 m(4 s) + `/clear`(5 s 中 / 3 s 空) ≈ 7–9 s |

    ⚠ **只在「本来会漏掉一个源」时才触发** —— 漏源是**硬约束**的失败，
    比多花一两千秒虚拟时间严重得多。`budget` 是防止病态输入的硬闸。

    走蛇形（隔行反向），避免每步横跳。
    """
    ux, uy = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg))
    nx, ny = -uy, ux
    # ★★ 2026-09-13【锥宽公式修正】—— 原式 `min(40, d·tan1°+20)` 在近距离**过宽**。
    #
    # ## 正确推导
    #
    # 源的**横向**不确定度 = $L = d\\sin1°$（真源在射线上、方位误差 ±1°；
    # 沿射线走到参数 $d$ 处，与真源方向的垂距就是 $d\\sin\\delta$）。
    # 轴向以 `step` 步进，最近一根探针距理想点 $\\le step/2$。要有点落进清除半径：
    #
    #     (L − k·step)² + (step/2)² ≤ 20²   ⟹   |L − k·step| ≤ LIM,
    #     LIM = sqrt(20² − (step/2)²) = sqrt(400 − 100) = **17.3 m**（step=20）
    #
    # $L$ 的最大值（$d\\le1500$）是 $1500\\sin1°=26.2$ ⟹ **$k$ 取到 1 就已足够**：
    #
    # | d | L=d·sin1° | 需要列数 | 原式列数 | 浪费 |
    # | --- | --- | --- | --- | --- |
    # | ≤992 | ≤17.3 | **1**（k=0 即够） | 3 | **3×** |
    # | ≥1146 | >20 | **3**（k=0,±1） | 5 | 1.7× |
    #
    # 实测代价（最慢局频道 17，官方）：`/clear` 462 次/局 vs 生产 46 次。
    # **保证不变（推导是充要的），只是不再白铺。**
    sin1 = math.sin(math.radians(1.0))
    LIM = math.sqrt(max(0.0, 20.0 ** 2 - (step / 2.0) ** 2))
    RAY_STATS["trigger"] += 1
    d = 2.0 * step
    row = 0
    while d <= r_max:
        L = d * sin1                                  # 横向不确定度
        m = max(0, math.ceil((L - LIM) / step))       # 覆盖它所需的最小列数
        ks = list(range(-m, m + 1))
        if row % 2:
            ks.reverse()
        for k in ks:
            budget -= 1
            if budget <= 0:
                return False
            x, y = Q.clamp_to_arena(px + ux * d + nx * (k * step),
                                    py + uy * d + ny * (k * step))
            robot.move_to(x, y)
            if robot.clear(ch) == "success":
                kb.cleared.add(ch)
                kb.refresh_empty()
                RAY_STATS["hit"] += 1
                return True
        d += step
        row += 1
    return False


def bench(seed: int, n: int, dir_ratio: float, variant: str,
          step: float = 300.0, k_ring: int = 180):
    js = OS.make_jammers(n, seed=seed, directional_ratio=dir_ratio)
    sim = LocalSim(js)
    rb = Robot(sim)
    kb = Q.Knowledge(dir_aware=(variant == "q4"))
    if variant == "q4":
        _, n_scan = run_q4(rb, kb, step=step, k_ring=k_ring)
    else:
        import q3_shared_measure as SM
        SM.run_shared(rb, kb)
        n_scan = 0
    mv, px, py = 0.0, 0.0, 0.0
    for r in rb.log:
        mv += math.hypot(r[1] - px, r[2] - py)
        px, py = r[1], r[2]
    nmeas = sum(1 for r in rb.log if r[0] == "measure")
    n_dir = sum(1 for j in js if j.directional)
    return sim.t, sim.cleared_count(), n, mv, nmeas, n_scan, n_dir


def main(argv=None) -> int:
    enable_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--dir", type=float, default=0.5, help="定向源比例")
    ap.add_argument("--step", type=float, default=800.0)
    ap.add_argument("--kring", type=int, default=8)
    ap.add_argument("--ringeps", type=float, default=200.0)
    a = ap.parse_args(argv)

    n = a.n
    print("=" * 104)
    print(f"表 Q4-1  定向源失效模式：Q3 策略 vs Q4 策略"
          f"（n={n}，定向比例 {a.dir}，{a.seeds} 局）")
    print("=" * 104, flush=True)
    print(f"  {'seed':>5}{'定向':>5} | {'Q3 清除':>9}{'Q3 用时':>9}{'Q3 s/个':>9}"
          f" | {'Q4 清除':>9}{'Q4 用时':>9}{'Q4 s/个':>9}{'扫描点':>7}")
    print("  " + "-" * 92)
    A, B = [], []
    ok3 = ok4 = 0
    for s in range(1, a.seeds + 1):
        t1, c1, nt, m1, k1, _, nd = bench(s, n, a.dir, "q3")
        t2, c2, _, m2, k2, ns, _ = bench(s, n, a.dir, "q4", a.step, a.kring)
        A.append(t1)
        B.append(t2)
        ok3 += (c1 == nt)
        ok4 += (c2 == nt)
        print(f"  {s:>5}{nd:>5} | {c1:>4}/{nt:<4}{t1:>9.0f}{t1 / n:>9.1f}"
              f" | {c2:>4}/{nt:<4}{t2:>9.0f}{t2 / n:>9.1f}{ns:>7}")
    m = a.seeds
    print("  " + "-" * 92)
    print(f"  {'合计':>5}{'':>5} | {ok3:>4}/{m:<4}{st.mean(A):>9.0f}{st.mean(A) / n:>9.1f}"
          f" | {ok4:>4}/{m:<4}{st.mean(B):>9.0f}{st.mean(B) / n:>9.1f}")
    print()
    print(f"  ⟹ **全清率：Q3 策略 {ok3}/{m}    Q4 策略 {ok4}/{m}**")
    print(f"     Q3 策略在定向源下**漏清**，正是因为它把「扇区外的 no_signal」"
          f"当成了「该频道为空」的证据。")
    print(f"     Q4 用 `covers_arena_dir`（距离覆盖 ∧ 角度覆盖）堵住这个缺口。")
    print()
    print(f"  Q4 布局：方格 {a.step:.0f} m + 场地边界环 {a.kring} 点"
          f"（ρ={ARENA:.0f}）")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
