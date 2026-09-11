"""
B题离线模拟器 —— 本地复现官方模拟器的物理与计时规则。

用途：
  1. 开发/调试搜索策略，**不消耗模拟器的演练与正式测试机会**
  2. 可用性测试的靶子（先在这里验证代码，再上真模拟器）
  3. 批量 Monte Carlo 实验，为论文提供统计结果

启动：  python offline_sim.py            # 默认 127.0.0.1:2026
       python offline_sim.py --port 2027 --jammers 14 --seed 42

与官方模拟器的差异（仅用于开发，正式数据必须来自官方模拟器）：
  - 不校验登录、不做服务器时间校验、无 25 分钟窗口
  - 不生成加密日志
  - 提供 /debug/truth 接口查看真值（官方没有，仅本地开发用）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARENA_RADIUS = 1800.0
NEAR_RADIUS = 5.0
CLEAR_RADIUS = 20.0
SPEED = 5.0
T_MEASURE = 5.0
T_SWITCH = 1.0
T_CLEAR_FOUND = 5.0
T_CLEAR_MISS = 3.0
BEARING_ERR_DEG = 1.0
R_EFF_MIN, R_EFF_MAX = 1000.0, 1500.0
MAX_VIRTUAL_S = 360000.0
MAX_REAL_S = 1200.0


@dataclass
class Jammer:
    channel: int
    x: float
    y: float
    r_eff: float
    directional: bool = False
    direction_deg: float | None = None
    cleared: bool = False

    def covers(self, px: float, py: float) -> bool:
        """位置是否在信号覆盖角度范围内（全向恒为 True）。"""
        if not self.directional or self.direction_deg is None:
            return True
        # 定向：定向方向两侧各 90°，共 180°
        bearing = math.degrees(math.atan2(py - self.y, px - self.x)) % 360
        diff = abs((bearing - self.direction_deg + 180) % 360 - 180)
        return diff <= 90.0

    def true_bearing(self, px: float, py: float) -> float:
        """检测点到干扰源的真实方位角 [0,360)。"""
        return math.degrees(math.atan2(self.y - py, self.x - px)) % 360

    def bearing_error(self, px: float, py: float) -> float:
        """±1° 误差。同一地点重复检测误差固定（电磁环境固定）—— 官方规则。"""
        key = f"{round(px, 3)}_{round(py, 3)}_{self.channel}".encode()
        h = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        # 映射到 [-1, 1]
        return (h / 2**63 - 1.0) * BEARING_ERR_DEG


@dataclass
class State:
    jammers: list[Jammer] = field(default_factory=list)
    entered: bool = False
    exited: bool = False
    x: float = 0.0
    y: float = 0.0
    channel: int = 1
    virtual_time_s: float = 0.0
    t0: float = 0.0
    seen_ids: dict[str, tuple[str, dict]] = field(default_factory=dict)

    def move_time(self, nx: float, ny: float) -> float:
        return math.hypot(nx - self.x, ny - self.y) / SPEED

    def to_truth(self) -> dict:
        return {
            "jammer_count": len(self.jammers),
            "omnidirectional": sum(1 for j in self.jammers if not j.directional),
            "directional": sum(1 for j in self.jammers if j.directional),
            "cleared": sum(1 for j in self.jammers if j.cleared),
            "jammers": [asdict(j) for j in self.jammers],
            "virtual_time_s": self.virtual_time_s,
        }


def make_jammers(n: int, seed: int | None, directional_ratio: float) -> list[Jammer]:
    rng = random.Random(seed)
    channels = rng.sample(range(1, 21), n)
    out = []
    for ch in channels:
        # 在半径 1800 的圆内均匀采样
        r = ARENA_RADIUS * math.sqrt(rng.random())
        th = rng.uniform(0, 2 * math.pi)
        directional = rng.random() < directional_ratio
        out.append(Jammer(
            channel=ch,
            x=r * math.cos(th),
            y=r * math.sin(th),
            r_eff=rng.uniform(R_EFF_MIN, R_EFF_MAX),
            directional=directional,
            direction_deg=rng.uniform(0, 360) if directional else None,
        ))
    return out


class Handler(BaseHTTPRequestHandler):
    state: State = None            # 由 make_server 注入
    lock = threading.Lock()

    def log_message(self, *a):     # 静音，避免刷屏
        pass

    # ── 工具 ────────────────────────────────────────────────
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _common(self) -> dict:
        return {
            "accepted": False,
            "real_timestamp_ms": int(time.time() * 1000),
            "virtual_time_s": 0,        # accepted=false 时官方恒为 0
        }

    def _read_body(self) -> dict | None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def _validate(self, body: dict, allow_extra: set[str]) -> str | None:
        """返回错误消息，None 表示通过。"""
        if body is None:
            return "invalid json"
        if body.get("arena_id") != "default":
            return "arena_id mismatch"
        for f in ("robot_id", "request_id"):
            v = body.get(f)
            if not isinstance(v, str) or not v:
                return f"{f} invalid"
        extra = set(body) - {"arena_id", "robot_id", "request_id"} - allow_extra
        if extra:
            return f"unknown fields: {sorted(extra)}"   # 官方：未知字段 → accepted=false
        return None

    def do_POST(self) -> None:      # noqa: N802
        path = self.path
        if path not in ("/enter", "/measure", "/clear", "/exit", "/debug/truth"):
            return self._send(404, {"error": "not found"})
        body = self._read_body()
        with self.lock:
            self._dispatch(path, body)

    def _dispatch(self, path: str, body: dict) -> None:
        st = self.state

        # 本地调试接口
        if path == "/debug/truth":
            return self._send(200, st.to_truth())

        allow = {"position", "channel"} if path in ("/measure", "/clear") else set()
        err = self._validate(body, allow)
        if err:
            if err.startswith("unknown"):
                return self._send(200, {**self._common(), "message": err})
            return self._send(400, {**self._common(), "message": err})

        rid = body["request_id"]
        # 幂等：同 ID 同内容返回首次响应；同 ID 不同内容 → 409
        if rid in st.seen_ids:
            old_path, old_body = st.seen_ids[rid]
            if old_path == path and old_body == body:
                return self._send(200, {"accepted": True, "real_timestamp_ms":
                                        int(time.time() * 1000),
                                        "virtual_time_s": st.virtual_time_s,
                                        "_replayed": True})
            return self._send(409, {"error": "request_id reused with different content"})

        if path == "/enter":
            if st.entered:
                return self._send(200, {**self._common(), "message": "already entered"})
            st.entered, st.exited = True, False
            st.t0 = time.time()
            st.seen_ids[rid] = (path, body)
            return self._send(200, {
                "accepted": True, "real_timestamp_ms": int(time.time() * 1000),
                "virtual_time_s": st.virtual_time_s,
                "max_virtual_duration_s": MAX_VIRTUAL_S,
                "max_real_duration_s": MAX_REAL_S,
                "remaining_real_duration_s": MAX_REAL_S,
            })

        if not st.entered or st.exited:
            return self._send(200, {**self._common(), "message": "not entered"})

        pos = body.get("position")
        if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
            return self._send(400, {**self._common(), "message": "missing position"})
        try:
            nx, ny = float(pos["x"]), float(pos["y"])
        except Exception:
            return self._send(400, {**self._common(), "message": "bad position"})
        if not (math.isfinite(nx) and math.isfinite(ny)) or max(abs(nx), abs(ny)) > 2e6:
            return self._send(400, {**self._common(), "message": "position out of range"})
        ch = body.get("channel")
        if not isinstance(ch, int) or not (1 <= ch <= 20):
            return self._send(400, {**self._common(), "message": "bad channel"})

        move = st.move_time(nx, ny)
        st.x, st.y = nx, ny
        st.seen_ids[rid] = (path, body)

        if path == "/measure":
            switch = T_SWITCH if ch != st.channel else 0.0
            st.channel = ch
            st.virtual_time_s += move + switch + T_MEASURE
            j = next((j for j in st.jammers if j.channel == ch and not j.cleared), None)
            resp = {"accepted": True, "real_timestamp_ms": int(time.time() * 1000),
                    "virtual_time_s": round(st.virtual_time_s, 6)}
            if j is None:
                resp["measure_result"] = "no_signal"
            else:
                d = math.hypot(j.x - nx, j.y - ny)
                if d > j.r_eff or not j.covers(nx, ny):
                    resp["measure_result"] = "no_signal"
                elif d <= NEAR_RADIUS:
                    resp["measure_result"] = "near"
                else:
                    resp["measure_result"] = "direction"
                    resp["svd_deg"] = round(
                        (j.true_bearing(nx, ny) + j.bearing_error(nx, ny)) % 360, 2)
            return self._send(200, resp)

        # /clear
        st.virtual_time_s += move
        j = next((j for j in st.jammers if j.channel == ch and not j.cleared), None)
        if j is not None and math.hypot(j.x - nx, j.y - ny) <= CLEAR_RADIUS:
            j.cleared = True
            st.virtual_time_s += T_CLEAR_FOUND
            result = "success"
        else:
            st.virtual_time_s += T_CLEAR_MISS
            result = "no_target_in_range"
        return self._send(200, {
            "accepted": True, "real_timestamp_ms": int(time.time() * 1000),
            "virtual_time_s": round(st.virtual_time_s, 6),
            "clear_result": result,
        })

    def do_GET(self) -> None:       # noqa: N802
        if self.path == "/debug/truth":
            with self.lock:
                return self._send(200, self.state.to_truth())
        return self._send(405, {"error": "use POST"})


def make_server(port: int, state: State) -> ThreadingHTTPServer:
    Handler.state = state
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return srv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=2026)
    ap.add_argument("--jammers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--directional-ratio", type=float, default=0.0,
                    help="定向干扰源比例（问题3用0，问题4用0.3~0.5）")
    args = ap.parse_args()

    state = State(jammers=make_jammers(args.jammers, args.seed, args.directional_ratio))
    srv = make_server(args.port, state)
    print(f"离线模拟器已启动 http://127.0.0.1:{args.port}")
    print(f"  干扰源 {args.jammers} 个，定向比例 {args.directional_ratio}，seed={args.seed}")
    print(f"  真值接口（仅本地）: GET http://127.0.0.1:{args.port}/debug/truth")
    print("  Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
