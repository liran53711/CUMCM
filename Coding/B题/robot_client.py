"""
B题模拟器通信客户端。

只负责「可靠通信」，不含搜索策略。策略写在单独文件里，调用本客户端。

参考 skill：.claude/skills/b-simulator/  （协议规范与策略笔记）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import BASE_URL, HTTP_TIMEOUT, MAX_RETRY, ROBOT_ID, SPEED


@dataclass
class Action:
    """一次动作的完整记录，用于日志、复盘、论文写作。"""
    idx: int
    path: str
    x: float
    y: float
    channel: int
    accepted: bool
    virtual_time_s: float
    virtual_delta_s: float = 0.0     # 本步推进的虚拟时间
    result: str = ""                  # measure_result / clear_result / exit_reason
    svd_deg: float | None = None
    http_status: int = 0
    note: str = ""

    @property
    def is_signal(self) -> bool:
        return self.result == "direction"

    @property
    def is_near(self) -> bool:
        return self.result == "near"

    @property
    def is_cleared(self) -> bool:
        return self.result == "success"


@dataclass
class RobotClient:
    robot_id: str = ROBOT_ID
    base_url: str = BASE_URL

    log: list[Action] = field(default_factory=list)
    # 最近一次 accepted=true 的虚拟时刻（accepted=false 时返回的 0 不可用）
    virtual_time_s: float = 0.0
    # 测向机当前频道：只有成功的 /measure 会更新；/clear 不影响
    current_channel: int = 1
    # 上一次合法动作的位置，用于本地估算移动耗时
    last_x: float = 0.0
    last_y: float = 0.0
    entered: bool = False
    _seq: int = 0

    # ── 内部 ────────────────────────────────────────────────
    def _next_id(self, tag: str) -> str:
        self._seq += 1
        return f"{tag}-{self._seq}"

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
        """发送请求，含网络层重试。返回 (响应JSON, HTTP状态码)。

        重试时严格复用**完全相同**的 payload（含同一 request_id）——协议要求。
        返回 (None, 0) 表示连不上（接口未开放 / 已结束 / 网络中断）。
        """
        body = json.dumps(payload).encode("utf-8")
        last_err: Exception | None = None

        for attempt in range(MAX_RETRY):
            req = Request(
                self.base_url + path,
                data=body,                                   # 复用同一 body
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8")), resp.status
            except HTTPError as e:
                # 4xx/5xx 通常仍带 JSON body，解析它便于定位
                try:
                    return json.loads(e.read().decode("utf-8")), e.code
                except Exception:
                    last_err = e
                    if e.code in (400, 404, 405, 409, 413, 415):
                        return None, e.code     # 客户端错误，重试无意义
            except (URLError, ConnectionError, TimeoutError, OSError) as e:
                last_err = e
            time.sleep(0.3 * (attempt + 1))

        print(f"[warn] {path} 连接失败（重试 {MAX_RETRY} 次）: {last_err}")
        return None, 0

    def _record(self, path: str, x: float, y: float, channel: int,
                resp: dict[str, Any] | None, status: int) -> Action:
        accepted = bool(resp and resp.get("accepted") is True)
        prev_vt = self.virtual_time_s
        vt = prev_vt
        if accepted:
            vt = float(resp.get("virtual_time_s", prev_vt))
            self.virtual_time_s = vt

        act = Action(
            idx=self._seq, path=path, x=x, y=y, channel=channel,
            accepted=accepted, virtual_time_s=vt,
            virtual_delta_s=vt - prev_vt, http_status=status,
        )
        if resp:
            if "measure_result" in resp:
                act.result = resp["measure_result"]
                act.svd_deg = resp.get("svd_deg")
            elif "clear_result" in resp:
                act.result = resp["clear_result"]
            elif "exit_reason" in resp:
                act.result = resp["exit_reason"]
            if accepted is False:
                act.note = str(resp.get("message", resp.get("error", "accepted=false")))
        elif status:
            act.note = f"HTTP {status}"
        else:
            act.note = "连接失败（接口未开放或已结束）"

        self.log.append(act)
        return act

    # ── 四条指令 ────────────────────────────────────────────
    def enter(self) -> dict[str, Any] | None:
        payload = {"arena_id": "default", "robot_id": self.robot_id,
                   "request_id": self._next_id("enter")}
        resp, _ = self._post("/enter", payload)
        if resp and resp.get("accepted") is True:
            self.virtual_time_s = float(resp.get("virtual_time_s", 0.0))
            self.entered = True
        return resp

    def measure(self, x: float, y: float, channel: int) -> Action:
        payload = {"arena_id": "default", "robot_id": self.robot_id,
                   "request_id": self._next_id("measure"),
                   "position": {"x": float(x), "y": float(y)},
                   "channel": int(channel)}
        resp, status = self._post("/measure", payload)
        act = self._record("/measure", x, y, channel, resp, status)
        if act.accepted:
            self.current_channel = int(channel)
            self.last_x, self.last_y = x, y
        return act

    def clear(self, x: float, y: float, channel: int) -> Action:
        payload = {"arena_id": "default", "robot_id": self.robot_id,
                   "request_id": self._next_id("clear"),
                   "position": {"x": float(x), "y": float(y)},
                   "channel": int(channel)}
        resp, status = self._post("/clear", payload)
        act = self._record("/clear", x, y, channel, resp, status)
        if act.accepted:
            self.last_x, self.last_y = x, y      # /clear 不改频道
        return act

    def exit(self) -> dict[str, Any] | None:
        payload = {"arena_id": "default", "robot_id": self.robot_id,
                   "request_id": self._next_id("exit")}
        resp, _ = self._post("/exit", payload)
        self._record("/exit", self.last_x, self.last_y, self.current_channel, resp, 0)
        self.entered = False
        return resp

    # ── 辅助 ────────────────────────────────────────────────
    def remaining_real_s(self, enter_resp: dict[str, Any]) -> float:
        """本局实际可用现实秒数。**不要硬编码 1200**，晚进就少。"""
        return float(enter_resp.get("remaining_real_duration_s", 0))

    def move_time(self, x: float, y: float) -> float:
        """本地估算移动耗时（虚拟秒），供策略打分用。"""
        d = ((x - self.last_x) ** 2 + (y - self.last_y) ** 2) ** 0.5
        return d / SPEED

    def dump_log(self, path: str) -> None:
        """导出动作日志。题目要求机器狗程序自行记录，模拟器不提供。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "robot_id": self.robot_id,
                "final_virtual_time_s": self.virtual_time_s,
                "action_count": len(self.log),
                "actions": [a.__dict__ for a in self.log],
            }, f, ensure_ascii=False, indent=2)
