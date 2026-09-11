"""
零成本连通性探针 —— 只发 GET 请求，永不改变模拟器状态。

为什么安全：官方 4 条指令全部是 POST。GET 到 /enter 不是合法动作，
最多返回 405/404，不会 /enter、不会推进虚拟时间、不会占用测试机会。

用来区分两种状态：
    接口未开放（还没点「开始测试」/ 倒计时中 / 已结束）
        → TCP 连得上，但服务端立即关闭连接，拿不到任何 HTTP 响应
    接口已开放（可以发指令了）
        → 返回一个 HTTP 状态码

用法：
    python probe.py                    # 查默认地址 http://127.0.0.1:2026
    python probe.py --url http://127.0.0.1:2027
    python probe.py --watch            # 循环等，接口一开就退出（用于卡倒计时）
"""

from __future__ import annotations

import argparse
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import BASE_URL


def probe_once(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    """返回 (接口是否开放, 说明文字)。"""
    req = Request(url + "/enter", method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return True, f"HTTP {resp.status}"
    except HTTPError as e:
        # 405 / 404 等：服务端在正常应答，说明接口已开放
        return True, f"HTTP {e.code}"
    except URLError as e:
        reason = str(e.reason)
        if "refused" in reason.lower():
            return False, "连接被拒绝：模拟器没在运行，或端口不对"
        return False, f"连上但无响应（接口未开放）: {reason}"
    except (ConnectionError, TimeoutError, OSError) as e:
        return False, f"连上但无响应（接口未开放）: {type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=BASE_URL, help=f"默认 {BASE_URL}")
    ap.add_argument("--watch", action="store_true", help="循环等待直到接口开放")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--timeout-s", type=float, default=120.0, help="--watch 的最长等待秒数")
    args = ap.parse_args()

    print(f"探针目标：{args.url}")
    if not args.watch:
        ok, msg = probe_once(args.url)
        print(("✅ 接口已开放  " if ok else "⛔ 接口未开放  ") + msg)
        return 0 if ok else 1

    print(f"等待接口开放（最多 {args.timeout_s:.0f}s，每 {args.interval:.0f}s 查一次）...")
    t0 = time.time()
    while time.time() - t0 < args.timeout_s:
        ok, msg = probe_once(args.url)
        el = time.time() - t0
        if ok:
            print(f"  ✅ [{el:6.1f}s] 接口已开放  {msg}")
            print("\n可以发指令了。现在运行策略脚本。")
            return 0
        print(f"  ⛔ [{el:6.1f}s] {msg}")
        time.sleep(args.interval)
    print("\n超时：接口始终未开放。确认已在模拟器里点「开始测试」并等完倒计时。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
