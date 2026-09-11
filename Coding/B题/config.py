"""B题全局配置。改这里，不要改 robot_client.py 的逻辑。

可用环境变量临时覆盖（不改文件）：
    B_SIM_URL  接口地址，例如 http://127.0.0.1:2027
    B_ROBOT_ID 参赛队号
"""

import os

# ── 参赛信息 ────────────────────────────────────────────────
# 必须逐字节等于模拟器当前登录的参赛队号，否则所有请求返回 accepted=false
ROBOT_ID = os.environ.get("B_ROBOT_ID", "202610038113")

# ── 接口地址 ────────────────────────────────────────────────
# 官方模拟器默认 2026。离线模拟器建议用 2027 避免冲突。
# 官方模拟器退出后，2026 可能被 Windows 短暂保留（WinError 10013），
# 此时直接用 B_SIM_URL 换端口，不要去改模拟器设置。
BASE_URL = os.environ.get("B_SIM_URL", "http://127.0.0.1:2026")

# ── 物理常数（来自题目附录，不要改）────────────────────────
SPEED = 5.0          # 机器狗移动速度 m/s
T_MEASURE = 5.0      # 检测动作耗时 s
T_SWITCH = 1.0       # 频道切换耗时 s（仅 /measure 触发）
T_CLEAR_FOUND = 5.0  # 精确定位3s + 清除2s
T_CLEAR_MISS = 3.0   # 仅精确定位

ARENA_RADIUS = 1800.0   # 目标区域半径 m
NEAR_RADIUS = 5.0       # 近距离阈值 m（≤此值且在覆盖角内 → near）
CLEAR_RADIUS = 20.0     # 清除半径 m
CHANNELS = 20           # 频道 1..20
MAX_VIRTUAL_S = 360000  # 虚拟限时 100h

# ── 网络 ────────────────────────────────────────────────────
HTTP_TIMEOUT = 5.0   # 单请求超时 s
MAX_RETRY = 5        # 网络层重试次数（重试复用同一 request_id）

# ── 路径 ────────────────────────────────────────────────

WORKDIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKDIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
