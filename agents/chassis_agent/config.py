"""ChassisAgent 的全部常量与配置。

────────────────────────────────────────────────────────────────────────
1. 普通配置（robot_id / 速度 / LLM 地址等）——直接改本文件的值即可。

2. 敏感信息（API Key 等）——**用环境变量**，不要写死在本文件里。
   LLM_API_KEY 从环境变量 RABO_LLM_KEY 读取（平台注入），勿写入明文。

3. 机器人配置依赖平台运行环境注入的变量：RABO_API_HOST / RDTP_USER_SCENE /
   RABO_ID + 沙箱 token —— rabo_robocap 的工厂构造时会自动读取，无需在此配置。
────────────────────────────────────────────────────────────────────────
"""

import os

# ── 机器人 ────────────────────────────────────────────────────────────
# 场景里放置的全向底盘（Agilex Ranger Mini 3, 四轮四转）的模型 ID。
ROBOT_ID = "rd430886d637381f3afac772a1e937b11"

# 运行模式: 'sim'（平台仿真场景，SDK 自建服务端节点）/ 'real'（真实硬件）。
MODE = "sim"

# ── 移动参数 ──────────────────────────────────────────────────────────
DEFAULT_LINEAR_SPEED = 0.5      # m/s, cruise 工具默认线速度
DEFAULT_ANGULAR_SPEED = 0.6     # rad/s, rotate 工具默认角速度

# ── LLM 配置（rabo 平台自带大模型 API, OpenAI 兼容, 支持 function calling）──
LLM_BASE_URL = "https://ai.rabo.cc/p/qwen"
LLM_MODEL = "qwen3.6-flash"     # 复杂推理/代码可换 qwen3.6-plus
LLM_API_KEY = os.getenv("RABO_LLM_KEY", "")

# ── 无输入源方案：启动时发给 agent 的演示指令 ─────────────────────────
DEMO_TEXT = "演示一下你的移动能力，走你的演示流程"
