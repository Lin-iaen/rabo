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

# LLM 采样温度: 越高越有创意/越不稳定。0.7 = 有发挥空间又不至于太飘。
# 想看更激进/更保守的发挥, 改这里即可 (0.0~1.0)。
TEMPERATURE = 0.7

# ── LLM 配置（rabo 平台自带大模型 API, OpenAI 兼容, 支持 function calling）──
LLM_BASE_URL = "https://ai.rabo.cc/p/qwen"
LLM_MODEL = "qwen3.6-flash"     # 复杂推理/代码可换 qwen3.6-plus
LLM_API_KEY = os.getenv("RABO_LLM_KEY", "")

# ── 无输入源方案：启动时发给 agent 的演示指令 ─────────────────────────
# 开放式指令: 让 LLM 自由发挥 (阶段 A)。当下面 REMOTE_ID 配置后, 此指令不再使用。
DEMO_TEXT = (
    "自由发挥：不要调用 run_demo_sequence，自己设计一套有创意的移动动作并执行"
    "（例如平移、斜行、原地自转、短暂停顿的组合，也可以连续走一段再变换方向）。"
    "中途可以调用 get_status 观察位姿，最后汇报你的设计思路和结果。"
)

# ── 平台控制面板（H5 遥控面板, 作为 agent 的输入/输出通道, 阶段 B）──
# 留空 = 无输入源, run() 走上面的 DEMO_TEXT 一次性演示;
# 填上 = run() 改为常驻监听面板对话, 收到文字 → agent 处理 → 回复回传到面板。
# 在场景里创建 H5 遥控面板, 面板 ID 填 REMOTE_ID, 面板内 chat 控件 ID 填 CHAT_CONTROL_ID。
# RemoteControl 的 WebSocket 地址 RABO_REMOTE_WS_URL 由平台运行环境自动注入, 无需配置。
REMOTE_ID = "cd49223f86e236c29f9f5ea6876e39307"
CHAT_CONTROL_ID = "chat-v1"

# ── Gazebo world 名 (复位/传送工具用) ────────────────────────────────
# 留空 = 运行时自动从 ROS 服务发现 (找 /world/<name>/set_pose);
# 若自动发现失败, 填这里 (在仿真场景的 Gazebo world 名)。
WORLD = ""

# ── 相机 (场景里给小车加装的 RGBD 相机, 本次只用 RGB 图像) ──────────
# 图像话题: 在场景的传感器面板里能看到, 形如 <相机名>/image。
CAMERA_IMAGE_TOPIC = "rd43088_tp_rgbd_d724d13b23/image"

# ── 赛道跟随 (寻迹) 参数 ─────────────────────────────────────────────
TRACK_STEP_M = 0.25             # 闭环每步直行距离 (米), 越短越跟得紧/越平滑
TRACK_MAX_STEER_DEG = 30        # 单步最大转向 (度)
TRACK_LOST_STEPS = 5            # 连续多少步看不到路面判定为偏出赛道
TRACK_ROTATE_TOL_DEG = 2.0      # 旋转到位容差 (度, 里程计闭环)
TRACK_STEER_SMOOTH = 0.6        # 转向 EMA 平滑系数 (0~1, 越大越跟手/越抖)
TRACK_HEADING_DEADBAND_DEG = 6  # 航向死区 (度): 偏角低于此不转车身, 只直行
TRACK_ROTATE_KP = 0.03          # 旋转比例增益 (rad/s 每度误差)
TRACK_MAX_ANGULAR_SPEED = 0.8   # 旋转最大角速度 (rad/s)

# ── 赛道感知校准 (HSV 阈值) ──────────────────────────────────────────
# 场景: 灰黑柏油路面 + 白色边界线 + 黑白相间路缘 + 绿色草坪背景。
# 用 camera_probe 工具看当前画面各颜色占比, 再回来调这里。
CAMERA_DEBUG_DIR = "/tmp/chassis_cam_debug"

VISION = {
    "steer_band": (0.10, 0.55),   # 转向信号带 (y 归一化): 灰质心主转向, 含弯道延续
    "bottom_y": 0.60,             # 底部段起点: "是否在路上" + "是否贴近边线"
    "grass_h_range": (35, 85),    # 草坪色相区间
    "grass_s_min": 60,
    "grass_v_min": 60,
    "asphalt_s_max": 60,          # 沥青: 饱和度上限
    "asphalt_v_min": 40,          # 沥青: 明度下限
    "asphalt_v_max": 200,         # 沥青: 明度上限 (排除白线)
    "white_s_max": 60,            # 白线: 饱和度上限
    "white_v_min": 190,           # 白线: 明度下限
    "road_visible_fraction": 0.10, # 底部灰占比下限 → 否则判偏出
    "edge_white_frac": 0.05,      # 底部白线占比高于此才考虑边线纠正 (防误报)
    "edge_bias_deg": 15,          # 贴近边线时向赛道内的纠正 (度, 已减弱)
    "max_steer_deg": TRACK_MAX_STEER_DEG,
}
