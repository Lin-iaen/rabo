"""ChassisAgent 的工具集: schema + handler + 调度。

照模板约定: schema / handler / 调度都在本文件, 加新工具三步走:
  1. TOOLS 加一项 OpenAI function schema
  2. 写 _xxx(agent, args) -> str
  3. _HANDLERS 加一行映射

API 参考 (rabo_robocap.AgilexRangeMini3, 全向底盘):
  move_distance(direction_rad, distance_m, blocking=True) -> bool  # odom 闭环阻塞
  rotate(angular_velocity_rad_s, blocking=...)                     # 连续自转, 需计时+stop
  set_velocity(direction_rad, speed_m_s, blocking=...)             # 连续平移
  get_odometry() -> (x, y, theta_rad)
  get_velocity() -> (vx, vy, omega_rad_s)
  stop() / shutdown()

方向约定: 相对车头, 0=正前, π/2=车头左, -π/2=车头右 (逆时针为正)。单位 rad。
对外暴露给 LLM 的统一用「度 + 米」, handler 内转弧度。
"""

import math
import time

from . import config


# ══════════════════════════════════════════════════════════════════════
#  工具 schema 列表 (OpenAI function calling 格式)
# ══════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "朝指定方向平移指定距离（odom 闭环，阻塞到到位后返回）。"
                           "方向相对车头，逆时针为正：0=正前，90=正左，-90=正右，45=左前方。"
                           "角度用度，距离用米。",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction_deg": {"type": "number", "description": "平移方向，相对车头，度（0=正前，90=正左）"},
                    "distance_m": {"type": "number", "description": "平移距离，米（正值）"},
                },
                "required": ["direction_deg", "distance_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate",
            "description": "原地自转指定角度。正=逆时针，负=顺时针。角度用度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle_deg": {"type": "number", "description": "自转角度，度（正=逆时针，负=顺时针）"},
                },
                "required": ["angle_deg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cruise",
            "description": "朝指定方向以指定速度持续平移指定秒数后自动停止。"
                           "方向含义同 move（相对车头，度）。用于让机器人持续走一段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction_deg": {"type": "number", "description": "平移方向，相对车头，度"},
                    "speed_mps": {"type": "number", "description": "平移速度，m/s"},
                    "duration_s": {"type": "number", "description": "持续时长，秒"},
                },
                "required": ["direction_deg", "duration_s"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "查询底盘当前位姿 (x, y, theta) 与速度 (vx, vy, omega)。"
                           "位姿来自里程计，theta 单位为弧度。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "立即停止机器人所有运动。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_demo_sequence",
            "description": "完整演示全向底盘能力：前进 2 米 → 斜行 45° 平移 1 米 → "
                           "原地逆时针自转 90° → 回到起点 → 转回原朝向。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ══════════════════════════════════════════════════════════════════════
#  handler 实现 (_xxx(agent, args) -> str)
# ══════════════════════════════════════════════════════════════════════

def _do_move(agent, direction_deg: float, distance_m: float):
    """平移一段距离（odom 闭环，阻塞到到位），返回 (成功, 起点odom, 终点odom)。"""
    start = agent.base.get_odometry()
    ok = agent.base.move_distance(math.radians(direction_deg), distance_m)
    end = agent.base.get_odometry()
    return ok, start, end


def _do_rotate(agent, angle_deg: float):
    """原地自转指定角度。SDK 的 rotate 是连续自转，采用「设转速→计时→停止」近似。"""
    rad = math.radians(angle_deg)
    if abs(rad) < 1e-6:
        return angle_deg, agent.base.get_odometry()[2]
    omega = math.copysign(config.DEFAULT_ANGULAR_SPEED, rad)
    agent.base.rotate(omega, blocking=False)
    time.sleep(abs(rad) / abs(omega))
    agent.base.stop()
    return angle_deg, agent.base.get_odometry()[2]


def _move(agent, args):
    direction_deg = args["direction_deg"]
    distance_m = args["distance_m"]
    if distance_m <= 0:
        return f"参数错误: distance_m({distance_m}) 必须为正"
    ok, start, end = _do_move(agent, direction_deg, distance_m)
    return (
        f"朝 {direction_deg}° 平移 {distance_m}m {'成功' if ok else '失败'}。"
        f"起点=({start[0]:.2f}, {start[1]:.2f})，终点=({end[0]:.2f}, {end[1]:.2f})。"
    )


def _rotate(agent, args):
    angle_deg = args["angle_deg"]
    requested, actual_theta = _do_rotate(agent, angle_deg)
    return f"自转 {requested}° 完成，当前朝向 {math.degrees(actual_theta):.2f}°"


def _cruise(agent, args):
    direction_deg = args["direction_deg"]
    duration_s = args["duration_s"]
    speed = args.get("speed_mps", config.DEFAULT_LINEAR_SPEED)
    agent.base.set_velocity(math.radians(direction_deg), speed, blocking=False)
    time.sleep(duration_s)
    agent.base.stop()
    x, y, theta = agent.base.get_odometry()
    return (
        f"以 {speed} m/s 朝 {direction_deg}° 平移 {duration_s}s 后停止。"
        f"当前位姿=({x:.2f}, {y:.2f}, {math.degrees(theta):.2f}°)。"
    )


def _get_status(agent, args):
    x, y, theta = agent.base.get_odometry()
    vx, vy, omega = agent.base.get_velocity()
    return (
        f"位姿: x={x:.2f} m, y={y:.2f} m, theta={math.degrees(theta):.2f}°; "
        f"速度: vx={vx:.3f}, vy={vy:.3f}, omega={omega:.3f} rad/s"
    )


def _stop(agent, args):
    agent.base.stop()
    return "已停止"


def _run_demo_sequence(agent, args):
    """演示流程: 前进2m → 斜行45° 1m → 逆时针90° → 回起点 → 转回原朝向。"""
    steps = []
    x0, y0, theta0 = agent.base.get_odometry()
    steps.append(f"起点 odom=({x0:.2f}, {y0:.2f}, {math.degrees(theta0):.2f}°)")

    ok, _, end = _do_move(agent, 0, 2.0)
    steps.append(f"[1/5] 前进 2m: {'成功' if ok else '失败'}，位置=({end[0]:.2f}, {end[1]:.2f})")

    ok, _, end = _do_move(agent, 45, 1.0)
    steps.append(f"[2/5] 斜行 45° 平移 1m: {'成功' if ok else '失败'}，位置=({end[0]:.2f}, {end[1]:.2f})")

    _do_rotate(agent, 90)
    steps.append("[3/5] 原地逆时针自转 90° 完成")

    x, y, theta = agent.base.get_odometry()
    dx, dy = x0 - x, y0 - y
    dist = math.hypot(dx, dy)
    if dist > 0.05:
        world_dir = math.atan2(dy, dx)          # 世界系指向起点的方向
        local_dir = math.degrees(world_dir - theta)  # 换算到车头相对方向
        ok, _, end = _do_move(agent, local_dir, dist)
        steps.append(f"[4/5] 回起点 {dist:.2f}m: {'成功' if ok else '失败'}，位置=({end[0]:.2f}, {end[1]:.2f})")
    else:
        steps.append("[4/5] 已在起点附近，跳过回位")

    x, y, theta = agent.base.get_odometry()
    delta = math.degrees(theta0 - theta)
    if abs(delta) > 2:
        _do_rotate(agent, delta)
    steps.append(f"[5/5] 转回原朝向（Δ={delta:.1f}°）")

    agent.base.stop()
    x, y, theta = agent.base.get_odometry()
    steps.append(f"演示结束，最终 odom=({x:.2f}, {y:.2f}, {math.degrees(theta):.2f}°)")
    return "\n".join(steps)


# ══════════════════════════════════════════════════════════════════════
#  调度表: tool name → handler
# ══════════════════════════════════════════════════════════════════════

_HANDLERS = {
    "move": _move,
    "rotate": _rotate,
    "cruise": _cruise,
    "get_status": _get_status,
    "stop": _stop,
    "run_demo_sequence": _run_demo_sequence,
}


def execute_tool(agent, name: str, args: dict) -> str:
    """工具调度入口。未知工具 / handler 异常都返回错误字符串，让 LLM 看到并重试。"""
    agent.logger.info(f"执行工具: {name}({args})")
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"未知工具: {name}"
    try:
        return handler(agent, args)
    except Exception as e:
        return f"执行出错: {e}"
