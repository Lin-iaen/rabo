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
import os
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
    {
        "type": "function",
        "function": {
            "name": "reset_position",
            "description": "把机器人瞬间复位到世界原点 (0, 0, 0, 朝向 0)。"
                           "用于小车跑偏/跑飞后重置测试起点。注意: 这是 Gazebo 位姿瞬移, "
                           "里程计读数不一定自动归零。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "teleport",
            "description": "把机器人瞬间传送到指定坐标 (x, y, 米) 并朝向 yaw_deg(度, 0=世界系正前)。"
                           "用于跳转测试/跨过障碍/快速到达某点。位姿瞬移, 里程计读数不保证同步。",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "目标 x 坐标, 米"},
                    "y": {"type": "number", "description": "目标 y 坐标, 米"},
                    "yaw_deg": {"type": "number", "description": "目标朝向, 度 (0=世界系正前方)"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "camera_view",
            "description": "用前视 RGB 相机看一眼赛道: 返回路面是否可见、路面中心相对视野的偏移"
                           "(左/右)、建议转向角。沿赛道行驶时用它做观测。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "camera_probe",
            "description": "调试工具: 报告当前画面里绿草/灰沥青/白线/暗部/其它颜色的像素占比, "
                           "并把当前帧与各类掩码存成 PNG。当相机判断不准(一直说居中但实际偏出)时用, "
                           "用于校准赛道颜色阈值。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "follow_track",
            "description": "沿柏油赛道自动跟随行驶指定秒数 (闭环寻迹): 持续看相机把赛道保持在"
                           "视野中央, 若偏出赛道会自动停车并说明。完成后返回行驶距离与当前位姿。",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number", "description": "跟随行驶时长, 秒 (建议 10~30)"},
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sdk_probe",
            "description": "调试工具: 把底盘 SDK (rabo_robocap AgilexRangeMini3 及其基类) 的实际"
                           "源码导出到项目 sdk_dump/ 目录, 用于核实 move_distance/rotate/"
                           "get_odometry 等方法的运动方向帧与符号约定。",
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
#  复位 / 传送 (基于 rabo_dev_kit.SetEntityPose, 走 /world/<name>/set_pose)
# ══════════════════════════════════════════════════════════════════════

def _discover_world(agent):
    """发现 Gazebo world 名: 优先用 config.WORLD, 否则扫 /world/<name>/set_pose 服务。"""
    if config.WORLD:
        return config.WORLD
    from rclpy.node import Node

    node = Node(f"world_probe_{os.getpid()}")
    try:
        services = node.get_service_names_and_types()  # [(name, [types]), ...]
    finally:
        node.destroy_node()
    marker = "/world/"
    for name, _types in services:
        if name.startswith(marker) and name.endswith("/set_pose"):
            return name[len(marker):-len("/set_pose")]
    raise RuntimeError(
        "未找到 /world/<name>/set_pose 服务, 无法自动发现 Gazebo world。"
        "请在 agents/chassis_agent/config.py 里设置 WORLD = '<world名>'。"
    )


def _get_pose_client(agent):
    """懒创建 SetEntityPose 客户端并缓存 (world 只发现一次)。"""
    client = getattr(agent, "_pose_client", None)
    if client is None:
        from rabo_dev_kit import SetEntityPose

        world = _discover_world(agent)
        client = SetEntityPose(world)
        agent._pose_client = client
        agent._pose_world = world
        agent.logger.info(f"SetEntityPose 就绪, world={world}")
    return client


def _reset_position(agent, args):
    try:
        client = _get_pose_client(agent)
    except Exception as e:
        return f"复位失败: {e}"
    ok = client.set(config.ROBOT_ID, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), timeout=10)
    if not ok:
        return "复位失败: SetEntityPose 调用超时或无响应"
    return ("已将机器人复位到原点 (0, 0, 0, yaw=0)。注意: 这是位姿瞬移, 里程计读数不一定自动归零, "
            "之后的移动会从当前位置继续累计。")


def _teleport(agent, args):
    x = args["x"]
    y = args["y"]
    yaw = math.radians(args.get("yaw_deg", 0))
    try:
        client = _get_pose_client(agent)
    except Exception as e:
        return f"传送失败: {e}"
    ok = client.set(config.ROBOT_ID, (x, y, 0.0, 0.0, 0.0, yaw), timeout=10)
    if not ok:
        return "传送失败: SetEntityPose 调用超时或无响应"
    return f"已传送到 ({x}, {y}, yaw={math.degrees(yaw):.1f}°)。注意: 里程计读数不保证同步。"


# ══════════════════════════════════════════════════════════════════════
#  赛道感知与跟随 (前视 RGB 相机, 灰黑柏油路 vs 绿色草坪)
# ══════════════════════════════════════════════════════════════════════

def _camera_view(agent, args):
    from drivers import detect_road

    frame = agent.camera.get_frame()
    if frame is None:
        return (
            f"相机还没收到图像 (topic={config.CAMERA_IMAGE_TOPIC})。"
            "请确认场景里相机已开启、话题名与 config.CAMERA_IMAGE_TOPIC 一致。"
        )
    det = detect_road(frame, config.VISION)
    return det["status"]


def _camera_probe(agent, args):
    """调试工具: 报告画面颜色统计 + 保存当前帧与掩码 PNG, 用于校准 HSV 阈值。"""
    from drivers import detect_road, probe_scene, save_debug

    frame = agent.camera.get_frame()
    if frame is None:
        return (
            f"相机还没收到图像 (topic={config.CAMERA_IMAGE_TOPIC})。"
            "请确认场景里相机已开启、话题名与 config.CAMERA_IMAGE_TOPIC 一致。"
        )
    stats = probe_scene(frame, config.VISION)
    det = detect_road(frame, config.VISION)
    try:
        paths = save_debug(frame, config.VISION, outdir=config.CAMERA_DEBUG_DIR)
        saved = f"调试图已存: {paths[0]} (另含 asphalt/white/grass 掩码)"
    except Exception as e:  # noqa: BLE001
        saved = f"保存调试图失败: {e}"

    def fmt(band):
        s = stats[band]
        return (f"{band}: 草{s['green']}% 灰{s['asphalt']}% 白{s['white']}% 暗{s['dark']}%")

    return (
        f"画面 {stats['image'][0]}x{stats['image'][1]} | {fmt('bottom')} | "
        f"{fmt('steer_band')} | {det['status']} | {saved}"
    )


def _sdk_probe(agent, args):
    """调试工具: 把底盘 SDK 源码导出到项目 sdk_dump/ 目录。

    rabo_robocap 顶层只是 loader shell, 真正实现在热更新到 ~/.rabo_robocap/*.zip 的
    _rabo_core 里 (子包 mobile)。zipimport 的类无法用 inspect.getmodule/getsource 正常解析,
    所以这里直接定位并解压 core zip, 把所有 .py 导出成 sdk_dump/core_*.py, 便于在网页
    VSCode 里直接点开查看。导出后重点看 mobile 子包里 move_distance / rotate /
    set_velocity / get_odometry 这几个方法的方向帧与符号约定。
    """
    import inspect
    import os
    import zipfile

    try:
        from rabo_robocap import AgilexRangeMini3
    except Exception as e:
        return f"sdk_probe 失败: 无法导入 AgilexRangeMini3 ({e})"

    outdir = os.path.join(os.getcwd(), "sdk_dump")
    os.makedirs(outdir, exist_ok=True)

    lines = [f"AgilexRangeMini3 = {AgilexRangeMini3}"]
    try:
        lines.append(f"MRO = {[c.__name__ for c in AgilexRangeMini3.__mro__]}")
    except Exception:  # noqa: BLE001
        pass
    try:
        lines.append(f"file = {inspect.getfile(AgilexRangeMini3)}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"file = <无法获取: {e}>")

    # ── 1. 尽量直接导出 MRO 里每个类的源码 (zipimport 类可能失败, 故只是尽力) ──
    for cls in AgilexRangeMini3.__mro__:
        try:
            src = inspect.getsource(cls)
        except Exception:  # noqa: BLE001
            src = None
        if not src:
            continue
        path = os.path.join(outdir, f"{cls.__name__}.py")
        with open(path, "w") as fh:
            fh.write(f"# class: {cls.__name__}\n# module: {cls.__module__}\n\n")
            fh.write(src)
        lines.append(f"dump class {cls.__name__} -> {path}")
        agent.logger.info(f"[sdk_probe] class {cls.__name__} -> {path}")

    # ── 2. 主路径: 直接解压 OSS core zip, 导出全部 .py ──
    zip_path = None
    try:
        from rabo_robocap._updater import get_core_zip_path

        zip_path = get_core_zip_path()
    except Exception:  # noqa: BLE001
        pass
    if not zip_path:
        try:
            f = inspect.getfile(AgilexRangeMini3)
            zip_path = f.split(".zip")[0] + ".zip"
        except Exception:  # noqa: BLE001
            pass

    if zip_path and os.path.isfile(zip_path):
        lines.append(f"core zip = {zip_path}")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                py_names = [n for n in zf.namelist() if n.endswith((".py", ".pyc"))]
                for n in py_names:
                    rel = n.replace("/", "_").replace("\\", "_")
                    with open(os.path.join(outdir, "core_" + rel), "wb") as fh:
                        fh.write(zf.read(n))
                lines.append(f"从 zip 导出 {len(py_names)} 个文件 (含 .pyc 字节码) 到 sdk_dump/core_*")
        except Exception as e:  # noqa: BLE001
            lines.append(f"解压 zip 失败: {e}")
    else:
        lines.append("未找到 core zip, 仅尝试导出类源码。")

    return "\n".join(lines)


def _omni_wheel_targets(vx, vy, omega, wheel_positions, wheel_radius):
    """复刻 SDK 的全向逆解: 由车体系 (vx, vy, omega) 求每个轮子的转向角与轮速。

    与 _rabo_core.base.omnidirectional_chassis_base.omni_wheel_targets 完全一致:
      Vx = vx - omega*y;  Vy = vy + omega*x;  steer = atan2(Vy, Vx);  speed = mag/r
    返回 (steers[rad], speeds[rad/s])。这样就能在 SDK 公开 API (纯平移/纯旋转) 之外
    合成「平移 + 自转」的弧线运动。
    """
    steers, speeds = [], []
    for x, y in wheel_positions:
        Vx = vx - omega * y
        Vy = vy + omega * x
        mag = math.hypot(Vx, Vy)
        if mag < 1e-9:
            steers.append(0.0)
            speeds.append(0.0)
        else:
            steers.append(math.atan2(Vy, Vx))
            speeds.append(mag / wheel_radius)
    return steers, speeds


def _set_twist(agent, vx, vy, omega):
    """把车体系 (vx, vy, omega) 直接写到服务端控制目标, 实现全向弧线运动。

    sim 模式下 agent.base 是 OmniChassisClient, 它持有 _server (本进程内的
    OmnidirectionalChassisBase 节点)。直接调 _server._set_targets 把逆解后的轮子
    目标写进去, 由服务端 50Hz 控制回路平滑执行 —— 这是绕过 SDK 只暴露纯平移/纯旋转、
    且「转向时不动」限制的关键, 让车能一边前进一边转弯而不必停-转-停。
    """
    server = getattr(agent.base, "_server", None)
    if server is None:
        raise RuntimeError("无法访问底盘服务端节点 (_server 不存在), 合成运动不可用")
    steers, speeds = _omni_wheel_targets(
        vx, vy, omega,
        wheel_positions=server.WHEEL_POSITIONS,
        wheel_radius=server.WHEEL_RADIUS,
    )
    server._set_targets(steers, speeds)


def _follow_track(agent, args):
    """闭环寻迹 (全向弧线跟随), 持续指定秒数。

    用合成运动 (前向 + 横向修正 + 航向跟随) 让车沿平滑弧线走, 不再「停-转-折线」。
    每周期: 看相机 → 算偏移/偏角 → 把 (vx, vy, omega) 经 _omni_wheel_targets 逆解后
    写进服务端 _set_targets, 由 50Hz 控制回路平滑执行。这是全向底盘的本命用法。
    """
    from drivers import detect_road

    duration = args["seconds"]
    speed = config.TRACK_SPEED
    dt = config.TRACK_CTRL_DT
    lost_limit = config.TRACK_LOST_STEPS
    k_lat = config.TRACK_LAT_GAIN
    k_yaw = config.TRACK_YAW_GAIN
    max_vy = config.TRACK_MAX_VY
    max_omega = config.TRACK_MAX_OMEGA
    smooth = config.TRACK_STEER_SMOOTH

    start = time.time()
    lost = 0
    prev_offset = 0.0
    prev_steer = 0.0
    last_log = 0.0
    x0, y0, _ = agent.base.get_odometry()

    while time.time() - start < duration:
        frame = agent.camera.get_frame()
        if frame is None:
            lost += 1
            if lost >= lost_limit:
                agent.base.stop()
                return "跟随失败: 长时间拿不到相机图像, 已停车。"
            time.sleep(0.1)
            continue

        det = detect_road(frame, config.VISION)
        if not det["road_visible"]:
            lost += 1
            if lost >= lost_limit:
                agent.base.stop()
                try:
                    from drivers import save_debug
                    save_debug(frame, config.VISION, outdir=config.CAMERA_DEBUG_DIR)
                except Exception:  # noqa: BLE001
                    pass
                return ("已偏出赛道: 连续多步看不到路面, 已停车。"
                        "调试图已存 " + config.CAMERA_DEBUG_DIR + "。")
            # 短暂丢失: 原地减速等待, 不继续冲
            _set_twist(agent, 0.0, 0.0, 0.0)
            time.sleep(0.1)
            continue

        lost = 0
        offset = det["road_offset"] or 0.0            # + 右, -1..1
        steer_deg = det["steer_angle_deg"] or 0.0     # + 左转

        # EMA 平滑
        offset = smooth * offset + (1.0 - smooth) * prev_offset
        steer_deg = smooth * steer_deg + (1.0 - smooth) * prev_steer
        prev_offset, prev_steer = offset, steer_deg

        vx = speed
        vy = max(-max_vy, min(max_vy, -k_lat * offset * speed))
        omega = max(-max_omega, min(max_omega, k_yaw * math.radians(steer_deg)))

        try:
            _set_twist(agent, vx, vy, omega)
        except Exception as e:  # noqa: BLE001
            agent.base.stop()
            return f"合成运动控制失败: {e}"

        t = time.time() - start
        if t - last_log >= 0.5:
            last_log = t
            _, _, theta = agent.base.get_odometry()
            agent.logger.info(
                f"[track] {t:.1f}s 偏移{offset:+.2f} 偏角{steer_deg:+.1f}° "
                f"vy={vy:+.2f} ω={omega:+.2f} 朝向{math.degrees(theta):.0f}° "
                f"底灰{det['road_fraction']:.0%} 上灰{det['far_fraction']:.0%}"
            )

        time.sleep(dt)

    agent.base.stop()
    x, y, theta = agent.base.get_odometry()
    traveled = math.hypot(x - x0, y - y0)
    return (
        f"沿赛道行驶完成: 约 {duration}s, 前进 {traveled:.1f} m。"
        f"当前位姿=({x:.2f}, {y:.2f}, {math.degrees(theta):.2f}°)。"
    )


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
    "reset_position": _reset_position,
    "teleport": _teleport,
    "camera_view": _camera_view,
    "camera_probe": _camera_probe,
    "follow_track": _follow_track,
    "sdk_probe": _sdk_probe,
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
