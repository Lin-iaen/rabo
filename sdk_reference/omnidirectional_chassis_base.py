"""全向底盘基类 —— 四轮四转，任意方向平移 + 原地自转（反编译重建）。

Python API: set_velocity(direction, speed) / rotate(ω) / move_distance(direction, dist) /
            get_odometry / get_velocity / stop
ROS2: 全部经通用 /{name}/call 与 /{name}/execute 调用（白名单）。

⚠️ 关键设计：底层控制回路 omni_control_step 规定「转向时绝不动车」——
即只有当 4 个轮子的转向角都接近目标（误差 <= STEER_TOL）后，才允许轮子加速转动。
因此任何「先转车身再前进」的策略都会每步停车；且 set_velocity/rotate 只能分别做
纯平移/纯旋转，无法合成弧线。
"""

import math
import time
import threading

try:
    import rclpy
    from rclpy.node import Node as NodeBase
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import Float64
    from nav_msgs.msg import Odometry
    from rclpy.action.server import GoalResponse, CancelResponse
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False
    class NodeBase:  # noqa
        pass


def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def _ramp(cur, tgt, step):
    diff = tgt - cur
    if abs(diff) <= step:
        return tgt
    return cur + (step if diff > 0 else -step)


def omni_control_step(current_steers, current_speeds, target_steers, target_speeds,
                      steer_step, accel_step, decel_step, steer_tol, speed_eps):
    """控制回路单步：先转向（静止）、到位后再加速，绝不边转边动。

    返回 (new_steers, new_speeds)。
    """
    max_err = max((abs(normalize_angle(t - c))
                   for t, c in zip(target_steers, current_steers)), default=0.0)
    moving = any(abs(s) > speed_eps for s in current_speeds)

    if max_err > steer_tol and moving:
        # 需要转向且正在动 → 先减速到 0（保持当前转向角）
        new_speeds = [_ramp(s, 0.0, decel_step) for s in current_speeds]
        new_steers = list(current_steers)
        return new_steers, new_speeds

    if max_err > steer_tol:
        # 需要转向且已停 → 以 steer_step 速率转转向角（轮子不动）
        new_steers = []
        for c, t in zip(current_steers, target_steers):
            diff = normalize_angle(t - c)
            new_steers.append(c + max(-steer_step, min(steer_step, diff)))
        new_speeds = [0.0] * len(current_speeds)
        return new_steers, new_speeds

    # 转向已到位 → 以 accel_step 加速到目标轮速
    new_speeds = [_ramp(s, t, accel_step) for s, t in zip(current_speeds, target_speeds)]
    new_steers = list(current_steers)
    return new_steers, new_speeds


def is_settled(current_steers, current_speeds, target_steers, target_speeds,
               steer_tol, speed_tol):
    steer_ok = all(abs(normalize_angle(t - c)) <= steer_tol
                   for t, c in zip(target_steers, current_steers))
    speed_ok = all(abs(t - c) <= speed_tol
                   for t, c in zip(target_speeds, current_speeds))
    return steer_ok and speed_ok


def omni_wheel_targets(vx, vy, omega, wheel_positions, wheel_radius):
    """全向逆解：由车体系 (vx, vy, omega) 求每个轮子的转向角与驱动轮转速。

    Vx = vx - omega*y;  Vy = vy + omega*x;  steer = atan2(Vy, Vx);  speed = mag/r
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


def translate_targets(direction, speed, num_wheels, wheel_radius):
    """纯平移：所有轮子转向同一个方向，驱动轮同转速。"""
    return [direction] * num_wheels, [speed / wheel_radius] * num_wheels


def rotate_targets(angular_velocity, wheel_positions, wheel_radius):
    """纯旋转：只给 |ω| 求轮目标，ω<0 时反转驱动轮方向。"""
    steers, speeds = omni_wheel_targets(0.0, 0.0, abs(angular_velocity),
                                        wheel_positions, wheel_radius)
    if angular_velocity < 0:
        speeds = [-s for s in speeds]
    return steers, speeds


class OmnidirectionalChassisBase(NodeBase):
    """全向底盘基类 - 四轮四转，任意方向平移 + 原地自转。"""

    NUM_WHEELS: int = 4
    WHEEL_RADIUS: float = 0.0
    WHEEL_POSITIONS = []
    JOINT_MARKERS = None

    STEER_SIGN: float = -1.0            # 转向关节指令方向翻倍（发布时乘此系数）

    DEFAULT_LINEAR_VELOCITY = 0.5
    DEFAULT_POSITION_TOLERANCE = 0.05

    DECEL = 5.0          # 减速度 (m/s²)
    ACCEL = 2.0          # 加速度 (m/s²)
    MAX_STEER_RATE = 8.0 # 转向角速率上限 (rad/s)
    STEER_TOL = 0.05     # 转向到位容差 (rad)
    SPEED_EPS = 0.05
    CONTROL_DT = 0.02    # 控制回路周期 (s) = 50Hz
    SETTLE_TIMEOUT = 6.0

    # 允许远程调用的白名单
    REMOTE_SERVICES = frozenset({'set_velocity', 'rotate', 'stop', 'get_odometry', 'get_velocity'})
    REMOTE_ACTIONS = {'move_distance'}

    def __init__(self, name: str, joints, odom_topic: str):
        validate_ros2_name(name)
        if RCLPY_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            super().__init__(name)

        expected = self.NUM_WHEELS * 2
        if len(joints) != expected:
            raise ValueError(f'关节数量不匹配: 期望 {expected}, 实际 {len(joints)}')

        self._name = name
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0
        self._odom_received = False
        self._is_moving = False
        self._stop_requested = False

        self._state_lock = threading.Lock()
        self._motion_lock = threading.Lock()

        self._target_steers = [0.0] * self.NUM_WHEELS
        self._target_speeds = [0.0] * self.NUM_WHEELS
        self._current_steers = [0.0] * self.NUM_WHEELS
        self._current_speeds = [0.0] * self.NUM_WHEELS
        self._control_lock = threading.Lock()
        self._control_running = False

        if RCLPY_AVAILABLE:
            self._callback_group = ReentrantCallbackGroup()
            self._setup_communication(joints, odom_topic)
            self._setup_remote_dispatch()

            self._executor = MultiThreadedExecutor(num_threads=4)
            self._executor.add_node(self)
            self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._spin_thread.start()

            self._control_running = True
            self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self._control_thread.start()

        self.get_logger().debug(f'{name} 初始化完成')

    def _setup_communication(self, joints, odom_topic: str):
        self._steering_pubs = []
        self._wheel_pubs = []
        for i, joint in enumerate(joints):
            pub = self.create_publisher(Float64, joint['cmd'], 10)
            if i % 2 == 0:
                self._steering_pubs.append(pub)
            else:
                self._wheel_pubs.append(pub)
        if odom_topic:
            self.create_subscription(Odometry, odom_topic, self._odom_callback, 10,
                                     callback_group=self._callback_group)

    def _odom_callback(self, msg):
        with self._state_lock:
            self._odom_x = msg.pose.pose.position.x
            self._odom_y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self._odom_theta = quaternion_to_yaw(q.x, q.y, q.z, q.w)
            self._vx = msg.twist.twist.linear.x
            self._vy = msg.twist.twist.linear.y
            self._omega = msg.twist.twist.angular.z
            self._odom_received = True

    def _goal_callback(self, _goal_request):
        if self._is_moving:
            self.get_logger().warning('底盘正在运动中，拒绝新目标')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _cancel_request):
        self.get_logger().info('收到取消请求')
        self._stop_requested = True
        return CancelResponse.ACCEPT

    def set_velocity(self, direction, speed, blocking=True):
        """纯平移（阻塞到转向到位并加速）。direction 车体系 rad，speed m/s。"""
        steers, speeds = translate_targets(direction, speed, self.NUM_WHEELS, self.WHEEL_RADIUS)
        self._set_targets(steers, speeds)
        if blocking:
            self._wait_until_settled()

    def rotate(self, angular_velocity, blocking=True):
        """纯旋转。angular_velocity rad/s，正=逆时针。"""
        steers, speeds = rotate_targets(angular_velocity, self.WHEEL_POSITIONS, self.WHEEL_RADIUS)
        self._set_targets(steers, speeds)
        if blocking:
            self._wait_until_settled()

    def stop(self):
        self._stop_requested = True
        with self._control_lock:
            self._target_speeds = [0.0] * self.NUM_WHEELS

    def _set_targets(self, steers, speeds):
        """写目标轮角/轮速（供本进程内直连使用，绕过 ROS 服务/动作）。"""
        self._stop_requested = False
        with self._control_lock:
            self._target_steers = list(steers)
            self._target_speeds = list(speeds)

    def _wait_until_settled(self, timeout=None):
        if timeout is None:
            timeout = self.SETTLE_TIMEOUT
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop_requested:
                return
            with self._control_lock:
                cs = list(self._current_steers)
                csp = list(self._current_speeds)
                ts = list(self._target_steers)
                tsp = list(self._target_speeds)
            if is_settled(cs, csp, ts, tsp, self.STEER_TOL, self.SPEED_EPS):
                return
            time.sleep(self.CONTROL_DT)

    def get_odometry(self):
        with self._state_lock:
            return (self._odom_x, self._odom_y, self._odom_theta)

    def get_velocity(self):
        with self._state_lock:
            return (self._vx, self._vy, self._omega)

    def move_distance(self, direction, distance, speed=None, blocking=True,
                      feedback_callback=None):
        """纯平移闭环：朝 direction 平移 distance 米（odom 闭环），阻塞到到位。

        注意：仅按 odom 的位移模长 hypot(dx,dy) 判停，不校验行进方向。
        """
        if speed is None:
            speed = self.DEFAULT_LINEAR_VELOCITY
        with self._motion_lock:
            self._is_moving = True
            self._stop_requested = False
            while not self._odom_received:
                time.sleep(0.01)
            start_x, start_y, _ = self.get_odometry()
            target = abs(distance)
            self.get_logger().info(f'全向直行: 方向={direction:.2f}rad 距离={distance:.2f}m 起点=({start_x:.2f},{start_y:.2f})')
            while True:
                if self._stop_requested:
                    self.stop()
                    self._is_moving = False
                    self.stop()
                    return False
                x, y, _ = self.get_odometry()
                traveled = math.hypot(x - start_x, y - start_y)
                if traveled >= target - self.DEFAULT_POSITION_TOLERANCE:
                    self.stop()
                    self.get_logger().info(f'到达目标，已移动 {traveled:.2f} m')
                    self._is_moving = False
                    self.stop()
                    return True
                self.set_velocity(direction, speed, blocking=False)
                if feedback_callback:
                    feedback_callback(traveled / target if target > 0 else 1.0)
                time.sleep(0.05)

    def _control_loop(self):
        """50Hz 后台控制回路：读取目标 → omni_control_step → 发布转向角/轮速。"""
        steer_step = self.MAX_STEER_RATE * self.CONTROL_DT
        decel_step = (self.DECEL / self.WHEEL_RADIUS) * self.CONTROL_DT
        accel_step = (self.ACCEL / self.WHEEL_RADIUS) * self.CONTROL_DT

        while self._control_running:
            if RCLPY_AVAILABLE and not rclpy.ok():
                return
            with self._control_lock:
                target_steers = list(self._target_steers)
                target_speeds = list(self._target_speeds)
                current_steers = list(self._current_steers)
                current_speeds = list(self._current_speeds)

            new_steers, new_speeds = omni_control_step(
                current_steers, current_speeds, target_steers, target_speeds,
                steer_step, decel_step, accel_step, self.STEER_TOL, self.SPEED_EPS)

            with self._control_lock:
                self._current_steers = new_steers
                self._current_speeds = new_speeds

            try:
                self._publish_steering(new_steers)
                self._publish_wheel_speeds(new_speeds)
            except Exception:
                return
            time.sleep(self.CONTROL_DT)

    def _publish_steering(self, angles):
        for pub, angle in zip(self._steering_pubs, angles):
            msg = Float64()
            msg.data = float(self.STEER_SIGN * angle)
            pub.publish(msg)

    def _publish_wheel_speeds(self, speeds):
        for pub, speed in zip(self._wheel_pubs, speeds):
            msg = Float64()
            msg.data = float(speed)
            pub.publish(msg)

    def _spin_loop(self):
        if RCLPY_AVAILABLE:
            self._executor.spin()

    def shutdown(self):
        # 停止控制回路 + 释放节点（略）
        self._control_running = False
        ...
