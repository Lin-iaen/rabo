"""全向底盘客户端 —— 反编译重建。

所有方法经通用 /{name}/call 与 /{name}/execute 调用（对应服务端白名单
REMOTE_SERVICES / REMOTE_ACTIONS）。sim 模式下与 OmnidirectionalChassisBase 服务端
同进程，且持有 `_server` 引用（可直连 _set_targets 做合成运动）。
"""

import time
import threading

try:
    import rclpy
    from rclpy.node import Node as RclpyNode
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    RCLPY_AVAILABLE = True
except ImportError:
    RCLPY_AVAILABLE = False


class OmniChassisClient:
    """全向底盘客户端。所有方法经通用 /{name}/call 与 /{name}/execute 调用。"""

    DEFAULT_ACTION_TIMEOUT = 30.0
    DEFAULT_SERVICE_TIMEOUT = 8.0

    def __init__(self, name: str, mode: str = 'sim'):
        validate_ros2_name(name)
        self._name = name
        self._mode = mode
        self._server = None
        self._shutdown_done = False

        if not RCLPY_AVAILABLE:
            return
        if not rclpy.ok():
            rclpy.init()
        self._node = RclpyNode(f'{name}_client')
        self._callback_group = ReentrantCallbackGroup()
        self._setup_remote_clients(name)          # 建立 /{name}/call 与 /{name}/execute 客户端
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

    # ── 公开 API（全部走通用 call/execute 调度）────────────────────

    def set_velocity(self, direction: float, speed: float, blocking: bool = True):
        self.call('set_velocity', direction=float(direction), speed=float(speed),
                  blocking=blocking)

    def rotate(self, angular_velocity: float, blocking: bool = True):
        self.call('rotate', angular_velocity=float(angular_velocity), blocking=blocking)

    def stop(self):
        self.call('stop')

    def get_odometry(self):
        r = self.call('get_odometry')
        if isinstance(r, list) and len(r) == 3:
            return tuple(r)
        return (0.0, 0.0, 0.0)

    def get_velocity(self):
        r = self.call('get_velocity')
        if isinstance(r, list) and len(r) == 3:
            return tuple(r)
        return (0.0, 0.0, 0.0)

    def move_distance(self, direction: float, distance: float, speed=None,
                      blocking=True, feedback_callback=None):
        params = {'direction': float(direction), 'distance': float(distance)}
        if speed is not None:
            params['speed'] = float(speed)
        return self.execute('move_distance', params, feedback_callback=feedback_callback)

    def shutdown(self):
        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True
        # 释放节点 / 停止自旋（略）

    # ── 通用同步调度（略，见 base/remote_command.py）────────────────

    def _call_service_sync(self, name, params):
        ...

    def _call_action_sync(self, name, params, feedback_callback=None):
        ...

    def _handle_feedback(self, fb):
        ...

    def _spin_loop(self):
        self._executor.spin()
