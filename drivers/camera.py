"""RGB 相机驱动 + 赛道感知（灰黑柏油路 vs 绿色草坪）。

────────────────────────────────────────────────────────────────────────
用法:
    from drivers import CameraNode, detect_road

    cam = CameraNode(topic="rd43088_tp_rgbd_d724d13b23/image")   # ROS2 节点, 需在
                                                                 # executor 里自旋
    bgr = cam.get_frame()          # 最新一帧 BGR ndarray, 无帧时返回 None
    det = detect_road(bgr)         # 识别路面, 返回结构化感知结果

本模块不依赖具体 agent; 底盘/机械臂都能复用。深度与点云暂时不处理
(当前任务只用 RGB, 见 README 教程四)。
────────────────────────────────────────────────────────────────────────
"""

import threading

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  图像解码: sensor_msgs/Image → numpy BGR
# ══════════════════════════════════════════════════════════════════════

def decode_image(msg):
    """把 sensor_msgs/Image 解码成 BGR ndarray (H, W, 3)。

    支持 rgb8 / bgr8 / rgba8 / bgra8 四种常见编码。
    """
    enc = (msg.encoding or "").lower()
    if enc in ("rgb8", "rgba8", "bgr8", "bgra8"):
        channels = 4 if "a" in enc else 3
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, channels)
        )[:, :, :3]
        return arr[:, :, ::-1] if enc.startswith("rgb") else arr
    raise ValueError(f"不支持的图像编码: {msg.encoding!r}")


# ══════════════════════════════════════════════════════════════════════
#  赛道感知: 灰黑柏油路 vs 绿色草坪
# ══════════════════════════════════════════════════════════════════════
#
# 原理: 草坪是绿色 (HSV 色相绿, 饱和/明度较高), 柏油路是灰黑色 (低饱和)。
# 用 HSV 把绿色草坪排除掉, 剩下就是路面; 路面像素质心的水平位置就是
# "路面在视野的哪个方向", 转成转向角给闭环跟随用。

GRASS_H_MIN, GRASS_H_MAX = 35, 85      # 绿色色相区间
GRASS_S_MIN = 60                        # 草坪最小饱和度
GRASS_V_MIN = 60                        # 草坪最小明度
ROAD_MIN_V = 20                         # 路面最小明度 (过滤纯黑/传感器死区)

# 视野底部多少比例算"近处路面" (ROI)。底部 = 车前最近的地面, 最可靠。
ROI_BOTTOM = 0.55

# 判定"能看到路"的最小路面占比 (否则认为偏出/被挡住)。
ROAD_VISIBLE_FRACTION = 0.15

# 最大转向角 (度): 把偏移 -1..1 映射到转向角 ±MAX_STEER。
MAX_STEER_DEG = 30


def detect_road(bgr):
    """识别柏油路面, 返回 dict:
        road_visible:   视野内是否看到足够多的路面
        road_fraction:  底部 ROI 中路面像素占比 (0~1)
        road_offset:    路面中心相对视野中心的偏移 (-1..1, 正=偏右)
        steer_angle_deg:建议转向角 (度, 正=左转, 负=右转, 相对车头)
        status:         人读描述
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    y0 = int(h * ROI_BOTTOM)
    roi = hsv[y0:, :]

    h_ch = roi[:, :, 0].astype(np.int16)
    s_ch = roi[:, :, 1].astype(np.int16)
    v_ch = roi[:, :, 2].astype(np.int16)

    grass = (
        (h_ch >= GRASS_H_MIN) & (h_ch <= GRASS_H_MAX)
        & (s_ch >= GRASS_S_MIN) & (v_ch >= GRASS_V_MIN)
    )
    road = (~grass) & (v_ch >= ROAD_MIN_V)

    total = road.shape[0] * road.shape[1]
    fraction = float(road.sum()) / total
    road_visible = fraction >= ROAD_VISIBLE_FRACTION

    if road_visible:
        cols = np.where(road.any(axis=0))[0]
        cx = float(cols.mean()) / road.shape[1]       # 0..1, 0=最左
        offset = (cx - 0.5) * 2.0                     # -1..1, 正=偏右
        steer = round(-offset * MAX_STEER_DEG, 1)     # 正=左转
        side = "偏左" if offset < -0.03 else ("偏右" if offset > 0.03 else "居中")
        status = f"看到路面(占{100 * fraction:.0f}%), 路面中心{side} (偏移 {offset:+.2f}), 建议转向 {steer:+.0f}°"
    else:
        offset = steer = None
        status = f"视野内路面占比仅 {100 * fraction:.0f}%, 判定已偏出赛道或路面被遮挡"

    return {
        "road_visible": road_visible,
        "road_fraction": round(fraction, 3),
        "road_offset": offset,
        "steer_angle_deg": steer,
        "status": status,
    }


# ══════════════════════════════════════════════════════════════════════
#  CameraNode: ROS2 图像订阅, 后台保存最新一帧
# ══════════════════════════════════════════════════════════════════════

class CameraNode:
    """订阅 sensor_msgs/Image, 始终保留最新一帧 BGR 图像。

    线程安全: get_frame() 任何时候调用都返回最近一次回调解出的帧 (或 None)。
    用法: 创建后把它 add 进某个 executor 自旋 (agent 里自带后台线程)。
    """

    def __init__(self, topic: str, qos: int = 10):
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        self._node = Node("chassis_camera")
        self._lock = threading.Lock()
        self._frame = None
        self._topic = topic
        self._sub = self._node.create_subscription(
            Image, topic, self._on_image, qos,
        )

    def _on_image(self, msg):
        try:
            bgr = decode_image(msg)
        except Exception:
            return
        with self._lock:
            self._frame = bgr

    def get_frame(self):
        with self._lock:
            return self._frame

    def get_node(self):
        return self._node

    def destroy(self):
        with self._lock:
            self._frame = None
        self._node.destroy_node()
