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
# 场景构成 (官方汽车赛道):
#   - 柏油路面: 灰黑色 (低饱和 + 中明度)
#   - 白色边界线: 赛道两侧的白线 (低饱和 + 高明度) —— 最好的"赛道边界"信号
#   - 黑白相间路缘: 高低明度交替 (两者都被排除)
#   - 背景草坪: 绿色 (色相绿 + 饱和/明度较高)
#
# 策略: 分别做 灰沥青 / 白线 / 绿草 三个掩码。
#   1) 两条白边线的中点 = 权威赛道中心 (最准);
#   2) 白线缺失时退化为 灰沥青质心;
#   3) 两者都没有 → 判定偏出赛道。
# 上一版"排除绿色=路面"会把白线/路缘当路面, 导致车身压到草地还报"居中"。

_DEFAULTS = {
    "roi_bottom": 0.55,            # 视野底部多少比例算"近处路面" (ROI)
    "grass_h_range": (35, 85),     # 草坪色相区间
    "grass_s_min": 60,
    "grass_v_min": 60,
    "asphalt_s_max": 60,           # 沥青: 饱和度上限
    "asphalt_v_min": 40,           # 沥青: 明度下限 (排除纯黑/路缘黑段)
    "asphalt_v_max": 200,          # 沥青: 明度上限 (排除白线)
    "white_s_max": 60,             # 白线: 饱和度上限
    "white_v_min": 190,            # 白线: 明度下限
    "road_visible_fraction": 0.10, # 灰沥青占比低于此 → 判偏出
    "max_steer_deg": 30,
}


def _merge(vision):
    params = dict(_DEFAULTS)
    if vision:
        params.update(vision)
    return params


def detect_road(bgr, vision=None):
    """识别赛道, 返回 dict (键向后兼容):
        road_visible:    是否"在赛道上" (白线成对 or 灰沥青占比够)
        road_fraction:   灰沥青在底部 ROI 的占比
        road_offset:     灰沥青质心偏移 (-1..1, 正=偏右)
        steer_angle_deg: 建议转向角 (度, 正=左, 负=右) —— 优先白线中线, 退化灰质心
        white_ok:        两条白边界线是否都可见
        white_center:    白线中点偏移
        white_fraction:  白线像素占比
        grass_fraction:  绿草像素占比
        status:          人读描述
    """
    p = _merge(vision)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    y0 = int(h * p["roi_bottom"])
    roi = hsv[y0:, :]

    h_ch = roi[:, :, 0].astype(np.int16)
    s_ch = roi[:, :, 1].astype(np.int16)
    v_ch = roi[:, :, 2].astype(np.int16)
    roi_w = roi.shape[1]

    gh0, gh1 = p["grass_h_range"]
    grass = (
        (h_ch >= gh0) & (h_ch <= gh1)
        & (s_ch >= p["grass_s_min"]) & (v_ch >= p["grass_v_min"])
    )
    asphalt = (
        (s_ch < p["asphalt_s_max"])
        & (v_ch >= p["asphalt_v_min"]) & (v_ch <= p["asphalt_v_max"])
    )
    white = (s_ch < p["white_s_max"]) & (v_ch >= p["white_v_min"])

    total = roi.shape[0] * roi_w
    road_frac = float(asphalt.sum()) / total
    grass_frac = float(grass.sum()) / total
    white_frac = float(white.sum()) / total

    # ── 灰沥青质心 ──────────────────────────────────────────────
    road_offset = steer = None
    road_visible = road_frac >= p["road_visible_fraction"]
    if road_visible:
        cols = np.where(asphalt.any(axis=0))[0]
        cx = float(cols.mean()) / roi_w
        road_offset = (cx - 0.5) * 2.0
        steer = round(-road_offset * p["max_steer_deg"], 1)

    # ── 白边界线中点 (优先) ─────────────────────────────────────
    white_ok = False
    white_center = white_steer = None
    cols_w = np.where(white.any(axis=0))[0]
    if len(cols_w) >= 2:
        left, right = cols_w.min(), cols_w.max()
        if right - left > roi_w * 0.15:          # 两条线相距足够远, 才是左右边界
            wcx = (left + right) / 2.0 / roi_w
            white_center = (wcx - 0.5) * 2.0
            white_steer = round(-white_center * p["max_steer_deg"], 1)
            white_ok = True

    # ── 综合 ────────────────────────────────────────────────────
    eff_visible = road_visible or white_ok
    eff_offset = white_center if white_ok else road_offset
    eff_steer = white_steer if white_ok else steer

    if not eff_visible:
        status = (
            f"判定已偏出赛道: 灰路面仅占 {100 * road_frac:.0f}%, "
            f"白线占 {100 * white_frac:.0f}%, 绿草占 {100 * grass_frac:.0f}%。"
        )
    else:
        src = "白边线中线" if white_ok else "灰路面质心"
        side = "偏左" if eff_offset < -0.03 else ("偏右" if eff_offset > 0.03 else "居中")
        status = (
            f"在赛道上({src}), 中心{side} (偏移 {eff_offset:+.2f}) → 转向 {eff_steer:+.0f}°; "
            f"灰路面 {100 * road_frac:.0f}% / 白线 {100 * white_frac:.0f}% / 草 {100 * grass_frac:.0f}%"
        )

    return {
        "road_visible": eff_visible,
        "road_fraction": round(road_frac, 3),
        "road_offset": road_offset,
        "steer_angle_deg": eff_steer,
        "white_ok": white_ok,
        "white_center": white_center,
        "white_steer": white_steer,
        "white_fraction": round(white_frac, 3),
        "grass_fraction": round(grass_frac, 3),
        "status": status,
    }


def probe_scene(bgr, vision=None):
    """调试用: 输出画面里绿草/灰沥青/白线/其它颜色的像素占比, 用于校准 HSV 阈值。"""
    p = _merge(vision)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    y0 = int(bgr.shape[0] * p["roi_bottom"])
    roi = hsv[y0:, :]
    h_ch, s_ch, v_ch = roi[:, :, 0].astype(np.int16), roi[:, :, 1].astype(np.int16), roi[:, :, 2].astype(np.int16)
    total = roi.shape[0] * roi.shape[1]

    gh0, gh1 = p["grass_h_range"]
    grass = (h_ch >= gh0) & (h_ch <= gh1) & (s_ch >= p["grass_s_min"]) & (v_ch >= p["grass_v_min"])
    asphalt = (s_ch < p["asphalt_s_max"]) & (v_ch >= p["asphalt_v_min"]) & (v_ch <= p["asphalt_v_max"])
    white = (s_ch < p["white_s_max"]) & (v_ch >= p["white_v_min"])
    dark = v_ch < p["asphalt_v_min"]
    other = (~grass) & (~asphalt) & (~white) & (~dark)

    return {
        "image": (bgr.shape[1], bgr.shape[0]),
        "green_pct": round(100 * float(grass.sum()) / total, 1),
        "asphalt_pct": round(100 * float(asphalt.sum()) / total, 1),
        "white_pct": round(100 * float(white.sum()) / total, 1),
        "dark_pct": round(100 * float(dark.sum()) / total, 1),
        "other_pct": round(100 * float(other.sum()) / total, 1),
    }


def save_debug(bgr, vision=None, outdir="/tmp/chassis_cam_debug"):
    """把当前帧 + 各类掩码存成 PNG, 返回文件路径列表。用于查看相机到底看到了什么。"""
    import os

    p = _merge(vision)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gh0, gh1 = p["grass_h_range"]
    grass = ((hsv[:, :, 0] >= gh0) & (hsv[:, :, 0] <= gh1)
             & (hsv[:, :, 1] >= p["grass_s_min"]) & (hsv[:, :, 2] >= p["grass_v_min"]))
    asphalt = ((hsv[:, :, 1] < p["asphalt_s_max"])
               & (hsv[:, :, 2] >= p["asphalt_v_min"]) & (hsv[:, :, 2] <= p["asphalt_v_max"]))
    white = ((hsv[:, :, 1] < p["white_s_max"]) & (hsv[:, :, 2] >= p["white_v_min"]))

    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, mask in [("frame", None), ("asphalt", asphalt), ("white", white), ("grass", grass)]:
        img = bgr.copy() if name == "frame" else cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        path = os.path.join(outdir, f"{name}.png")
        cv2.imwrite(path, img)
        paths.append(path)
    return paths


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
