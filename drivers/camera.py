"""RGB 相机驱动 + 赛道感知（灰黑柏油路 vs 绿色草坪）。

────────────────────────────────────────────────────────────────────────
用法:
    from drivers import CameraNode, detect_road

    cam = CameraNode(topic="rd43088_tp_rgbd_d724d13b23/image")   # ROS2 节点, 需在
                                                                 # executor 里自旋
    bgr = cam.get_frame()          # 最新一帧 BGR ndarray, 无帧时返回 None
    det = detect_road(bgr)         # 识别赛道, 返回结构化感知结果

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
#   - 白色边界线: 赛道两侧的白线 (低饱和 + 高明度)
#   - 黑白相间路缘: 高低明度交替 (两者都被排除)
#   - 背景草坪: 绿色 (色相绿 + 饱和/明度较高)
#
# 关键几何 (踩过的坑):
#   赛道比视野宽得多 → "底部近处"几乎总是被沥青铺满, 灰色质心恒≈0, 无法反映
#   横向位置 (上一版因此漂到边线还报"居中")。
#   正确信号:
#   1) 「上/中段」(能看到路缘与背景交界处) 的灰色质心 —— 离边线越近, 远段路肩
#      越偏向一侧, 质心越偏。这是主转向信号。
#   2) 「底部段」检测到白线 → 车贴近该侧边界 → 施加"向赛道内推"的纠正。
#   3) 「上段」两条白线成对可见时, 用其中点精修转向角。

_DEFAULTS = {
    "steer_band": (0.05, 0.45),   # 上/中段 (y 归一化): 灰色质心主转向信号
    "bottom_y": 0.60,             # 底部段起点: "是否在路上" + "是否贴近边线"
    "grass_h_range": (35, 85),    # 草坪色相区间
    "grass_s_min": 60,
    "grass_v_min": 60,
    "asphalt_s_max": 60,          # 沥青: 饱和度上限
    "asphalt_v_min": 40,          # 沥青: 明度下限 (排除纯黑/路缘黑段)
    "asphalt_v_max": 200,         # 沥青: 明度上限 (排除白线)
    "white_s_max": 60,            # 白线: 饱和度上限
    "white_v_min": 190,           # 白线: 明度下限
    "road_visible_fraction": 0.10, # 底部灰占比低于此 → 判偏出
    "edge_white_frac": 0.03,      # 底部白线占比高于此 → 判定贴近边线
    "edge_bias_deg": 25,          # 贴近边线时的附加纠正 (度)
    "max_steer_deg": 30,
}


def _merge(vision):
    params = dict(_DEFAULTS)
    if vision:
        params.update(vision)
    return params


def _column_runs(mask, min_frac=0.04):
    """找出"纵向白像素数超阈值"的连续列区间, 返回 [(start, end), ...]。

    用于从白线掩码里提取左右边界线。min_frac: 列内白像素行数占比下限。
    """
    cnt = mask.sum(axis=0)
    th = max(3, int(mask.shape[0] * min_frac))
    strong = cnt > th
    runs = []
    i, n = 0, len(strong)
    while i < n:
        if strong[i]:
            j = i
            while j < n and strong[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def detect_road(bgr, vision=None):
    """识别赛道, 返回 dict (键向后兼容):
        road_visible:    是否"在赛道上" (底部灰占比够)
        road_fraction:   灰沥青在底部段的占比
        road_offset:     有效横向偏移 (-1..1, 正=偏右); 优先白线中点, 退化上段灰质心
        steer_angle_deg: 建议转向角 (度, 正=左, 负=右), 含贴近边线的纠正
        white_ok:        上段两条白边界线是否成对可见
        white_center:    白线中点偏移
        white_fraction:  白线像素占比 (整图)
        grass_fraction:  绿草像素占比 (整图)
        far_fraction:    上/中段灰沥青占比
        edge_bias:       底部边线纠正量 (度)
        status:          人读描述
    """
    p = _merge(vision)
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h_ch = hsv[:, :, 0].astype(np.int16)
    s_ch = hsv[:, :, 1].astype(np.int16)
    v_ch = hsv[:, :, 2].astype(np.int16)

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

    # ── 底部段: 是否在路面上 + 是否贴近边线 ──────────────────
    by = int(h * p["bottom_y"])
    bottom_gray = asphalt[by:, :]
    bottom_white = white[by:, :]
    bottom_total = bottom_gray.size
    gray_frac_bottom = float(bottom_gray.sum()) / bottom_total
    white_frac_bottom = float(bottom_white.sum()) / bottom_total
    on_road = gray_frac_bottom >= p["road_visible_fraction"]

    edge_bias = 0.0
    if on_road and white_frac_bottom > p["edge_white_frac"]:
        wcols = np.where(bottom_white.any(axis=0))[0]
        if len(wcols):
            wcx = float(wcols.mean()) / bottom_white.shape[1]
            # 白线偏左 → 车贴近左边界 → 右转(负); 反之为左转
            edge_bias = round(-p["edge_bias_deg"] * (0.5 - wcx) * 2.0, 1)

    # ── 上/中段: 灰色质心 (主转向信号) ──────────────────────
    sy0, sy1 = int(h * p["steer_band"][0]), int(h * p["steer_band"][1])
    band_gray = asphalt[sy0:sy1, :]
    bw = band_gray.shape[1]
    far_frac = float(band_gray.sum()) / band_gray.size
    far_offset = far_steer = None
    if band_gray.sum() > band_gray.size * 0.02:
        cols = np.where(band_gray.any(axis=0))[0]
        cx = float(cols.mean()) / bw
        far_offset = (cx - 0.5) * 2.0
        far_steer = round(-far_offset * p["max_steer_deg"], 1)

    # ── 上段白线成对 (精修) ─────────────────────────────────
    band_white = white[sy0:sy1, :]
    runs = _column_runs(band_white, min_frac=0.04)
    left_runs = [r for r in runs if (r[0] + r[1]) / 2 < bw * 0.5]
    right_runs = [r for r in runs if (r[0] + r[1]) / 2 >= bw * 0.5]
    white_ok = False
    line_offset = line_steer = None
    if left_runs and right_runs:
        lcx = (left_runs[0][0] + left_runs[0][1]) / 2 / bw
        rcx = (right_runs[-1][0] + right_runs[-1][1]) / 2 / bw
        if rcx - lcx > bw * 0.10:                # 两线距离足够, 才是左右边界
            mc = (lcx + rcx) / 2
            line_offset = (mc - 0.5) * 2.0
            line_steer = round(-line_offset * p["max_steer_deg"], 1)
            white_ok = True

    # ── 综合 ────────────────────────────────────────────────
    if white_ok:
        base = line_steer
        src = "白线中点"
    elif far_steer is not None:
        base = far_steer
        src = "上段灰质心"
    else:
        base = 0.0
        src = "无有效信号"

    steer = round(base + edge_bias, 1)
    eff_offset = line_offset if white_ok else far_offset
    max_steer = p["max_steer_deg"]
    steer = max(-max_steer, min(max_steer, steer))

    white_frac = float(white.sum()) / white.size
    grass_frac = float(grass.sum()) / grass.size

    if not on_road:
        status = (
            f"判定已偏出赛道: 底部灰路面仅 {100 * gray_frac_bottom:.0f}%, "
            f"白线 {100 * white_frac:.0f}%, 草 {100 * grass_frac:.0f}%。"
        )
    else:
        side = "偏左" if eff_offset is not None and eff_offset < -0.03 else (
            "偏右" if eff_offset is not None and eff_offset > 0.03 else "居中")
        extra = f", 边线纠正 {edge_bias:+.0f}°" if abs(edge_bias) > 1 else ""
        status = (
            f"在赛道上({src}), 中心{side} (偏移 {eff_offset if eff_offset is not None else 0:+.2f})"
            f" → 转向 {steer:+.0f}°{extra}; "
            f"底部灰 {100 * gray_frac_bottom:.0f}% / 上段灰 {100 * far_frac:.0f}% / 白 {100 * white_frac:.0f}%"
        )

    return {
        "road_visible": on_road,
        "road_fraction": round(gray_frac_bottom, 3),
        "road_offset": round(eff_offset, 3) if eff_offset is not None else None,
        "steer_angle_deg": steer,
        "white_ok": white_ok,
        "white_center": round(line_offset, 3) if line_offset is not None else None,
        "white_fraction": round(white_frac, 3),
        "grass_fraction": round(grass_frac, 3),
        "far_fraction": round(far_frac, 3),
        "edge_bias": edge_bias,
        "status": status,
    }


def probe_scene(bgr, vision=None):
    """调试用: 输出底部段与上/中段的 绿草/灰沥青/白线/暗部/其它 像素占比。"""
    p = _merge(vision)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = (hsv[:, :, 0].astype(np.int16), hsv[:, :, 1].astype(np.int16),
                        hsv[:, :, 2].astype(np.int16))
    gh0, gh1 = p["grass_h_range"]
    grass = ((h_ch >= gh0) & (h_ch <= gh1) & (s_ch >= p["grass_s_min"]) & (v_ch >= p["grass_v_min"]))
    asphalt = ((s_ch < p["asphalt_s_max"]) & (v_ch >= p["asphalt_v_min"]) & (v_ch <= p["asphalt_v_max"]))
    white = ((s_ch < p["white_s_max"]) & (v_ch >= p["white_v_min"]))
    dark = v_ch < p["asphalt_v_min"]

    def pct(mask, rows):
        sub = mask[rows, :]
        return round(100 * float(sub.sum()) / sub.size, 1)

    bottom_rows = slice(int(bgr.shape[0] * p["bottom_y"]), None)
    steer_rows = slice(int(bgr.shape[0] * p["steer_band"][0]), int(bgr.shape[0] * p["steer_band"][1]))
    out = {
        "image": (bgr.shape[1], bgr.shape[0]),
        "bottom": {"green": pct(grass, bottom_rows), "asphalt": pct(asphalt, bottom_rows),
                   "white": pct(white, bottom_rows), "dark": pct(dark, bottom_rows)},
        "steer_band": {"green": pct(grass, steer_rows), "asphalt": pct(asphalt, steer_rows),
                       "white": pct(white, steer_rows), "dark": pct(dark, steer_rows)},
    }
    return out


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
