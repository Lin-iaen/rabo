"""AgileX Ranger Mini 3 全向底盘实现（四轮四转 4WIS）—— 反编译重建。

四轮独立转向全向底盘，核心能力：任意方向平移（crab）+ 原地自转。每个轮组 = 转向关节
（JointPositionController，发角度）+ 驱动轮关节（JointController，发角速度 rad/s）。
全向运动学在 Python 侧实现（见 OmnidirectionalChassisBase.omni_wheel_targets）。

参数来源: original_data/agilex_range_mini3/robot.json。
左侧驱动轮轴向已调（fl/rl 0 -1 0），使所有驱动轮发同一速度指令即正常前进。

支持两种模式：
- mode='sim' (默认): 启动 OmnidirectionalChassisBase 服务端节点 + 返回 OmniChassisClient
- mode='real': 仅返回 OmniChassisClient，通过通用接口连接远程节点

Usage:
    import time, math
    base = AgilexRangeMini3(robot_id='range_mini3', mode='sim')

    base.set_velocity(0.0, 0.5);        time.sleep(3)   # 前进 0.5 m/s
    base.set_velocity(math.pi/2, 0.5);  time.sleep(3)   # 正左方平移
    base.set_velocity(math.pi/4, 0.5);  time.sleep(3)   # 斜行 45°
    base.rotate(0.3);                   time.sleep(3)   # 原地逆时针自转
    base.stop()

    base.move_distance(0.0, 1.0)        # 朝正前直行 1m（odom 闭环，阻塞）
    x, y, theta = base.get_odometry()
    base.shutdown()
"""

from typing import List, Dict, Optional

from base.omnidirectional_chassis_base import OmnidirectionalChassisBase
from utils.config import fetch_robot_config
from utils.sdf_parser import order_by_markers


class AgilexRangeMini3(OmnidirectionalChassisBase):
    ROBOT_TYPE = "agilex_range_mini3"

    # 关节 marker ID（8 个 = 4 转向 + 4 驱动）
    JOINT_MARKERS = [
        'i1f74e5686b91', 'i61485198b326', 'i0e3c0326b5c4', 'i976b37525abd',
        'i811b4981ce07', 'i9e1f064b2333', 'ic258d144d4e3', 'i6b0b7132eb69',
    ]

    NUM_WHEELS = 4
    WHEEL_RADIUS = 0.1
    # 车轮位置 (x, y)：x 前为正、y 左为正（REP-103）
    WHEEL_POSITIONS = [
        (0.25, -0.19), (0.25, 0.19), (-0.25, 0.19), (-0.25, -0.19),
    ]

    def __new__(cls, robot_id: str = 'sim', mode: str = 'sim',
                scene_id: Optional[str] = None,
                joints: Optional[List[Dict]] = None):
        from clients.omni_chassis_client import OmniChassisClient

        name = "m_" + robot_id

        if mode == 'real':
            client = OmniChassisClient(name=name, mode='real')
            client._server = None
            return client

        odom_topic = None
        if joints is None:
            config = fetch_robot_config(cls.ROBOT_TYPE, robot_id, mode, scene_id)
            joints = order_by_markers(config['joints'], cls.JOINT_MARKERS,
                                      expected=cls.NUM_WHEELS * 2)
            odom_topic = config.get('odom', {}).get('odom_topic')

        server = OmnidirectionalChassisBase.__new__(cls)
        OmnidirectionalChassisBase.__init__(server, name, joints, odom_topic)

        client = OmniChassisClient(name=name, mode='sim')
        client._server = server
        return client

    def __init__(self, robot_id='sim', mode='sim', scene_id=None, joints=None):
        # 真正的初始化都在 __new__ 里完成（工厂模式）
        pass
