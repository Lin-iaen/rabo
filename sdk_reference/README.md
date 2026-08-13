# rabo_robocap 底盘 SDK 源码重建

> ⚠️ 这些文件是**从 `.pyc` 字节码手工反汇编还原**的（Python 3.12 尚无可用的纯 Python
> 反编译器）。函数/方法逻辑逐条还原、真实可靠；但变量名、注释、空行等是重建的，
> **不等于原始 `.py` 源码**。原始文件在平台的 `~/.rabo_robocap/core-cp312.zip` 里，
> 只有编译后的 `.pyc`，没有 `.py`。

## 为什么只有 `.pyc`

`rabo_robocap` 顶层只是一个 loader 壳（PyPI 发布的那几行），真正代码从阿里云 OSS
热更新拉取：

```
https://rabo-apt-repo.oss-cn-hangzhou.aliyuncs.com/rabo_robocap/core-cp312.zip
```

解压后是 `_rabo_core/` 包，全部 `.pyc`。目录结构：

```
_rabo_core/
├── mobile/agilex_range_mini3.pyc      # AgilexRangeMini3（本工程用的底盘）
├── base/omnidirectional_chassis_base.pyc   # 全向底盘基类（核心控制）
├── base/mobile_base.pyc               # 普通移动底盘基类（未用）
├── clients/omni_chassis_client.pyc    # 全向底盘客户端
├── clients/mobile_client.pyc          # 普通底盘客户端（未用）
├── utils/kinematics.pyc               # 通用运动学工具
├── arms/ ... grippers/ ... legged/ ...  # 机械臂/夹爪/足式等
```

## 关键约定（写 agent 必须知道）

| 方法 | 约定 |
| --- | --- |
| `move_distance(direction, dist)` | 纯平移；`direction` 是**车体系**，0=正前，+π/2=左，单位 rad |
| `rotate(ω)` | 纯旋转；+ω = **逆时针** |
| `get_odometry()` | 世界系 `(x, y, theta)`，theta 逆时针为正（`quaternion_to_yaw` 标准解算） |
| `set_velocity(direction, speed)` | 纯平移，可 `blocking=False` |
| 底层控制 | `omni_control_step`：**转向时绝不动车**，且 SDK 只暴露纯平移/纯旋转，无法合成弧线 |

## 对 agent 工程的影响

- SDK 的 `set_velocity`/`rotate` 只能分别做纯平移/纯旋转，且底层"转向时停车"。
  因此 `follow_track` 若要平滑过弯，需要**直接写服务端控制目标**（本工程 `tools.py`
  里的 `_omni_wheel_targets` + `_set_twist` 就是复刻 SDK 的逆解，绕过公开 API 限制）。
