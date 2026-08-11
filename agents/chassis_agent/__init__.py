"""ChassisAgent 子包入口。

main.py 通过 importlib 加载本包并调用 run() —— 这是平台启动的入口。

两种运行模式 (由 config 决定):
- 配置了 REMOTE_ID: 常驻监听 H5 遥控面板的 chat 控件, 实时对话控制 (阶段 B)。
- 未配置 REMOTE_ID: 一次性演示, 启动后跑一遍 DEMO_TEXT 指令即退出 (阶段 A)。
"""

import logging

from . import config
from .agent import ChassisAgent

__all__ = ["ChassisAgent", "run"]


def _one_shot_demo():
    """无输入源: 启动 → 跑一遍 DEMO_TEXT 指令 → 退出。"""
    log = logging.getLogger("ChassisAgent")

    if not config.LLM_API_KEY:
        log.error("缺少 LLM_API_KEY: 请在平台个人中心申请【内部使用】Key, 平台会自动"
                  "注入为 RABO_LLM_KEY 环境变量; 或在 config.py 手动配置。")
        return

    agent = ChassisAgent()
    try:
        log.info(f"指令: {config.DEMO_TEXT}")
        reply = agent.run(config.DEMO_TEXT)
        log.info("=== 最终回复 ===\n%s", reply)
    finally:
        agent.shutdown()
    log.info("演示结束, 进程退出。")


def _panel_mode():
    """常驻监听 H5 遥控面板的 chat 控件: 收到文字 → agent 处理 → 回复回传到面板。

    说明:
    - RemoteControl 是 ROS2 节点, 进程靠 rclpy.spin(rc) 常驻;
    - 底盘 SDK (sim 模式) 自带后台 executor 自旋, 与这里 spin 的 RemoteControl
      节点互不影响 (见 README「教程四」);
    - 回调在 RemoteControl 的 executor 线程里执行, agent.run() 是阻塞的 (LLM+工具
      可能耗时数秒), 期间不处理新的面板消息 —— 对单用户对话足够。
    """
    import rclpy
    from rabo_dev_kit import RemoteControl

    log = logging.getLogger("ChassisAgent")

    rclpy.init()
    agent = ChassisAgent()

    rc = None

    def on_control(data: dict):
        """RemoteControl 回调。data 形如 {'chat-v1': {'value': '用户输入'}, ...}。

        只处理 chat 控件的 value, 其它控件事件 (按钮/滑块) 直接忽略。
        """
        text = (data.get(config.CHAT_CONTROL_ID) or {}).get("value")
        if not text:
            return
        agent.logger.info(f"[panel] 收到: {text}")
        try:
            reply = agent.run(str(text))
        except Exception as e:
            reply = f"处理出错: {e}"

        if rc is not None:
            try:
                rc.send(config.CHAT_CONTROL_ID, reply)
            except Exception as e:
                agent.logger.error(f"[panel] 回传失败: {e}")

    rc = RemoteControl(remote_id=config.REMOTE_ID, callback=on_control)
    agent.logger.info(
        f"ChassisAgent 已就绪, 监听控制面板 {config.REMOTE_ID} 的 "
        f"{config.CHAT_CONTROL_ID} 控件。"
    )

    try:
        rclpy.spin(rc)
    except KeyboardInterrupt:
        pass
    finally:
        agent.shutdown()
        rc.destroy_node()
        rclpy.try_shutdown()


def run():
    """平台入口: 按 config.REMOTE_ID 是否配置, 选择面板模式或一次性演示模式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if config.REMOTE_ID:
        _panel_mode()
    else:
        _one_shot_demo()
