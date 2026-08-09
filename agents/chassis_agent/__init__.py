"""ChassisAgent 子包入口。

main.py 通过 importlib 加载本包并调用 run() —— 这是平台启动的入口。
当前为「无输入源」方案: run() 里实例化 agent 后直接发一条演示指令,
LLM 规划并调用底盘工具, 跑完即退出 (验证用, 见 README「接输入源」部分)。
"""

import logging

from .agent import ChassisAgent
from . import config

__all__ = ["ChassisAgent", "run"]


def run():
    """在 rabo 平台控制器容器里被 `python3 -u main.py` 启动。

    无输入源: 启动 → 跑一遍演示流程 → 退出。日志 (每轮 LLM 规划 + 每个工具
    结果) 打印到平台控制台, 在那里核对小车每一步动作。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
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
