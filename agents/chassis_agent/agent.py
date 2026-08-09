"""ChassisAgent: 全向四轮底盘（Agilex Ranger Mini 3）的 LLM 控制 agent。

照模板约定: 继承 BaseAgent, 在 __init__ 里设好 self.llm / self.model /
self.system_prompt / self.tools 并 setup_messages(), 实现 execute_tool()。
底盘对象挂在 self.base 上, 工具 handler 通过 agent.base.xxx 调用 SDK。

要点:
- AgilexRangeMini3 是工厂类: 构造时自动从平台 API 拉取机器人配置(SDF), 内部自建
  ROS2 节点并自旋, 返回可直接调用的 client。无需手动 rclpy.spin。
- 只跑在 rabo 平台控制器容器里 (依赖平台注入的 RABO_API_HOST / RDTP_USER_SCENE /
  RABO_ID + 沙箱 token 与 ROS2 环境), 本地无法联调。
"""

import logging

from openai import OpenAI

from core import BaseAgent

from . import config
from . import tools as tools_module
from .prompts import SYSTEM_PROMPT
from .tools import TOOLS


class ChassisAgent(BaseAgent):
    """全向底盘控制 agent: 用 LLM 工具调用驱动机器人平移 / 自转 / 查询 / 演示。"""

    def __init__(self):
        super().__init__()

        # ── 1. 初始化底盘 ─────────────────────────────────────────
        self.base = self._create_base()

        # ── 2. BaseAgent 协议要求的 4 个属性 ──────────────────────
        self.llm = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )
        self.model = config.LLM_MODEL
        self.system_prompt = SYSTEM_PROMPT
        self.tools = TOOLS
        self.setup_messages()

        self.logger.info(f"ChassisAgent 初始化完成, robot_id={config.ROBOT_ID}, mode={config.MODE}")

    @staticmethod
    def _create_base():
        """构造底盘客户端。SDK 依赖 ROS2 与平台环境, 缺失时给出明确错误。"""
        try:
            import rclpy
        except ImportError as e:
            raise RuntimeError("环境缺少 rclpy (ROS2)。请在 rabo 平台控制器中运行。") from e
        if not rclpy.ok():
            rclpy.init()

        from rabo_robocap import AgilexRangeMini3

        return AgilexRangeMini3(robot_id=config.ROBOT_ID, mode=config.MODE)

    def execute_tool(self, name: str, args: dict) -> str:
        """BaseAgent 钩子: 把工具调度转给 tools 模块。"""
        return tools_module.execute_tool(self, name, args)

    def shutdown(self):
        """释放底盘节点。"""
        try:
            self.base.shutdown()
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"底盘 shutdown 异常: {e}")


def _demo_run():
    """无输入源单发入口: 实例化 agent, 跑一遍演示, 退出。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log = logging.getLogger("ChassisAgent")
    log.info("启动演示...")
    agent = ChassisAgent()
    try:
        reply = agent.run(config.DEMO_TEXT)
        log.info("=== 最终回复 ===\n%s", reply)
    finally:
        agent.shutdown()
