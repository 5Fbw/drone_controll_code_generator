import LLm_provider
import os
import airsim
# 安全地获取API Key (推荐方式)
# 1. 设置环境变量 DASHSCOPE_API_KEY
# 2. 在代码中读取
# try:
#     api_key = os.getenv("DASHSCOPE_API_KEY")
#     if not api_key:
#         # 如果环境变量未设置，为了演示，我们仍然使用硬编码的key，但会给出警告
#         # 强烈建议你在实际项目中删除这个备用方案
#         print("Warning: DASHSCOPE_API_KEY environment variable not set. Using hardcoded key.")
#         api_key = "sk-16d264b6b2d94e90a04b9e2dec5d2373"
# except Exception:
#     api_key = "sk-16d264b6b2d94e90a04b9e2dec5d2373"
# 初始化LLMProvider
# 使用DashScope的兼容模式URL
llm_provider = LLm_provider.LLMProvider()
library_name = "airsim"
action_name = "起飞"
action_name = "以指定速度飞行"
# 定义要问的问题
prompt = f"""
    ## Context（背景）
    你是一名无人机控制代码专家，精通 {library_name} 库。

    ## Objective（目标）
    1. 找到 {library_name} 库中关于 "{action_name}" 动作的最新API。
    2. 给出该API的参数和返回值。
    3. 给出代码示例。

    ## Style（风格）
    风格应是学术性的、严谨的。

    ## Tone（语调）
    语调应保持中立和专业。

    ## Audience（受众）
    使用Python代码控制无人机飞行的程序员。

    ## Response（回应）
    回应格式应包括以下部分：
    - **无人机动作描述**：描述 "{action_name}" 动作。
    - **函数输入参数**：参数名、参数类型。
    - **函数返回值**：返回内容。
    - **代码示例**：给出在 {library_name} 中执行 "{action_name}" 的实际示例。
    """
# 调用API并获取回答
# 默认使用流式输出
# response = llm_provider.qwen_api(question)

# 你也可以选择非流式输出

# response = llm_provider.qwen_api(prompt, stream=False)
