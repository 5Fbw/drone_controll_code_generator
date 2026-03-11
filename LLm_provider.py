import os
from openai import OpenAI
import config
class LLMProvider:
    """
    一个大语言模型提供者类，用于封装不同模型的API调用。
    """
    def __init__(self, api_key=config.api_key_new, base_url=config.base_url):
        """
        初始化LLM提供者。

        Args:
            api_key (str): API密钥。
            base_url (str): API的基础URL。
        """
        # 在初始化时创建客户端，避免重复创建
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        messages = [{"role": "user", "content": "你是谁"}]
        completion = self.client.chat.completions.create(
            model="qwen3-32b",  # 您可以按需更换为其它深度思考模型
            messages=messages,
            # extra_body={"enable_thinking": True},
            stream=True
        )
        self.reset_conversation()

        print(f"LLMProvider initialized for {base_url}")
    def reset_conversation(self):
        """重置/清空对话历史，只保留系统提示词"""
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    # def qwen_api(self, prompt: str, model: str = "qwen3-coder-plus-2025-07-22", stream: bool = True) -> str:
    def qwen_api(self, prompt: str, model: str = "qwen3-32b", stream: bool = True) -> str:
        """
        调用千问API回答问题。

        Args:
            prompt (str): 用户输入的问题。
            model (str): 要使用的模型名称。
            stream (bool): 是否使用流式输出。

        Returns:
            str: 千问模型的完整回答。
        """
        print(f"n--- Calling model: {model} ---")
        print("Assistant: ", end="", flush=True)

        full_response = ""
        try:
            # 根据stream参数决定是否设置enable_thinking
            extra_body = {"enable_thinking": stream}
            
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                extra_body=extra_body,
                stream=stream
            )

            if stream:
                # 流式处理
                for chunk in completion:
                    # 关键修改：检查 delta.content 是否存在
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
            else:
                # 非流式处理
                full_response = completion.choices[0].message.content
                print(full_response)

        except Exception as e:
            print(f"nAn error occurred: {e}")
            return "" # 发生错误时返回空字符串

        # 在流式输出结束后打印一个换行符，让界面更整洁
        print("n--- End of response ---")
        return full_response

















# --- 如何使用这个类 ---

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
#
#
# # 初始化LLMProvider
# # 使用DashScope的兼容模式URL
# llm_provider = LLMProvider(
#     api_key=api_key,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )
#
# # 定义要问的问题
# question = "用Python写helloworld例子。"
#
# # 调用API并获取回答
# # 默认使用流式输出
# # response = llm_provider.qwen_api(question)
#
# # 你也可以选择非流式输出
# response = llm_provider.qwen_api(question, stream=False)

# 函数返回了完整的回答，你可以继续使用它
# print(f"nFull response captured: {response}")
