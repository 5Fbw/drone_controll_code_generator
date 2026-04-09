import os
from openai import OpenAI
import config


class LLMProvider:
    """ 一个大语言模型提供者类，用于封装不同模型的API调用。 """

    # [修改点] 新增 model_name 参数，默认值为 "qwen3-14b" 以兼容旧代码
    def __init__(self, api_key=config.api_key_new, base_url=config.base_url, model_name="qwen3-14b"):
        """
        初始化LLM提供者。
        """
        # 在初始化时创建客户端，避免重复创建
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        # [修改点] 存储默认模型
        self.default_model = model_name

        self.reset_conversation()
        print(f"LLMProvider initialized for {base_url} with default model: {self.default_model}")

    def reset_conversation(self):
        """重置/清空对话历史，只保留系统提示词"""
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    # [修改点] 将 model 默认值改为 None，优先使用实例属性
    def qwen_api(self, prompt: str, model: str = None, stream: bool = True) -> str:
        """
        调用千问API回答问题。
        """
        # [修改点] 如果调用时未指定 model，则使用初始化时的 default_model
        current_model = model if model is not None else self.default_model

        print(f"\n--- Calling model: {current_model} ---")
        print("Assistant: ", end="", flush=True)
        full_response = ""
        # 用于存储 token 消耗信息
        usage_info = None
        try:
            # 根据stream参数决定是否设置enable_thinking
            extra_body = {"enable_thinking": stream}

            # 构造请求参数
            api_params = {
                "model": current_model,  # 使用确定的模型
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "extra_body": extra_body,
                "stream": stream
            }
            # 如果是流式模式，开启 usage 统计选项
            if stream:
                api_params["stream_options"] = {"include_usage": True}

            completion = self.client.chat.completions.create(**api_params)

            if stream:
                # 流式处理
                for chunk in completion:
                    # 1. 处理内容
                    # 关键修改：检查 delta.content 是否存在
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                    # 2. 处理 Token 统计 (通常在最后一个 chunk)
                    if hasattr(chunk, 'usage') and chunk.usage is not None:
                        usage_info = chunk.usage
            else:
                # 非流式处理
                full_response = completion.choices[0].message.content
                # 直接获取 usage
                if hasattr(completion, 'usage') and completion.usage is not None:
                    usage_info = completion.usage
                print(full_response)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            return ""

        # 打印换行使输出整洁
        print()
        # 打印 Token 消耗统计
        if usage_info:
            print(f"📊 Token消耗统计:")
            print(f" - Prompt Tokens: {usage_info.prompt_tokens}")
            print(f" - Completion Tokens: {usage_info.completion_tokens}")
            print(f" - Total Tokens: {usage_info.total_tokens}")
        else:
            print("⚠️ 未能获取 Token 统计信息 (可能当前API版本不支持流式统计)")
        print("--- End of response ---")
        return full_response, usage_info
