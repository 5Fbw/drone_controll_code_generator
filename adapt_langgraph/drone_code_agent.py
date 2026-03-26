from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import config
import re
from langgraph.graph.message import add_messages
import operator


# --- Reducer 函数定义 ---
def keep_last_value(left: str, right: str) -> str:
    """保留最新的值"""
    return right


def max_int(left: int, right: int) -> int:
    """返回两个整数中的最大值"""
    return max(left, right)


def sum_int(left: int, right: int) -> int:
    """累加整数"""
    return left + right


# --- 状态定义 ---
class AgentState(TypedDict):
    # 消息列表：追加合并
    messages: Annotated[list, add_messages]

    # 字符串字段：保留最新值
    instruction: Annotated[str, keep_last_value]
    elements_list: Annotated[str, keep_last_value]
    code: Annotated[str, keep_last_value]
    code_elements_list: Annotated[str, keep_last_value]
    comparison_result: Annotated[str, keep_last_value]
    current_agent: Annotated[str, keep_last_value]

    # 字典字段：合并
    code_output: Annotated[dict, operator.or_]

    # 布尔值：或运算
    code_consistent: Annotated[bool, operator.or_]

    # 整数字段：取最大值或累加
    iteration: Annotated[int, max_int]
    max_iterations: Annotated[int, max_int]

    # Token消耗字段：累加
    total_tokens: Annotated[int, sum_int]
    prompt_tokens: Annotated[int, sum_int]
    completion_tokens: Annotated[int, sum_int]


# --- Agent 类 ---
class DroneCodeGenerator:
    def __init__(self, model_name: str = "qwen3-14b"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=config.api_key_new,
            base_url=config.base_url,
            model_kwargs={"extra_body": {"enable_thinking": False}}
        )
        self.seed_code = config.seed_code

    def _update_token_usage(self, response, state: AgentState) -> dict:
        """从LLM响应中提取并累加token使用信息"""
        token_info = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }

        # 尝试从不同位置获取token使用信息
        try:
            # 方式1: 从usage_metadata获取
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_info["total_tokens"] = response.usage_metadata.get('total_tokens', 0)
                token_info["prompt_tokens"] = response.usage_metadata.get('input_tokens', 0)
                token_info["completion_tokens"] = response.usage_metadata.get('output_tokens', 0)

            # 方式2: 从response_metadata获取
            elif hasattr(response, 'response_metadata') and response.response_metadata:
                token_usage = response.response_metadata.get('token_usage', {})
                token_info["total_tokens"] = token_usage.get('total_tokens', 0)
                token_info["prompt_tokens"] = token_usage.get('prompt_tokens', 0)
                token_info["completion_tokens"] = token_usage.get('completion_tokens', 0)

            # 打印当前调用的token消耗
            if token_info["total_tokens"] > 0:
                print(f"📊 本次调用Token消耗:")
                print(f"   - Prompt Tokens: {token_info['prompt_tokens']}")
                print(f"   - Completion Tokens: {token_info['completion_tokens']}")
                print(f"   - Total Tokens: {token_info['total_tokens']}")

                # 计算并打印累计token消耗
                current_total = state.get("total_tokens", 0) + token_info["total_tokens"]
                current_prompt = state.get("prompt_tokens", 0) + token_info["prompt_tokens"]
                current_completion = state.get("completion_tokens", 0) + token_info["completion_tokens"]
                print(f"📊 累计Token消耗:")
                print(f"   - 总Prompt Tokens: {current_prompt}")
                print(f"   - 总Completion Tokens: {current_completion}")
                print(f"   - 总Tokens: {current_total}")
        except Exception as e:
            print(f"⚠️ 无法获取token使用信息: {e}")

        return token_info

    def movement_extractor_agent(self, state: AgentState) -> dict:
        print("=" * 60)
        print("🔄 执行 Movement Extractor Agent")
        print("=" * 60)
        instruction = state["instruction"]
        print(f"输入指令: {instruction}")

        prompt = f"""
        ## Objective（目标）
        1. 讲输入的指令拆解成单步动作指令。
        2. 给出距离、速度、坐标系（全局或机身），一般来说向前、向后、向左向右（包含前进、后退等提示）提取为机身，向西、向东、向北、向南是全局，向上、向下是全局，移动后的当前飞机全局位置与朝向。
        回应格式只包括以下几个动作：
        - 起飞。
        - 降落。
        - 移动： 距离 速度（没有提到即为null） 时间（没有提到即为null） 方向 坐标系（全局或机身）。（x,y,z,角度）当前动作完成后的全局位置与朝向
        - 移动至 位置 速度（没有提到即为null） 时间（没有提到即为null） 方向 坐标系（全局或机身）。（x,y,z,角度）
        - 转向： 角度 速度（可以为null） 时间（没有提到即为null） 方向 坐标系（全局或机身）。（x,y,z,角度）
        3.采用NED坐标系，x正方向北，0度偏航角指向正北
        4.重要注意，当指令没有提机身坐标系时，只需要提取东南西北指令，向北移动x变大，向南移动第一个参数x变小，向东移动y变大，向西移动y变小，
        4.重要注意，转向提到东南西北时，坐标系为全局，0度偏航角指向正北，90度偏航角指向正东，180度偏航角指向正南，270度偏航角指向正西
        4.重要注意，转向提到顺时针、向右偏航角变大，逆时针、向左偏航角变小
        5.当提到升高5m，提取移动，上升至5m，提取移动至
        5.默认起始坐标为（0,0, 0, 角度:0°），注意如果动作是移动至、上升至，不再需要使用初始坐标，向上移动至10m，只需要z = -10

        ## context(内容)
        {instruction}
         ## Response（回应示例）
        重要注意：移动至上升至描述，不再考虑原坐标，直接设为目标值
        重要注意：东南西北不考虑机身方向，向南移动x变小，向东移动y变大，向西移动y变小
        重要注意：移动动作，向前x为正，向后x为夫负，向右y为正，向左y为负
        重要注意：顺时针、向右偏航角变大，逆时针、向左偏航角变小，正北对应0，正东对应90，正南对应180，正西对应270
        中要注意：提到速度是需要提取速度参数
        1 - 原始位置（0, 0, 0, 角度:0°）。
        2 - 移动：3米 null null 向上 全局
        2 - 移动至：10米 null null 向上 全局
        3 - 移动 5米 null null 向北 全局 (5, 0, -10, 角度:0°)x变大
        4 - 移动 5米 null null 向西 全局 (5, -5, -10, 角度:0°)y变小
        5 - 移动：5米 null null 向后 机身 (-5, -5, -10, 角度:0°)
        7 - 移动：5米 1m/s 5s 向南 全局 (-10, 0, -10, 角度:0°)x变小
        6 - 转向：45度 null null 向右 机身 (0, 0, -10, 角度:45°)
        7 - 转向：270度 null null 正西 全局 (0, 0, -10, 角度:270°)
        飞机位置要根据动作改变
        """

        messages = [
            SystemMessage(content="你是一个专业的无人机指令解析专家，负责将自然语言指令转换为结构化的动作序列。"),
            HumanMessage(content=prompt)
        ]

        print("正在解析指令...")
        response = self.llm.invoke(messages)
        elements_list = response.content
        print("指令解析完成")
        print(f"解析结果: {elements_list}...")

        # 更新token消耗
        token_info = self._update_token_usage(response, state)

        print("🔄 Movement Extractor Agent 执行完成")
        print("=" * 60)

        return {
            "elements_list": elements_list,
            "current_agent": "code_generator",
            **token_info
        }

    def code_generator_agent(self, state: AgentState) -> dict:
        print("=" * 60)
        print("🔄 执行 Code Generator Agent")
        print("=" * 60)
        elements_list = state["elements_list"]
        print(f"动作序列: {elements_list}...")

        prompt = f"""
        ## Objective（目标）
        1. 将输入的动作指令用{config.language_name}语言的{config.lib_name}库函数给出代码。
        2. 在开头加入控制解锁和在结尾加入控制结束
        4。注意只有以机身坐标系前后左右用moveByVelocityBodyFrameAsync，全局坐标系东南西北使用moveToPositionAsync，移动参数根据动作解析结果转换
        5.转向函数默认偏向角误差1度
        ## context(内容)
        {elements_list}

        ## apis(参考api)
        {self.seed_code}
        
        注意：重点检查相对机身向前移动使用moveByVelocityBodyFrameAsync(1, 0, 0, 1)，moveByVelocityBodyFrameAsync(0, 1, 0, 1)是错误的向左移动
        moveByVelocityBodyFrameAsync(1, 0, 0, 15)的第一个参数是速度，最后一个参数是时间，表示以机身坐标系向前的方向以1m/s的速度，前进15s
        顺时针，向右旋转，角度变大；逆时针向左旋转角度变小
        速度为null时，一般设置为1m/s
        """

        messages = [
            SystemMessage(
                content="你是一个专业的无人机控制代码生成专家，负责将结构化动作序列转换为可执行的AirSim代码。默认起点是（0，0，0），注释只需要写出动作目的，不需要思考过程"),
            HumanMessage(content=prompt)
        ]

        print("正在生成代码...")
        response = self.llm.invoke(messages)
        code = response.content
        print("代码生成完成")
        print(f"生成的代码: {code}...")

        # 更新token消耗
        token_info = self._update_token_usage(response, state)

        code_output = {
            "version": 0,
            "code": code,
            "status": "PENDING",
            "description": None
        }

        print("🔄 Code Generator Agent 执行完成")
        print("=" * 60)

        return {
            "code": code,
            "code_output": code_output,
            "current_agent": "code_check_agent",
            **token_info
        }

    def code_check_agent(self, state: AgentState) -> dict:
        print("=" * 60)
        print("🔄 执行 Code Check Agent")
        print("=" * 60)
        elements_list = state["elements_list"]
        code_output = state["code_output"]
        print(f"检查代码版本: {code_output.get('version', 0)}")
        print(f"目标动作序列: {elements_list}...")

        prompt = f"""
        ## Objective（目标）
        给出代码的对应动作序列，回应格式只包括以下几个动作，不包含飞机状态信息：
        - 起飞。
        - 降落。
        - 移动： 距离 速度（没有提到即为null） 时间 方向 坐标系（全局或机身）。当前动作完成后的全局位置与朝向
        - 转向： 角度 速度（可以为null） 方向 时间 坐标系（全局或机身）。
        注意：moveByVelocityBodyFrameAsync(1, 0, 0, 15)的第一个参数是速度1m/s，最后一个参数是时间15s，表示以机身坐标系向前的方向以1m/s的速度，前进15s,共15m
        当指令中没有提机身坐标系时，只需要提取东南西北上下指令，向北移动x变大，向东移动y变大
        注意 moveByVelocityBodyFrameAsync(0, 5, 0, 5)移动5*5 =25m
        client.moveToZAsync(-10, 1).join()的第一个参数是距离，代表移动10m第二个参数是速度参数，代表1m/s,
        ## Context
        给出{code_output['code']}对应的动作序列
        ## Response（回应示例）
            重要注意：moveByVelocityBodyFrameAsync函数的移动距离需要用速度*时间所得，z方向速度一般为零
            1 - 起飞。
            2 - 移动：3米 null null 向上 全局 
            3 - 移动：5米 1米/秒 5s 向后 机身
            4 - 转向：45度 null null 向右 机身
        """

        messages = [
            SystemMessage(content="你是一个专业的代码审核专家，负责检查生成的代码是否符合目标动作序列。"),
            HumanMessage(content=prompt)
        ]

        print("正在提取代码动作序列...")
        response = self.llm.invoke(messages)
        code_elements_list = response.content
        print("动作序列提取完成")
        print(f"代码动作序列: {code_elements_list}...")

        # 第一次LLM调用的token消耗
        token_info_1 = self._update_token_usage(response, state)

        prompt = f"""
        目标动作序列：{elements_list}
        当前动作序列：{code_elements_list}
        首先提取移动动作的速度序列，速度通常以米每秒作为单位，转向、起飞、降落等动作不参与比较,目标动作序列速度为null的判断为正确
        举例
        client.moveByVelocityBodyFrameAsync(0, 5, 0, 5).join()，代表向右移动了5*5=25m，目标是5m，应修改为1*5 = 5，代码改正为client.moveByVelocityBodyFrameAsync(0, 1, 0, 5).join()
        moveToZAsync(-8, 1)代表以1m/s的速度运动8m
        
        逐条比较速度和距离，分析是否正确，不关注方向和坐标
        ##response
        格式要求：
        1.列出具体的错误的点
        2.逐条分析，给出最后结论，正确或者错误，
        重要注意：目标动作序列速度为null的，当前目标速度为多少，都是默认正确
        示例：
        1.动作1：目标序列速度为null，正确
        2.动作2：目标序列移动3m，但当前序列移动15m
        3.动作3: 目标序列速度为null，正确
        结果正确/错误
        """

        messages = [
            HumanMessage(content=prompt)
        ]

        print("正在比较动作序列...")
        response = self.llm.invoke(messages)
        comparison_result = response.content
        print("动作序列比较完成")
        print(f"比较结果: {comparison_result}")

        # 第二次LLM调用的token消耗
        token_info_2 = self._update_token_usage(response, state)

        # 检查代码是否一致
        import re

        # 提取所有的"正确"或"错误"
        pattern = r'(正确|错误)'
        all_judgments = re.findall(pattern, comparison_result)

        if not all_judgments:
            # 如果没提取到任何判断词，默认为错误，触发修正流程
            code_consistent = False
            final_judgment = "未知"
            print("⚠️ 未提取到判断词，默认为错误，需要修正")
        else:
            # 以最后一个判断词为准
            final_judgment = all_judgments[-1]
            code_consistent = (final_judgment == "正确")
            print(f"提取到所有判断词: {all_judgments}")
            print(f"最终判断: {final_judgment}")

        print(f"代码是否一致: {code_consistent}")

        # 关键：根据一致性检查结果，决定下一个节点
        next_agent = "code_correct_agent" if not code_consistent else "supervisor"
        print(f"下一步路由: {next_agent}")

        print("🔄 Code Check Agent 执行完成")
        print("=" * 60)

        # 合并两次调用的token信息
        total_token_info = {
            "total_tokens": token_info_1["total_tokens"] + token_info_2["total_tokens"],
            "prompt_tokens": token_info_1["prompt_tokens"] + token_info_2["prompt_tokens"],
            "completion_tokens": token_info_1["completion_tokens"] + token_info_2["completion_tokens"]
        }

        return {
            "code_consistent": code_consistent,
            "code_elements_list": code_elements_list,
            "comparison_result": comparison_result,
            "current_agent": next_agent,
            **total_token_info
        }

    def code_correct_agent(self, state: AgentState) -> dict:
        print("=" * 60)
        print("🔄 执行 Code Correct Agent")
        print("=" * 60)
        elements_list = state["elements_list"]
        code_output = state["code_output"]
        iteration = state["iteration"]
        comparison_result = state["comparison_result"]
        print(f"当前代码版本: {code_output.get('version', 0)}")
        print(f"当前迭代次数: {iteration}")
        print(f"不一致原因: {comparison_result}")

        # 提取不一致的原因
        prompt = f"""
        提取不一致的原因，只关注速度和举距离相关，与坐标系、方向相关的忽略
        {comparison_result}
        """

        messages = [
            SystemMessage(content="你是一个专业的代码修改专家。"),
            HumanMessage(content=prompt)
        ]

        print("正在分析不一致原因...")
        response = self.llm.invoke(messages)
        cause = response.content
        print("原因分析完成")
        print(f"错误原因: {cause}")

        # 第一次LLM调用的token消耗
        token_info_1 = self._update_token_usage(response, state)

        # 修正代码
        prompt = f"""
        ##object
        对代码按照以下流程修改
        当前代码{code_output['code']}
        目标动作序列{elements_list}
        错误原因{cause}
        请给出修改后的代码,同时修改速度和时间,距离保持不变，最后给出完整代码

        ## 修改示例
        原代码client.moveByVelocityBodyFrameAsync(3, 0, 0, 1).join()，当前速度3m/s,目标速度1m/s
        修改后client.moveByVelocityBodyFrameAsync(1, 0, 0, 3).join()
        完整代码
        """

        messages = [
            SystemMessage(content="你是一个专业的代码修正专家。"),
            HumanMessage(content=prompt)
        ]

        print("正在修正代码...")
        response = self.llm.invoke(messages)
        code_new = response.content
        print("代码修正完成")
        print(f"修正后的代码: {code_new}...")

        # 第二次LLM调用的token消耗
        token_info_2 = self._update_token_usage(response, state)

        # 更新版本号
        new_version = code_output.get('version', 0) + 1

        # 构造新的输出字典
        new_code_output = {
            **code_output,  # 继承旧信息
            "version": new_version,
            "code": code_new
        }

        print(f"修正后代码版本: {new_version}")

        print("🔄 Code Correct Agent 执行完成")
        print("=" * 60)

        # 合并两次调用的token信息
        total_token_info = {
            "total_tokens": token_info_1["total_tokens"] + token_info_2["total_tokens"],
            "prompt_tokens": token_info_1["prompt_tokens"] + token_info_2["prompt_tokens"],
            "completion_tokens": token_info_1["completion_tokens"] + token_info_2["completion_tokens"]
        }

        return {
            "code_output": new_code_output,
            "code": code_new,
            "iteration": iteration + 1,
            "current_agent": "code_check_agent",
            **total_token_info
        }

    def supervisor_agent(self, state: AgentState) -> dict:
        print("=" * 60)
        print("🔄 执行 Supervisor Agent")
        print("=" * 60)
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        code_consistent = state.get("code_consistent", False)
        print(f"当前迭代次数: {iteration}")
        print(f"最大迭代次数: {max_iterations}")
        print(f"代码是否一致: {code_consistent}")

        # 打印最终token消耗统计
        print("\n" + "=" * 60)
        print("📊 最终Token消耗统计")
        print("=" * 60)
        print(f"   - 总Prompt Tokens: {state.get('prompt_tokens', 0)}")
        print(f"   - 总Completion Tokens: {state.get('completion_tokens', 0)}")
        print(f"   - 总Tokens: {state.get('total_tokens', 0)}")
        print("=" * 60)

        print("✅ 代码已一致或进入最终检查，准备结束流程")
        print("🔄 Supervisor Agent 执行完成")
        print("=" * 60)

        return {
            "current_agent": "end"
        }

    def should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        current_agent = state["current_agent"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        code_consistent = state.get("code_consistent", False)

        # 如果检查结果一致，直接结束
        if code_consistent:
            print("=" * 60)
            print("🔚 检查结果一致，流程结束")
            print("=" * 60)
            return "end"

        # 如果修正次数超限，结束
        if iteration >= max_iterations:
            print("=" * 60)
            print(f"🔚 达到最大迭代次数 ({max_iterations})，流程结束")
            print("=" * 60)
            return "end"

        return "continue"


# --- 构建图 ---
def create_drone_code_graph():
    generator = DroneCodeGenerator()

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("movement_extractor", generator.movement_extractor_agent)
    workflow.add_node("code_generator", generator.code_generator_agent)
    workflow.add_node("code_check_agent", generator.code_check_agent)
    workflow.add_node("code_correct_agent", generator.code_correct_agent)
    workflow.add_node("supervisor", generator.supervisor_agent)
    #
    # 设置入口
    workflow.set_entry_point("movement_extractor")
    #
    # 定义基础边
    workflow.add_edge("movement_extractor", "code_generator")
    workflow.add_edge("code_generator", "code_check_agent")

    # 添加条件边：根据检查结果动态路由
    def route_after_check(state: AgentState) -> str:
        return state["current_agent"]

    workflow.add_conditional_edges(
        "code_check_agent",
        route_after_check,
        {
            "code_correct_agent": "code_correct_agent",
            "supervisor": "supervisor"
        }
    )

    # 定义修正后的边：总是回到检查
    workflow.add_edge("code_correct_agent", "code_check_agent")

    # 定义 Supervisor 的边
    workflow.add_edge("supervisor", END)

    return workflow.compile()


# --- 运行函数 ---
def run_langgraph_agent(instruction: str, max_iterations: int = 3):
    graph = create_drone_code_graph()

    initial_state = {
        "messages": [],
        "instruction": instruction,
        "elements_list": "",
        "code": "",
        "code_output": {},
        "current_agent": "movement_extractor",
        "iteration": 0,
        "max_iterations": max_iterations,
        "code_consistent": False,
        "code_elements_list": "",
        "comparison_result": "",
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0
    }

    result = graph.invoke(initial_state)

    return result


if __name__ == "__main__":
    instruction = "起飞至10m高度，向右飞行5m，顺时针旋转45度，向前飞行10m"
    result = run_langgraph_agent(instruction)

    print("\n" + "=" * 30)
    print("🎉 最终生成结果")
    print("=" * 30)
    print(f"迭代次数: {result['iteration']}")
    print(f"最终代码版本: {result['code_output']['version']}")
    print(f"代码是否一致: {result.get('code_consistent', False)}")
    print(f"总Token消耗: {result.get('total_tokens', 0)}")
    print(f"  - Prompt Tokens: {result.get('prompt_tokens', 0)}")
    print(f"  - Completion Tokens: {result.get('completion_tokens', 0)}")
    print(f"生成的代码:\n{result['code_output']['code']}")

