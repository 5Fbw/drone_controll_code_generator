from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
import config
import re


class AgentState(TypedDict):
    messages: list
    instruction: str
    elements_list: str
    code: str
    code_output: dict
    current_agent: str
    iteration: int
    max_iterations: int


class DroneCodeGenerator:
    def __init__(self, model_name: str = "qwen3-32b"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=config.api_key_new,
            base_url=config.base_url,
        )
        self.seed_code = config.seed_code
        
    def movement_extractor_agent(self, state: AgentState) -> AgentState:
        instruction = state["instruction"]
        
        prompt = f"""
        ## Objective（目标）
        1. 讲输入的指令拆解成单步动作指令。
        2. 给出距离、速度、坐标系（全局或机身），一般来说向前、向左之类的是机身，向西、向东、向北、向南是全局，向上、向下是全局，移动后的当前飞机全局位置与朝向。
        回应格式只包括以下几个动作：
        - 起飞。
        - 降落。
        - 移动： 距离 速度（没有提到即为null） 时间（没有提到即为null） 方向 坐标系（全局或机身）。（x,y,z,角度）当前动作完成后的全局位置与朝向
        - 移动至 位置 速度（没有提到即为null） 时间（没有提到即为null） 方向 坐标系（全局或机身）。（x,y,z,角度）
        - 转向： 角度 速度（可以为null） 时间（没有提到即为null） 方向 坐标系（全局或机身）。（x,y,z,角度）
        3.采用NED坐标系，x正方向北，0度偏航角指向正北
        4.重要！！！当指令中没有提机身坐标系时，只需要提取东南西北上下指令，向北移动x变大，向南移动x变小，向东移动y变大，向西移动y变小
        5.当提到升高5m，提取移动，上升至5m，提取移动至
        5.默认起始坐标为（startX, startY, startZ, 角度:0°），注意如果动作是移动至、上升至，不再需要使用初始坐标，向上移动至10m，只需要z = -10

        ## context(内容)
        {instruction}
         ## Response（回应示例）
          重要注意：移动至上升至描述，不再考虑原坐标，直接设为目标值
         重要注意：东南西北不考虑机身方向，向南移动x变小，向东移动y变大，向西移动y变小
        重要注意：顺时针、向右偏航角变大，逆时针、向左偏航角变小
        1 - 原始位置（startX, startY, startZ, 角度:0°）。
        2 - 移动：3米 null null 向上 全局 (startX, startY, startZ-3, 角度:0°)
        2 - 移动至：10米 null null 向上 全局 (startX, startY, -10, 角度:0°)
        3 - 移动 5米 null null 向北 全局 (startX+5, startY, -10, 角度:0°)
        4 - 移动 5米 null null 向西 全局 (startX+5, startY-5, -10, 角度:0°)
        5 - 移动：5米 1米/秒 5s 向后 机身 (startX, startY-5, -10, 角度:0°)
        6 - 移动：5米 1米/秒 5s 向东 全局 (startX, startY, -10, 角度:0°)
        7 - 移动：5米 1米/秒 5s 向南 全局 (startX-5, startY, -10, 角度:0°)
        6 - 转向：45度 null null 向右 机身 (startX, startY, -10, 角度:45°)
        飞机位置要根据动作改变
        """
        
        messages = [
            SystemMessage(content="你是一个专业的无人机指令解析专家，负责将自然语言指令转换为结构化的动作序列。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        elements_list = response.content
        
        return {
            **state,
            "elements_list": elements_list,
            "current_agent": "code_generator"
        }
    
    def code_generator_agent(self, state: AgentState) -> AgentState:
        elements_list = state["elements_list"]
        
        prompt = f"""
        ## Objective（目标）
        1. 将输入的动作指令用{config.language_name}语言的{config.lib_name}库函数给出代码。
        2. 在开头加入控制解锁和在结尾加入控制结束
        3. 直接给出代码，不包含其他任何内容。
        ## context(内容)
        {elements_list}

        ## apis(参考api)
        {self.seed_code}
        注意：重点检查相对机身向前移动使用moveByVelocityBodyFrameAsync(1, 0, 0, 1)，moveByVelocityBodyFrameAsync(0, 1, 0, 1)是错误的向左移动
        moveByVelocityBodyFrameAsync(1, 0, 0, 15)的第一个参数是速度，最后一个参数是时间，表示以机身坐标系向前的方向以1m/s的速度，前进15s
        顺时针，向右旋转，角度变大；逆时针向左旋转角度变小
        """
        
        messages = [
            SystemMessage(content="你是一个专业的无人机控制代码生成专家，负责将结构化动作序列转换为可执行的AirSim代码。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        code = response.content
        
        code_output = {
            "version": 0,
            "code": code,
            "status": "PENDING",
            "description": None
        }
        
        return {
            **state,
            "code": code,
            "code_output": code_output,
            "current_agent": "code_checker"
        }
    
    def code_checker_agent(self, state: AgentState) -> AgentState:
        elements_list = state["elements_list"]
        code_output = state["code_output"]
        iteration = state["iteration"]
        
        prompt = f"""
        ## Objective（目标）
        给出代码的对应动作序列，回应格式只包括以下几个动作：
        - 起飞。
        - 降落。
        - 移动： 距离 速度（没有提到即为null） 时间 方向 坐标系（全局或机身）。（x,y,z,角度）当前动作完成后的全局位置与朝向
        - 转向： 角度 速度（可以为null） 方向 时间 坐标系（全局或机身）。（x,y,z,角度）
        注意：moveByVelocityBodyFrameAsync(1, 0, 0, 15)的第一个参数是速度1m/s，最后一个参数是时间15s，表示以机身坐标系向前的方向以1m/s的速度，前进15s,共15m
        当指令中没有提机身坐标系时，只需要提取东南西北上下指令，向北移动x变大，向东移动y变大
        注意 moveByVelocityBodyFrameAsync(0, 5, 0, 5)移动5*5 =25m
        ## Context
        给出{code_output['code']}对应的动作序列
        ## Response（回应示例）
            重要注意：moveByVelocityBodyFrameAsync函数的移动距离需要用速度*时间所得，z方向速度一般为零
            1 - 起飞。
            2 - 移动：3米 null null 向上 全局 (0, 0, -3, 角度:0°)
            3 - 移动：5米 1米/秒 5s 向后 机身 (-5, 0, -3, 角度:0°)
            4 - 转向：45度 null null 向右 机身 (-5, 0, -3, 角度:45°)
        """
        
        messages = [
            SystemMessage(content="你是一个专业的代码审核专家，负责检查生成的代码是否符合目标动作序列。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        code_elements_list = response.content
        
        prompt = f'''
        目标动作序列：{elements_list}
        当前动作序列：{code_elements_list}
        首先提取移动动作的速度序列，速度通常以米每秒作为单位，转向、起飞、降落等动作不参与比较
        逐一是否一致，如果目标动作序列为null则随意，如果目标动作序列指定速度，则需要判断
        举例client.moveByVelocityBodyFrameAsync(0, 5, 0, 5).join()，代表向右移动了5*5=25m，目标是5m，应修改为1*5 = 5，代码改正为client.moveByVelocityBodyFrameAsync(0, 1, 0, 5).join()
        
        ##response
        存在1个不一致处
        1.动作3：目标序列速度为2m/s，但当前序列指定了5米/秒的速度
        2.动作1：目标序列移动3m，但当前序列移动15m
        '''
        
        messages = [
            SystemMessage(content="你是一个专业的代码审核专家。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        comparison_result = response.content
        
        prompt = f'''
        提取结论部分
        '''
        
        messages = [
            SystemMessage(content="你是一个专业的代码审核专家。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        cause = response.content
        
        prompt = f'''
        ##object
        首先判断是否一致，一致直接返回源代码
        否则对代码按照以下流程修改
        当前代码{code_output['code']}
        目标动作序列{elements_list}
        错误原因{cause}
        请给出修改后的代码,同时修改速度和时间距离保持不变
        
        ## 修改示例
        原代码client.moveByVelocityBodyFrameAsync(3, 0, 0, 1).join()，当前速度3m/s,目标速度1m/s
        修改后client.moveByVelocityBodyFrameAsync(1, 0, 0, 3).join()
        '''
        
        messages = [
            SystemMessage(content="你是一个专业的代码修正专家。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        code_new = response.content
        
        code_output['version'] = code_output['version'] + 1
        code_output['code'] = code_new
        
        return {
            **state,
            "code_output": code_output,
            "iteration": iteration + 1,
            "current_agent": "supervisor"
        }
    
    def supervisor_agent(self, state: AgentState) -> AgentState:
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        
        if iteration >= max_iterations:
            return {**state, "current_agent": "end"}
        
        return {**state, "current_agent": "code_checker"}
    
    def should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        current_agent = state["current_agent"]
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        
        if current_agent == "end" or iteration >= max_iterations:
            return "end"
        return "continue"


def create_drone_code_graph():
    generator = DroneCodeGenerator()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("movement_extractor", generator.movement_extractor_agent)
    workflow.add_node("code_generator", generator.code_generator_agent)
    workflow.add_node("code_checker", generator.code_checker_agent)
    workflow.add_node("supervisor", generator.supervisor_agent)
    
    workflow.set_entry_point("movement_extractor")
    
    workflow.add_edge("movement_extractor", "code_generator")
    workflow.add_edge("code_generator", "code_checker")
    workflow.add_edge("code_checker", "supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        generator.should_continue,
        {
            "continue": "code_checker",
            "end": END
        }
    )
    
    return workflow.compile()


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
        "max_iterations": max_iterations
    }
    
    result = graph.invoke(initial_state)
    
    return result


if __name__ == "__main__":
    instruction = "起飞至10m高度，向右飞行5m，顺时针旋转45度，向前飞行10m"
    result = run_langgraph_agent(instruction)
    
    print("生成完成")
    print(f"迭代次数: {result['iteration']}")
    print(f"代码版本: {result['code_output']['version']}")
    print(f"生成的代码:\n{result['code_output']['code']}")
