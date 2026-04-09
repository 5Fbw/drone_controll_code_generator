import airsim
import LLm_provider
import config
from config import Status

llm_provider = LLm_provider.LLMProvider()
code_name = "python"
lib_name = "airsim"
movements = '''
n--- Calling model: qwen3-coder-plus-2025-09-23 ---
Assistant: 1 - 起飞。
2 - 移动：10米 1米/秒 向上 全局 (0, 0, -10, 角度:0°)
3 - 移动：5米 null 向前 机身 (5, 0, -10, 角度:0°)
4 - 转向：90度 null 向左 机身 (5, 0, -10, 角度:270°)
5 - 移动：10米 1米/秒 向前 机身 (5, 10, -10, 角度:270°)
6 - 降落。
n--- End of response ---'''





code = '''
n--- Calling model: qwen3-coder-plus-2025-09-23 ---
Assistant: ```python
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

client.moveToZAsync(-10, 1).join()

client.moveByVelocityBodyFrameAsync(5, 0, 0, 1).join()

client.rotateToYawAsync(270).join()

client.moveByVelocityBodyFrameAsync(10, 0, 0, 1).join()

client.landAsync().join()

client.armDisarm(False)
client.enableApiControl(False)
```
n--- End of response ---

'''


now = {
    "height":[10],
    "velocity":[1, 5, 1]
}
res = security_constraints = {
    "max_velocity": 3,
    "max_height": 15
}
prompt = f'''
    当前无人机高度速度参数：{now}，高度为负代表地上，velocity列表代表n次不同速度，不是三维分量
    限制因素：{res}
    是否符合安全飞行规范，不符合的话说出原因，以json格式返回包含status（true或error），description（描述错误原因）
'''
# response = llm_provider.qwen_api(prompt, stream=True)

{
    "version":1,
    "code":"...",
    "status":"error",
    "description":"速度参数不符合安全飞行规范，velocity列表中的第二个值为5，超过了最大速度限制max_velocity: 3"
}

prompt = f'''
    错误原因：速度参数不符合安全飞行规范，velocity列表中的第二个值为5，超过了最大速度限制max_velocity: 3
    当前代码：{code}
    要求低于最大速度限制的不变，高于的降低速度为1或者2，前进距离不变
    给出修改后代码
'''
# response = llm_provider.qwen_api(prompt, stream=True)
def code_generate_base(llm_provider,model_name,input: str):
    prompt = f"""不经过思考直接给出用airsim函数实现{input}"""

    response = llm_provider.qwen_api(prompt, model_name,stream=False)
    code_output = {
        "version": 0,
        "code": {response},
        "status": Status.PENDING,
        "description": None
    }
    return code_output
def code_generate(llm_provider,elements_list: str):
    prompt = f"""
        ## Objective（目标）
        1. 将输入的动作指令用{config.language_name}语言的{config.lib_name}库函数给出代码。
        2. 在开头加入控制解锁和在结尾加入控制结束
        3. 直接给出代码，不包含其他任何内容。
        ## context(内容)
        {elements_list}

        ## apis(参考api)
        {config.seed_code}
        注意：重点检查相对机身向前移动使用moveByVelocityBodyFrameAsync(1, 0, 0, 1)，moveByVelocityBodyFrameAsync(0, 1, 0, 1)是错误的向左移动
        moveByVelocityBodyFrameAsync(1, 0, 0, 15)的第一个参数是速度，最后一个参数是时间，表示以机身坐标系向前的方向以1m/s的速度，前进15s
        顺时针，向右旋转，角度变大；逆时针向左旋转角度变小
        """


    response = llm_provider.qwen_api(prompt, stream=False)
    code_output = {
        "version": 0,
        "code":{response},
        "status":Status.PENDING,
        "description":None
    }
    return code_output

def check(elements_list,code_output):
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
    code_elements_list = llm_provider.qwen_api(prompt, stream=False)
    # print("当前代码动作序列")
    # print(code_elements_list)
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
    response = llm_provider.qwen_api(prompt, stream=False)
    # print("动作不同")
    # print(response)
    prompt=f'''
    提取结论部分
    '''
    cause = llm_provider.qwen_api(prompt, stream=False)
    # print("原因如下")

    prompt=f'''
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
    code_new = llm_provider.qwen_api(prompt, stream=False)
    # print(code_new)
    code_output['version'] = code_output['version'] + 1
    code_output['code'] = code_new
    return code_output
# elements_list = '''
# Assistant: 1 - 起飞。
# 2 - 移动：10米 1米/秒 10s 向上 全局 (0, 0, -10, 角度:0°)
# 3 - 移动：5米 null null 向前 机身 (5, 0, -10, 角度:0°)
# 4 - 转向：90度 null null 向左 机身 (5, 0, -10, 角度:270°)
# 5 - 移动：10米 1米/秒 10s 向前 机身 (5, -10, -10, 角度:270°)
# 6 - 降落。
# n--- End of response ---
# '''
# code = '''
#
# n--- Calling model: qwen3-coder-plus-2025-09-23 ---
# Assistant: ```python
# import airsim
# import time
#
# client = airsim.MultirotorClient()
# client.confirmConnection()
# client.enableApiControl(True)
# client.armDisarm(True)
#
# client.takeoffAsync().join()
#
# client.moveToZAsync(-10, 1).join()
# time.sleep(10)
#
# client.moveByVelocityBodyFrameAsync(5, 0, 0, 1).join()
#
# client.rotateToYawAsync(270, 5, 60).join()
#
# client.moveByVelocityBodyFrameAsync(10, 0, 0, 1).join()
# time.sleep(10)
#
# client.landAsync().join()
#
# client.enableApiControl(False)
# ```
# n--- End of response ---
#
# '''
# code_output ={
#         "version": 0,
#         "code":code,
#         "status":Status.PENDING,
#         "description":None
#     }
# # check(elements_list,code_output)
#
# input = "起飞，以每秒1米的速度升空至10米高度。前进5米，" \
#             "然后转向正西，并以每秒1米的速度前进10米。然后降落。"
#     # input = "起飞，转向正西，前进8米，右转向前移动10m。然后降落"
#     # elements_list = movement_extractor.movement_extract(input)
#     # code_output = code_generator.code_generate(elements_list)
#     # # # print(code_output)
#     # code_output = code_generator.check(elements_list,code_output)
# code_output = code_generate_base(input)