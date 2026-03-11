import math

# -*- coding: utf-8 -*-
from config import Status
import re
import subprocess
import sys
import ast
import airsim
code = '''
首先判断是否一致：\n\n当前代码的动作序列：\n1. 起飞\n2. 移动到Z=-10（向上移动10米）- 全局坐标\n3. 向前移动5米 - 机身坐标，速度5m/s，时间1s\n4. 转向270度（向左转90度）\n5. 向前移动10米 - 机身坐标，速度10m/s，时间1s\n6. 降落\n\n目标动作序列：\n1. 起飞\n2. 移动：10米，1米/秒，10s，向上，全局\n3. 移动：5米，向前，机身\n4. 转向：90度，向左，机身\n5. 移动：10米，1米/秒，10s，向前，机身\n6. 降落\n\n不一致，需要修改。主要问题是第3步和第5步的速度参数需要调整以保持距离不变但符合目标速度要求。\n\n```python\nimport airsim\n\nclient = airsim.MultirotorClient()\nclient.confirmConnection()\nclient.enableApiControl(True)\nclient.armDisarm(True)\n\nclient.takeoffAsync().join()\n\nclient.moveToZAsync(-10, 1).join()\n\nclient.moveByVelocityBodyFrameAsync(1, 0, 0, 5).join()\n\nclient.rotateToYawAsync(270).join()\n\nclient.moveByVelocityBodyFrameAsync(1, 0, 0, 10).join()\n\nclient.landAsync().join()\n\nclient.armDisarm(False)\nclient.enableApiControl(False)\n```
'''
code_output ={
        "version": 0,
        "code":code,
        "status":Status.PENDING,
        "description":None
    }

def get_python_code(code_output):
    code_origin = code_output['code']
    if code_origin is None:
        print("警告：传入的代码内容为空 (None)，无法提取。")
        return ""

        # 2. 确保输入是字符串类型（如果传的是字典或其他对象，先转成字符串）
    if not isinstance(code_origin, str):
        print("警告：传入的类型不是字符串，而是 {}，正在尝试转换...".format(type(code_origin)))
        code_origin = str(code_origin)
    pattern = r"```python\s*(.*?)\s*```"

    # Find all matches using re.DOTALL flag to match across multiple lines
    matches = re.findall(pattern, code_origin, re.DOTALL)
    # Clean up each code block (remove leading/trailing whitespace)
    codes_only = [block.strip() for block in matches]
    code_only = codes_only[0]
    return code_only

def syntax_check(code):
    checker_code = f'''
try:
    compile({repr(code)}, "<string>", "exec")
    print("OK")
except SyntaxError as e:
    print(f"SyntaxError: {{e}}")
    exit(1)
except Exception as e:
    print(f"Error: {{e}}")
    exit(1)
'''
    result = subprocess.run(
        [sys.executable, "-c", checker_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    print("result.returncode")
    print(result.returncode)
    print("result.stdout")
    print(result.stdout)
    print("result.stderr")
    print(result.stderr)
    return result.returncode == 0


def extract_velocity_params(code):
    """
    从 AirSim 代码中提取速度参数 (vx, vy, vz)
    :param code_str: 完整的代码字符串
    :return: 包含所有速度参数的列表
    """
    def _get_value(node):
        """
        辅助函数：从 AST 芧点中提取数值
        """
        # --- 情况 1: 处理负数 (例如 -1.5) ---
        if isinstance(node, ast.UnaryOp):
            # 检查操作符是否为 USub (Unary Subtraction, 即负号)
            if isinstance(node.op, ast.USub):
                # 递归获取操作数 的值
                # 操作数通常是 Constant 或 Num
                value = _get_value(node.operand)
                if value is not None:
                    return -value  # 取反
            # 检查操作符是否为 UAdd (正号, 如 +5，虽然很少见)
            elif isinstance(node.op, ast.UAdd):
                return _get_value(node.operand)

        # --- 情况 2: 处理 Python 3.8 之前的旧版数字节点 ---
        elif isinstance(node, ast.Num):
            return node.n

        # --- 情况 3: 处理 Python 3.8+ 的新版常量节点 ---
        elif isinstance(node, ast.Constant):
            # 确保是数字类型
            if isinstance(node.value, (int, float)):
                return node.value

        return None
    V_PARAM_MAP = {
        "moveByVelocityZAsync": [0, 1],  # 提取 vx, vy
        "moveToZAsync": [1],
        "moveByVelocityBodyFrameAsync": [0, 1, 2],  # 提取 vx, vy, vz
        "moveByVelocityAsync": [0, 1, 2],  # 提取 vx, vy, vz
    }
    H_PARAM_MAP ={
        "moveByVelocityZAsync": [2],
        "moveToZAsync":[0],
        "moveToPositionAsync":[2],
    }
    velocity_params = []
    height_params = []
    try:
        tree = ast.parse(code)
        extracted_results = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # 获取函数名
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    # print("=========")
                    # print(func_name)
                # 2. 检查该函数是否在我们的映射表中
                if func_name in V_PARAM_MAP:
                    # 3. 从映射表中获取该函数需要提取的参数序号
                    indices_to_extract = V_PARAM_MAP[func_name]

                    # 4. 根据序号提取参数值
                    values = []
                    v_square = 0
                    for idx in indices_to_extract:
                        if idx < len(node.args):
                            val = _get_value(node.args[idx])
                            if val is not None:
                                v_square = v_square + val ** 2
                                values.append(val)

                    # 5. 只有当提取到的值数量等于需要的数量时才记录
                    if len(values) == len(indices_to_extract):
                        velocity_params.append(math.sqrt(v_square))
                        # result = {
                        #     "function": func_name,
                        #     "line": node.lineno,
                        #     "values": values
                        # }
                        # extracted_results.append(result)
            if isinstance(node, ast.Call):
                # 获取函数名
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    # print("=========")
                    # print(func_name)
                # 2. 检查该函数是否在我们的映射表中
                if func_name in H_PARAM_MAP:
                    # 3. 从映射表中获取该函数需要提取的参数序号
                    indices_to_extract = H_PARAM_MAP[func_name]
                    # print(indices_to_extract)
                    # 4. 根据序号提取参数值
                    for idx in indices_to_extract:
                        if idx < len(node.args):
                            val = _get_value(node.args[idx])
                            if val is not None:
                                height_params.append(- val)

                    # # 5. 只有当提取到的值数量等于需要的数量时才记录
                    # if len(values) == len(indices_to_extract):
                    #     velocity_params.append(math.sqrt(v_square))
                        # result = {
                        #     "function": func_name,
                        #     "line": node.lineno,
                        #     "values": values
                        # }
                        # extracted_results.append(result)
        result_map = {
            "height":height_params,
            "velocity":velocity_params,
        }
        return result_map

    except SyntaxError as e:
        print("❌ 代码解析失败: {}".format(e))
        return []
    except Exception as e:
        print("⚠️ 发生意外错误: {}".format(e))
        return []



# code = get_python_code(code_output)
# flag = syntax_check(code)
# result = extract_velocity_params(code)
# print(result)
# print("===========")
# print(code)