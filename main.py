import code_generator
import movement_extractor
import code_checker
import re
import LLm_provider
def run(llm_provider,instruction: str):
    """
    根据给定的自然语言指令生成无人机控制代码并进行检验

    Args:
        instruction: 自然语言描述的无人机飞行指令

    Returns:
        code_output: 包含生成代码的字典，包含version、code、status、description字段
    """
    elements_list = movement_extractor.movement_extract(llm_provider,instruction)
    code_output = code_generator.code_generate(llm_provider,elements_list)
    code_output = code_generator.check(elements_list, code_output)
    return code_output


def run_without_agent(llm_provider,instruction: str):
    """
    直接根据自然语言指令生成无人机控制代码（不经过agent解析和检验）

    Args:
        instruction: 自然语言描述的无人机飞行指令

    Returns:
        code_output: 包含生成代码的字典，包含version、code、status、description字段
    """
    code_output = code_generator.code_generate_base(llm_provider,instruction)
    return code_output

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
if __name__ == '__main__':
    # instruction = "起飞并上升5米。你应该以5米边长的正方形模式飞行，通过向前移动并在每个角落向右转实现"
    # instruction = "起飞，以每秒1米的速度升空至10米高度。前进5米,然后转向正西，并以每秒1米的速度前进10米。然后降落"
    instruction = "起飞至10m高度，向右飞行5m，顺时针旋转45度，向前飞行10m"
    llm_provider = LLm_provider.LLMProvider()
    result = run(llm_provider,instruction)
    # result = run_without_agent(llm_provider,instruction)
    # result = get_python_code(result)
    # elements_list = movement_extractor.movement_extract(llm_provider,instruction)
    print("生成完成")
    # print(f"版本: {result.get('version')}")
    # print(f"状态: {result.get('status')}")
    print(f"代码:\n{result.get('code')}")
