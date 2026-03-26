import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapt_langgraph.drone_code_agent import run_langgraph_agent


def test_simple_instruction():
    print("=" * 50)
    print("测试1：简单指令")
    print("=" * 50)
    
    instruction = "起飞至10m高度，转向正南"
    result = run_langgraph_agent(instruction, max_iterations=2)
    
    print(f"指令: {instruction}")
    print(f"迭代次数: {result['iteration']}")
    print(f"代码版本: {result['code_output']['version']}")
    print(f"代码是否一致: {result.get('code_consistent', False)}")
    print(f"生成的代码:\n{result['code_output']['code']}")
    print()

def test_complex_instruction():
    print("=" * 50)
    print("测试2：复杂指令")
    print("=" * 50)
    
    instruction = "起飞，以每秒1米的速度升空至10米高度。前进5米,然后转向正西，并以每秒1米的速度前进10米。然后降落。"
    result = run_langgraph_agent(instruction, max_iterations=3)
    
    print(f"指令: {instruction}")
    print(f"迭代次数: {result['iteration']}")
    print(f"代码版本: {result['code_output']['version']}")
    print(f"代码是否一致: {result.get('code_consistent', False)}")
    print(f"生成的代码:\n{result['code_output']['code']}")
    print()

def test_multistep_instruction():
    print("=" * 50)
    print("测试3：多步动作指令")
    print("=" * 50)
    instruction = "起飞到5m，在无人机机体坐标系的YZ平面内，沿相对于水平轴30度角的右上方向飞行10米。"
    result = run_langgraph_agent(instruction, max_iterations=3)
    
    print(f"指令: {instruction}")
    print(f"迭代次数: {result['iteration']}")
    print(f"代码版本: {result['code_output']['version']}")
    print(f"代码是否一致: {result.get('code_consistent', False)}")
    print(f"生成的代码:\n{result['code_output']['code']}")
    print()
def test_agent(query):
    result = run_langgraph_agent(query, max_iterations=3)
    print(f"指令: {query}")
    print(f"迭代次数: {result['iteration']}")
    print(f"代码版本: {result['code_output']['version']}")
    print(f"代码是否一致: {result.get('code_consistent', False)}")
    print(f"生成的代码:\n{result['code_output']['code']}")
    # 添加 token 消耗信息
    print(f"\n📊 Token消耗统计:")
    print(f"   - 总Tokens: {result.get('total_tokens', 0)}")
    print()
    return result
if __name__ == "__main__":
    # test_agent(" 起飞至8m高度,转向正西,向前移动5m")
    # test_agent(" 起飞至8m高度,逆时针旋转90度,向正南移动5m")
    # test_agent("起飞，以每秒1米的速度升空至10米高度。前进5米,然后转向正西，并以每秒1米的速度前进10米。然后降落。")
    # test_agent(" 起飞到5m高度，向东移动，再向北移动，画出一个边长5m的正方形轨迹。")
    test_agent("起飞并上升5米。你应该以5米边长的正方形模式飞行，通过向前移动并在每个角落向右转实现")
    # test_simple_instruction()
    # test_complex_instruction()
    # test_multistep_instruction()
    
    print("=" * 50)
    print("所有测试完成")
    print("=" * 50)
