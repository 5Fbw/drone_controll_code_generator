import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapt_langgraph.drone_code_agent import run_langgraph_agent


def test_simple_instruction():
    print("=" * 50)
    print("测试1：简单指令")
    print("=" * 50)
    
    instruction = "起飞至10m高度"
    result = run_langgraph_agent(instruction, max_iterations=2)
    
    print(f"指令: {instruction}")
    print(f"迭代次数: {result['iteration']}")
    print(f"代码版本: {result['code_output']['version']}")
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
    print(f"生成的代码:\n{result['code_output']['code']}")
    print()


def test_multistep_instruction():
    print("=" * 50)
    print("测试3：多步动作指令")
    print("=" * 50)
    
    instruction = "起飞至10m高度，向右飞行5m，顺时针旋转45度，向前飞行10m"
    result = run_langgraph_agent(instruction, max_iterations=3)
    
    print(f"指令: {instruction}")
    print(f"迭代次数: {result['iteration']}")
    print(f"代码版本: {result['code_output']['version']}")
    print(f"生成的代码:\n{result['code_output']['code']}")
    print()


if __name__ == "__main__":
    test_simple_instruction()
    test_complex_instruction()
    test_multistep_instruction()
    
    print("=" * 50)
    print("所有测试完成")
    print("=" * 50)
