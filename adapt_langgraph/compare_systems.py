import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import LLm_provider
import main
from adapt_langgraph.drone_code_agent import run_langgraph_agent


def compare_systems(instruction: str):
    print("=" * 80)
    print(f"测试指令: {instruction}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("原系统（main.py）")
    print("=" * 80)
    llm_provider = LLm_provider.LLMProvider()
    result_original = main.run(llm_provider, instruction)
    print(f"代码版本: {result_original.get('version')}")
    print(f"生成的代码:\n{result_original.get('code')}")
    
    print("\n" + "=" * 80)
    print("LangGraph系统")
    print("=" * 80)
    result_langgraph = run_langgraph_agent(instruction, max_iterations=3)
    print(f"迭代次数: {result_langgraph['iteration']}")
    print(f"代码版本: {result_langgraph['code_output']['version']}")
    print(f"生成的代码:\n{result_langgraph['code_output']['code']}")
    
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    print(f"原系统代码版本: {result_original.get('version')}")
    print(f"LangGraph系统代码版本: {result_langgraph['code_output']['version']}")
    print(f"LangGraph系统迭代次数: {result_langgraph['iteration']}")


if __name__ == "__main__":
    instruction1 = "起飞至10m高度"
    instruction2 = "起飞，以每秒1米的速度升空至10米高度。前进5米,然后转向正西，并以每秒1米的速度前进10米。然后降落。"
    instruction3 = "起飞至10m高度，向右飞行5m，顺时针旋转45度，向前飞行10m"
    
    print("\n\n")
    compare_systems(instruction1)
    print("\n\n")
    compare_systems(instruction2)
    print("\n\n")
    compare_systems(instruction3)
