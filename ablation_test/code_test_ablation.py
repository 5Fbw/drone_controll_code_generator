import json
import os
import sys
import argparse

# 添加项目根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 只需要导入 test_agent，不需要 main 和 LLm_provider
from adapt_langgraph.test_agent import test_agent
import re
def get_python_code(code_origin):
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
def run_test(input_file: str, output_file: str, model_name: str = "qwen3-32b", start_idx: int = 0, end_idx: int = None):
    """
    读取输入文件，生成代码，并保存到输出文件

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        model_name: 基座大模型名称
        start_idx: 起始序号（包含）
        end_idx: 截止序号（不包含）
    """
    print(f"读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    total_cases = len(test_cases)
    print(f"共读取 {total_cases} 个测试用例")
    print(f"使用模型: {model_name}")
    print(f"处理范围: 从序号 {start_idx} 到 {end_idx if end_idx is not None else total_cases}")

    # 计算实际处理的用例范围
    if end_idx is None:
        end_idx = total_cases

    # 确保索引在有效范围内
    start_idx = max(0, start_idx)
    end_idx = min(total_cases, end_idx)

    # 提取要处理的测试用例
    cases_to_process = test_cases[start_idx:end_idx]
    print(f"实际处理 {len(cases_to_process)} 个测试用例")

    # 移除了 LLm_provider 初始化，因为 test_agent 内部处理

    success_count = 0
    fail_count = 0

    # Token 统计变量
    total_tokens_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, test_case in enumerate(cases_to_process, start=start_idx):
        case_id = test_case.get('case_id', f'case_{i}')
        instruction = test_case.get('instruction', '')

        print(f"\n[{i + 1}/{total_cases}] 处理用例: {case_id}")
        print(f"  指令: {instruction}")

        try:
            # 调用 test_agent
            result = test_agent(instruction, model_name=model_name)

            # 提取代码
            code = result['code_output']['code']

            # 提取 token 信息
            current_tokens = result.get('total_tokens', 0)
            current_prompt = result.get('prompt_tokens', 0)
            current_completion = result.get('completion_tokens', 0)

            # 更新 test_case 字段
            test_case['code'] = get_python_code(code)
            test_case['model'] = model_name

            # 新增 token 字段
            test_case['token'] = current_tokens
            test_case['prompt_tokens'] = current_prompt
            test_case['completion_tokens'] = current_completion

            # 累加统计
            total_tokens_count += current_tokens
            total_prompt_tokens += current_prompt
            total_completion_tokens += current_completion

            success_count += 1
            print(f"  状态: 成功 (Token: {current_tokens})")

        except Exception as e:
            fail_count += 1
            test_case['code'] = f"Error: {str(e)}"
            test_case['model'] = model_name
            # 错误时 token 设为 0 或 null
            test_case['token'] = 0
            test_case['prompt_tokens'] = 0
            test_case['completion_tokens'] = 0
            print(f"  状态: 失败 - {str(e)}")

    print(f"\n处理完成: 成功 {success_count}, 失败 {fail_count}")

    # 打印总 Token 统计
    if success_count > 0:
        print(f"\n{'=' * 60}")
        print(f"📊 总Token消耗统计")
        print(f"{'=' * 60}")
        print(f"   - 总Tokens: {total_tokens_count}")
        print(f"   - 总Prompt Tokens: {total_prompt_tokens}")
        print(f"   - 总Completion Tokens: {total_completion_tokens}")
        print(f"   - 平均Tokens/用例: {total_tokens_count // success_count}")
        print(f"{'=' * 60}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {output_file}")


def main_cli():
    parser = argparse.ArgumentParser(description='无人机控制代码生成测试 (Agent模式)')
    parser.add_argument('--input', type=str, default='../test/data.json',
                        help='输入JSON文件路径')
    parser.add_argument('--output', type=str, default='output_qwen3_32b_action.json',
                        help='输出JSON文件路径')
    parser.add_argument('--model', type=str, default='qwen3-32b',
                        help='基座大模型名称')
    parser.add_argument('--start', type=int, default=0,
                        help='起始序号（包含）')
    parser.add_argument('--end', type=int, default=None,
                        help='截止序号（不包含）')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    model_name = args.model
    start_idx = args.start
    end_idx = args.end

    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    run_test(input_path, output_path, model_name, start_idx, end_idx)


if __name__ == '__main__':
    main_cli()
