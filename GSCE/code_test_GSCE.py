import json
import os
import sys
import argparse
import re

# 添加项目根目录路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# [修改点] 导入 LLm_provider 和 GSCE
import LLm_provider
from GSCE import GSCE


def get_python_code(code_origin):
    """
    从LLM返回的文本中提取Python代码块
    """
    if code_origin is None:
        print("警告：传入的代码内容为空，无法提取。")
        return ""

    if not isinstance(code_origin, str):
        print("警告：传入的类型不是字符串，正在尝试转换...")
        code_origin = str(code_origin)

    pattern = r"```(?:python)?\s*(.*?)\s*```"
    # 使pattern = r"python\s * (.* ?)\s * ```"用 re.DOTALL 标志进行跨行匹配
    matches = re.findall(pattern, code_origin, re.DOTALL)

    if not matches:
        # 如果没有找到标记，尝试直接返回原文
        return code_origin.strip()

    # 清理并合并所有匹配到的代码块
    codes_only = [block.strip() for block in matches]
    code_only = "\n\n".join(codes_only)
    return code_only


def run_test(input_file: str, output_file: str, model_name: str, start_idx: int = 0, end_idx: int = None):

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
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    total_cases = len(test_cases)
    print(f"共读取 {total_cases} 个测试用例")
    print(f"使用模型: {model_name}")

    # 计算实际处理的用例范围
    if end_idx is None:
        end_idx = total_cases

    # 确保索引在有效范围内
    start_idx = max(0, start_idx)
    end_idx = min(total_cases, end_idx)

    # 提取要处理的测试用例
    cases_to_process = test_cases[start_idx:end_idx]
    print(f"实际处理 {len(cases_to_process)} 个测试用例 (索引 {start_idx} 到 {end_idx-1})")

    # [修改点] 初始化 LLM Provider，只初始化一次
    try:
        llm_provider = LLm_provider.LLMProvider(model_name=model_name)
    except Exception as e:
        print(f"初始化 LLMProvider 失败: {e}")
        return

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
        print(f" 指令: {instruction}")

        try:
            # [修改点] 接收 GSCE 返回的文本和 Token 信息
            response_text, usage_info = GSCE(llm_provider, instruction)

            # 提取代码
            code = get_python_code(response_text)

            # 更新 test_case 字段
            test_case['code'] = code
            test_case['model'] = model_name

            # [修改点] 提取并记录 Token 信息
            if usage_info:
                test_case['token'] = usage_info.total_tokens
                test_case['prompt_tokens'] = usage_info.prompt_tokens
                test_case['completion_tokens'] = usage_info.completion_tokens

                # 累加到总统计变量中
                total_tokens_count += usage_info.total_tokens
                total_prompt_tokens += usage_info.prompt_tokens
                total_completion_tokens += usage_info.completion_tokens
            else:
                test_case['token'] = 0
                test_case['prompt_tokens'] = 0
                test_case['completion_tokens'] = 0

            success_count += 1
            print(f" 状态: 成功")

        except Exception as e:
            fail_count += 1
            test_case['code'] = f"Error: {str(e)}"
            test_case['model'] = model_name
            test_case['token'] = 0
            test_case['prompt_tokens'] = 0
            test_case['completion_tokens'] = 0
            print(f" 状态: 失败 - {str(e)}")

    print(f"\n处理完成: 成功 {success_count}, 失败 {fail_count}")

    # 打印总 Token 统计
    if success_count > 0:
        print(f"\n{'=' * 60}")
        print(f"📊 批量处理完成")
        print(f"{'=' * 60}")
        print(f" - 成功数: {success_count}")
        print(f" - 失败数: {fail_count}")
        # [修改点] 打印实际的汇总消耗
        print(f" - 总 Prompt Tokens: {total_prompt_tokens}")
        print(f" - 总 Completion Tokens: {total_completion_tokens}")
        print(f" - 总 Tokens: {total_tokens_count}")
        print(f"{'=' * 60}")


    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {output_file}")
def main_cli():
    parser = argparse.ArgumentParser(description='无人机控制代码生成测试 (GSCE模式)')
    parser.add_argument('--input', type=str, default='../test/data.json',help='输入JSON文件路径')
    parser.add_argument('--output', type=str, default='output.json',help='输出JSON文件路径')
    # parser.add_argument('--model', type=str, default='qwen3-32b', help='基座大模型名称')
    parser.add_argument('--model', type=str, default='qwen3.5-397b-a17b',help='基座大模型名称')
    parser.add_argument('--start', type=int, default=16,help='起始序号（包含）')
    parser.add_argument('--end', type=int, default=None,help='截止序号（不包含，默认为None表示到结尾）')

    args = parser.parse_args()

    run_test(args.input, args.output, args.model, args.start, args.end)
if __name__ == '__main__':
    main_cli()