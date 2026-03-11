import json
import os
import sys
import argparse
import openpyxl


def excel_to_json(input_file: str, output_file: str):
    """
    将Excel文件转换为JSON文件

    Args:
        input_file: 输入Excel文件路径
        output_file: 输出JSON文件路径
    """
    print(f"读取输入文件: {input_file}")
    
    # 加载Excel文件
    wb = openpyxl.load_workbook(input_file)
    ws = wb.active

    # 获取表头
    headers = []
    for col in range(1, ws.max_column + 1):
        header = ws.cell(row=1, column=col).value
        if header:
            headers.append(header)

    print(f"表头: {headers}")

    # 读取数据
    data = []
    for row in range(2, ws.max_row + 1):
        row_data = {}
        for col, header in enumerate(headers, 1):
            value = ws.cell(row=row, column=col).value
            
            # 尝试将字符串转换为列表（如果是位置数据）
            if value and isinstance(value, str):
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = eval(value)
                    except:
                        pass
            
            row_data[header] = value if value is not None else ""
        
        # 只添加非空行
        if any(row_data.values()):
            data.append(row_data)

    print(f"共读取 {len(data)} 条记录")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {output_file}")


def main_cli():
    parser = argparse.ArgumentParser(description='Excel转JSON工具')
    parser.add_argument('--input', type=str, default='output_qwen3_32b_no.xlsx',
                        help='输入的JSON文件路径')

    parser.add_argument('--output', type=str, default='output_qwen3_32b_no.json',
                        help='输出的Excel文件路径')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        sys.exit(1)

    excel_to_json(input_path, output_path)


if __name__ == '__main__':
    main_cli()
