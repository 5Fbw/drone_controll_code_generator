import json
import os
import time
import argparse
import airsim
import math
from pathlib import Path

# 引入 path_eval 中的相关类
from path_eval import Path_eval_utils, TrajectoryAnalyzer


def get_latest_recording_path(base_path):
    """ 获取AirSim录制目录下最新生成的文件夹路径 """
    if not os.path.exists(base_path):
        return ""
    try:
        subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path)
                   if os.path.isdir(os.path.join(base_path, d))]
    except Exception:
        return ""
    if not subdirs:
        return ""
    # 按修改时间排序，获取最新的一个
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir


def process_trajectory_data(log_dir):
    """
    读取指定目录下的日志文件，提取关键点信息
    返回: 包含时间、偏航角、坐标的字典列表
    """
    # AirSim 默认录制文件名
    rec_file_path = os.path.join(log_dir, "airsim_rec.txt")

    if not os.path.exists(rec_file_path):
        print(f"  [警告] 未找到录制文件: {rec_file_path}")
        return []

    try:
        # 1. 解析轨迹数据
        utils = Path_eval_utils()
        path_array = utils.to_path_ponit_array(rec_file_path)

        if not path_array:
            print("  [警告] 轨迹数据为空")
            return []

        # 2. 分析提取关键点
        analyzer = TrajectoryAnalyzer()
        key_points = analyzer.extract_key_points(path_array)

        # 3. 格式化数据：仅保留时间、偏航角、坐标
        formatted_points = []
        for kp in key_points:
            pos = kp.get('position', (0, 0, 0))
            formatted_points.append({
                "timestamp": round(kp.get('timestamp', 0), 2),
                "yaw": round(kp.get('yaw', 0), 1),
                "x": round(pos[0], 3),
                "y": round(pos[1], 3),
                "z": round(pos[2], 3)
            })

        print(f"  [分析] 成功提取 {len(formatted_points)} 个关键点")
        return formatted_points

    except Exception as e:
        print(f"  [错误] 处理轨迹数据失败: {e}")
        return []


def main(input_path, output_path, start_idx, end_idx):
    # 1. 读取JSON数据
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_path}")
        return
    if not isinstance(data, list):
        print("错误：JSON文件根节点不是列表")
        return

    # 2. 初始化客户端
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # 录制路径
    airsim_data_path = Path("D:/AirSim_Data")
    total_cases = len(data)
    start_idx = max(0, start_idx)
    if end_idx == -1:
        end_idx = total_cases - 1
    else:
        end_idx = min(total_cases - 1, end_idx)

    print(f"开始处理，总共 {total_cases} 条，范围：{start_idx} - {end_idx}")

    # 3. 循环执行
    for i in range(start_idx, end_idx + 1):
        case = data[i]
        case_id = case.get("case_id", str(i))
        code = case.get("code", "")
        print(f"正在执行 Case {case_id} ({i}/{end_idx})...")

        try:
            # --- 准备执行环境 ---
            exec_globals = {
                "__builtins__": __builtins__,
                "airsim": airsim, "time": time, "math": math, "os": os,
                "client": client
            }

            # --- 开启录制 ---
            client.startRecording()

            # --- 执行代码 ---
            exec(code, exec_globals)

            # --- 停止录制 ---
            current_client = exec_globals.get('client', client)
            time.sleep(0.2)
            current_client.stopRecording()

            # --- 获取日志路径 ---
            time.sleep(1)  # 等待文件落盘
            log_path = get_latest_recording_path(airsim_data_path)
            case['log_path'] = log_path

            # === 新增功能：提取关键点 ===
            if log_path:
                key_points = process_trajectory_data(log_path)
                case['actual_path'] = key_points
            else:
                case['actual_path'] = []

            print(f"Case {case_id} 执行完成. 日志路径: {log_path}")

            # === 新增：打印 exp_path 和 actual_path ===
            print(f"\n--- Case {case_id} 路径数据对比 ---")

            # 打印预期路径
            exp_path = case.get('exp_path')
            print(f"[预期路径 exp_path]:")
            print(json.dumps(exp_path, indent=2, ensure_ascii=False))

            # 打印实际路径
            actual_path = case.get('actual_path')
            print(f"\n[实际路径 actual_path]:")
            print(json.dumps(actual_path, indent=2, ensure_ascii=False))
            print("-" * 50 + "\n")

        except Exception as e:
            print(f"Case {case_id} 执行出错: {e}")
            case['log_path'] = f"Execution Error: {str(e)}"
            case['actual_path'] = []
            try:
                client.stopRecording()
            except:
                pass
        finally:
            # --- 重置环境 ---
            client.reset()
            time.sleep(2)

    # 4. 保存结果
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"处理完成，结果已保存至 {output_path}")
    except Exception as e:
        print(f"保存文件出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirSim JSON 自动化执行脚本")
    parser.add_argument("--input", type=str, default="output_qwen3_32b_all_0318_agent.json", help="输入JSON文件路径")
    parser.add_argument("--output", type=str, default="output_qwen3_32b_all_0318_agent_log.json",
                        help="输出JSON文件路径")
    parser.add_argument("--start", type=int, default=0, help="起始序号 (包含)")
    parser.add_argument("--end", type=int, default=-1, help="截止序号 (包含，-1表示最后一条)")
    args = parser.parse_args()
    main(args.input, args.output, args.start, args.end)
