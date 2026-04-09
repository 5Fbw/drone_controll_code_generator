import json
import os
import time
import argparse
import airsim
import math
import threading  # [修改1] 引入threading模块
from pathlib import Path

# 引入 path_eval 中的相关类
from path_eval import Path_eval_utils, TrajectoryAnalyzer


def get_latest_recording_path(base_path):
    """ 获取AirSim录制目录下最新生成的文件夹路径 """
    if not os.path.exists(base_path):
        return ""
    try:
        subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if
                   os.path.isdir(os.path.join(base_path, d))]
    except Exception:
        return ""
    if not subdirs:
        return ""
    # 按修改时间排序，获取最新的一个
    latest_dir = max(subdirs, key=os.path.getmtime)
    return latest_dir


def process_trajectory_data(log_dir):
    """ 读取指定目录下的日志文件，提取关键点信息 """
    rec_file_path = os.path.join(log_dir, "airsim_rec.txt")
    if not os.path.exists(rec_file_path):
        print(f" [警告] 未找到录制文件: {rec_file_path}")
        return []
    try:
        # 1. 解析轨迹数据
        utils = Path_eval_utils()
        path_array = utils.to_path_ponit_array(rec_file_path)
        if not path_array:
            print(" [警告] 轨迹数据为空")
            return []
        # 2. 分析提取关键点
        analyzer = TrajectoryAnalyzer()
        key_points = analyzer.extract_key_points(path_array)
        # 3. 格式化数据
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
        print(f" [分析] 成功提取 {len(formatted_points)} 个关键点")
        return formatted_points
    except Exception as e:
        print(f" [错误] 处理轨迹数据失败: {e}")
        return []


# [修改2] 新增超时执行函数
def exec_with_timeout(code, exec_globals, timeout_seconds):
    """
    在独立线程中执行代码，并限制执行时间
    返回: (success: bool, error_msg: str or None)
    """
    result = {'success': False, 'error': None}

    def run_code():
        try:
            exec(code, exec_globals)
            result['success'] = True
        except Exception as e:
            result['error'] = str(e)

    thread = threading.Thread(target=run_code)
    thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # 线程仍在运行，说明超时了
        return False, f"执行超时（超过 {timeout_seconds} 秒）"

    return result['success'], result['error']


def main(input_path, output_path, start_idx, end_idx, timeout):
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
    airsim_data_path = Path("D:/AirSim_Data")

    total_cases = len(data)
    start_idx = max(0, start_idx)
    if end_idx == -1:
        end_idx = total_cases - 1
    else:
        end_idx = min(total_cases - 1, end_idx)

    print(f"开始处理，总共 {total_cases} 条，范围：{start_idx} - {end_idx}，超时设置: {timeout}秒")

    # 3. 循环执行
    for i in range(start_idx, end_idx + 1):
        case = data[i]
        case_id = case.get("case_id", str(i))
        code = case.get("code", "")
        # ✅ 新增：修复转义换行符
        code = code.replace('\\n', '\n')
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

            # [修改3] 使用带超时的执行函数替换原 exec
            success, error_msg = exec_with_timeout(code, exec_globals, timeout)

            if not success:
                # 如果执行失败或超时，抛出异常进入错误处理流程
                raise Exception(error_msg or "执行失败")

            # --- 停止录制 ---
            current_client = exec_globals.get('client', client)
            time.sleep(0.2)
            current_client.stopRecording()

            # --- 获取日志路径 ---
            time.sleep(1)  # 等待文件落盘
            log_path = get_latest_recording_path(airsim_data_path)
            case['log_path'] = log_path

            # 提取关键点
            if log_path:
                key_points = process_trajectory_data(log_path)
                case['actual_path'] = key_points
            else:
                case['actual_path'] = []
            print(f"Case {case_id} 执行完成. 日志路径: {log_path}")

            # 打印对比信息
            print(f"\n--- Case {case_id} 路径数据对比 ---")
            print(f"[预期路径 exp_path]:")
            print(json.dumps(case.get('exp_path'), indent=2, ensure_ascii=False))
            print(f"\n[实际路径 actual_path]:")
            print(json.dumps(case.get('actual_path'), indent=2, ensure_ascii=False))
            print("-" * 50 + "\n")

        except Exception as e:
            print(f"Case {case_id} 执行出错: {e}")
            case['log_path'] = f"Execution Error: {str(e)}"
            case['actual_path'] = []

            # 异常处理中尝试停止录制并保存现有数据
            try:
                client.stopRecording()
                time.sleep(1)
                log_path = get_latest_recording_path(airsim_data_path)
                case['log_path'] = log_path  # 覆盖错误信息为路径或保留错误信息？
                # 通常保留错误信息更有价值，这里逻辑保持原样稍作调整
                if log_path:
                    case['log_path'] = log_path

                if log_path:
                    key_points = process_trajectory_data(log_path)
                    case['actual_path'] = key_points
                print(f"\n--- 日志路径 ---")
                print(f"{log_path}")
                print(f"\n--- Case {case_id} 路径数据对比 (异常后) ---")
                print(f"[预期路径 exp_path]:")
                print(json.dumps(case.get('exp_path'), indent=2, ensure_ascii=False))
                print(f"\n[实际路径 actual_path]:")
                print(json.dumps(case.get('actual_path'), indent=2, ensure_ascii=False))
                print("-" * 50 + "\n")
            except:
                pass
            # 注意：原有的外层 try-except 结构中嵌套了finally，下面的finally会处理reset
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
    parser.add_argument("--input", type=str, default="output_qwen3_32b_action.json", help="输入JSON文件路径")
    parser.add_argument("--output", type=str, default="output_qwen3_32b_action_log.json", help="输出JSON文件路径")
    parser.add_argument("--start", type=int, default=0, help="起始序号 (包含)")
    parser.add_argument("--end", type=int, default=-1, help="截止序号 (包含，-1表示最后一条)")
    # [修改4] 新增命令行参数
    parser.add_argument("--timeout", type=int, default=120, help="单条用例执行超时时间(秒)")

    args = parser.parse_args()
    main(args.input, args.output, args.start, args.end, args.timeout)
