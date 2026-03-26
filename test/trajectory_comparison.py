#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹比对工具 (增强指标版)
功能：读取JSON文件，对比期望路径和实际路径，
     计算偏差、完成率等指标并输出结果
修正点：
1. 单步误差 = 当前累计误差 - 上一步累计误差
2. 点位完成判定依据改为：abs(单步误差) <= tolerance
新增功能：
1. 轨迹偏差指标
2. 形状相似度指标
"""

import json
import argparse
import numpy as np
from typing import List, Dict, Any

# --- 配置参数 ---
MERGE_TIME_THRESHOLD = 1.0  # 时间聚类阈值（秒）
MERGE_DIST_THRESHOLD = 0.5  # 空间聚类阈值（米）


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """计算两个3D点之间的欧氏距离"""
    if not point1 or not point2:
        return float('inf')
    return np.sqrt((point1[0] - point2[0]) ** 2 +
                   (point1[1] - point2[1]) ** 2 +
                   (point1[2] - point2[2]) ** 2)


def merge_actual_path(actual_path: List[Dict[str, Any]]) -> List[List[float]]:
    """
    智能合并实际路径点：
    - 如果连续两个点的时间差和距离差都小于阈值，则合并为一个点。
    """
    if not actual_path:
        return []

    merged_path = []
    current_cluster = [actual_path[0]]

    for i in range(1, len(actual_path)):
        prev_point = current_cluster[-1]
        curr_point = actual_path[i]

        p1_coords = [prev_point.get('x', 0), prev_point.get('y', 0), prev_point.get('z', 0)]
        p2_coords = [curr_point.get('x', 0), curr_point.get('y', 0), curr_point.get('z', 0)]

        dt = abs(curr_point.get('timestamp', 0) - prev_point.get('timestamp', 0))
        dist = calculate_distance(p1_coords, p2_coords)

        # 注意：这里沿用了你提供的代码逻辑，使用了 'or'。
        # 如果发现合并过于激进（把快速移动的点合并了），建议改为 'and'
        if dt <= MERGE_TIME_THRESHOLD or dist <= MERGE_DIST_THRESHOLD:
            current_cluster.append(curr_point)
        else:
            avg_x = np.mean([p.get('x', 0) for p in current_cluster])
            avg_y = np.mean([p.get('y', 0) for p in current_cluster])
            avg_z = np.mean([p.get('z', 0) for p in current_cluster])
            merged_path.append([avg_x, avg_y, avg_z])
            current_cluster = [curr_point]

    if current_cluster:
        avg_x = np.mean([p.get('x', 0) for p in current_cluster])
        avg_y = np.mean([p.get('y', 0) for p in current_cluster])
        avg_z = np.mean([p.get('z', 0) for p in current_cluster])
        merged_path.append([avg_x, avg_y, avg_z])

    return merged_path


def resample_path(path: List[List[float]], target_count: int) -> List[List[float]]:
    """将路径重采样为固定的目标点数"""
    if not path:
        return []

    current_count = len(path)

    if current_count >= target_count:
        indices = np.linspace(0, current_count - 1, target_count, dtype=int)
        return [path[i] for i in indices]
    else:
        resampled = path.copy()
        while len(resampled) < target_count:
            resampled.append(path[-1])
        return resampled


def match_key_points_aligned(exp_path: List[List[float]], actual_path: List[List[float]]) -> List[Dict[str, Any]]:
    """
    匹配期望路径与实际路径
    """
    matched_points = []

    merged_actual = merge_actual_path(actual_path)
    resampled_actual = resample_path(merged_actual, len(exp_path))

    # 第一遍循环：计算所有点的累计误差
    cumulative_distances = []
    for idx, exp_point in enumerate(exp_path):
        if not resampled_actual:
            cumulative_distances.append(float('inf'))
            continue

        actual_point = resampled_actual[idx]
        dist = calculate_distance(exp_point, actual_point)
        cumulative_distances.append(dist)

    # 第二遍循环：计算单步误差并构建结果
    for idx, exp_point in enumerate(exp_path):
        if not resampled_actual:
            matched_points.append({
                'exp_point_idx': idx,
                'exp_point': exp_point,
                'matched_actual_point': None,
                'cumulative_dist': float('inf'),
                'step_error': float('inf')
            })
            continue

        actual_point = resampled_actual[idx]
        curr_cum_dist = cumulative_distances[idx]

        # 计算单步误差
        if idx == 0:
            step_error = curr_cum_dist
        else:
            prev_cum_dist = cumulative_distances[idx - 1]
            step_error = curr_cum_dist - prev_cum_dist

        matched_points.append({
            'exp_point_idx': idx,
            'exp_point': exp_point,
            'matched_actual_point': actual_point,
            'cumulative_dist': curr_cum_dist,
            'step_error': step_error
        })

    return matched_points


def calculate_completion_rate(matched_points: List[Dict[str, Any]],
                              tolerance: float = 2.0) -> float:
    """计算完成率：依据 abs(单步误差) <= tolerance"""
    if not matched_points:
        return 0.0
    reached_count = sum(1 for point in matched_points if abs(point['step_error']) <= tolerance)
    return reached_count / len(matched_points)


def is_endpoint_reached_cumulative(matched_points: List[Dict[str, Any]],
                        tolerance: float = 2.0) -> bool:
    """判断是否到达终点：依据终点的单步误差"""
    if not matched_points:
        return False
    last_point = matched_points[-1]
    return bool(abs(last_point['cumulative_dist']) <= tolerance)
def is_endpoint_reached_step(matched_points: List[Dict[str, Any]],
                        tolerance: float = 2.0) -> bool:
    """判断是否到达终点：依据终点的单步误差"""
    if not matched_points:
        return False
    last_point = matched_points[-1]
    return bool(abs(last_point['step_error']) <= tolerance)

def calculate_path_deviation(matched_points: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算累计偏差统计信息"""
    if not matched_points:
        return {'avg_deviation': 0.0, 'max_deviation': 0.0, 'min_deviation': 0.0, 'rmse': 0.0}

    distances = [point['cumulative_dist'] for point in matched_points]

    # 新增 RMSE 计算
    rmse = np.sqrt(np.mean(np.square(distances)))

    return {
        'avg_deviation': float(np.mean(distances)),
        'max_deviation': float(np.max(distances)),
        'min_deviation': float(np.min(distances)),
        'rmse': float(rmse)
    }


def calculate_shape_similarity(matched_points: List[Dict[str, Any]]) -> float:
    """
    计算形状相似度：
    基于方向向量的余弦相似度。
    比较每一段路径的移动方向是否一致。
    返回值范围 [-1, 1], 1 表示完全一致。
    """
    if len(matched_points) < 2:
        return 1.0  # 只有一个点，视为形状一致

    similarities = []
    for i in range(len(matched_points) - 1):
        # 获取当前点和下一点
        p_curr_exp = matched_points[i]['exp_point']
        p_next_exp = matched_points[i + 1]['exp_point']

        p_curr_act = matched_points[i]['matched_actual_point']
        p_next_act = matched_points[i + 1]['matched_actual_point']

        if not p_curr_act or not p_next_act:
            continue

        # 计算向量
        vec_exp = np.array(p_next_exp) - np.array(p_curr_exp)
        vec_act = np.array(p_next_act) - np.array(p_curr_act)

        norm_exp = np.linalg.norm(vec_exp)
        norm_act = np.linalg.norm(vec_act)

        # 如果两段都是静止的，认为相似度为1
        if norm_exp == 0 and norm_act == 0:
            similarities.append(1.0)
            continue
        # 如果一段静止一段运动，相似度为0
        if norm_exp == 0 or norm_act == 0:
            similarities.append(0.0)
            continue

        # 计算余弦相似度
        cos_theta = np.dot(vec_exp, vec_act) / (norm_exp * norm_act)

        # 截断在 [-1, 1] 之间，防止浮点误差
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        similarities.append(cos_theta)

    if not similarities:
        return 0.0

    return float(np.mean(similarities))


def evaluate_success(matched_points: List[Dict[str, Any]],
                     completion_threshold: float = 0.8,
                     tolerance: float = 2.0) -> bool:
    endpoint_reached = is_endpoint_reached_step(matched_points, tolerance)
    completion_rate = calculate_completion_rate(matched_points, tolerance)
    return bool(endpoint_reached and completion_rate >= completion_threshold)


def process_trajectory(data: Dict[str, Any],
                       tolerance: float = 2.0,
                       completion_threshold: float = 0.8) -> Dict[str, Any]:
    exp_path = data.get('exp_path', [])
    actual_path_raw = data.get('actual_path', [])

    matched_points = match_key_points_aligned(exp_path, actual_path_raw)

    endpoint_reached = is_endpoint_reached_cumulative(matched_points, tolerance)
    completion_rate = calculate_completion_rate(matched_points, tolerance)
    deviation_stats = calculate_path_deviation(matched_points)
    shape_similarity = calculate_shape_similarity(matched_points)  # 新增计算
    is_success = evaluate_success(matched_points, completion_threshold, tolerance)

    # --- 打印详细信息 ---
    print(f"\n========== Case ID: {data.get('case_id', 'unknown')} ==========")
    print("【点位匹配详情】")
    header = f"{'序号':<6}{'期望点':<24}{'实际点':<24}{'累计偏差':<12}{'单步误差':<12}{'状态':<6}"
    print(header)
    print("-" * 86)

    reached_count = 0
    for point in matched_points:
        exp_p = point['exp_point']
        act_p = point['matched_actual_point']
        cum_dist = point['cumulative_dist']
        step_err = point['step_error']

        exp_str = f"[{exp_p[0]:.2f}, {exp_p[1]:.2f}, {exp_p[2]:.2f}]"
        act_str = f"[{act_p[0]:.2f}, {act_p[1]:.2f}, {act_p[2]:.2f}]" if act_p else "None"

        status = "✓" if abs(step_err) <= tolerance else "✗"
        if abs(step_err) <= tolerance:
            reached_count += 1

        print(f"{point['exp_point_idx']:<6}{exp_str:<24}{act_str:<24}"
              f"{cum_dist:.4f}m     {step_err:.4f}m     {status:<6}")

    total_count = len(matched_points)
    print("-" * 86)
    print(f"【统计结果】")
    print(f"  完成点数: {reached_count} / {total_count}")
    print(f"  完成率: {completion_rate:.2f}%")
    print(f"  平均累计偏差: {deviation_stats['avg_deviation']:.4f}m")
    print(f"  轨迹偏差(RMSE): {deviation_stats['rmse']:.4f}m")  # 新增打印
    print(f"  形状相似度: {shape_similarity:.4f}")  # 新增打印
    print(f"  终点单步误差达标: {'是' if endpoint_reached else '否'}")

    success_str = "成功" if is_success else "失败"
    print(f"【最终判定】: {success_str}")
    print("=" * 42 + "\n")

    result = data.copy()
    result['trajectory_analysis'] = {
        'matched_points': matched_points,
        'endpoint_reached': endpoint_reached,
        'completion_rate': round(completion_rate * 100, 2),
        'deviation': deviation_stats,
        'shape_similarity': round(shape_similarity, 4),  # 新增保存
        'success': is_success,
        'tolerance': tolerance,
        'completion_threshold': completion_threshold
    }

    return result


def calculate_overall_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    total_count = len(results)
    success_count = sum(1 for r in results if r.get('trajectory_analysis', {}).get('success', False))
    endpoint_reached_count = sum(1 for r in results if r.get('trajectory_analysis', {}).get('endpoint_reached', False))

    completion_rates = [r.get('trajectory_analysis', {}).get('completion_rate', 0) for r in results]
    avg_deviations = [r.get('trajectory_analysis', {}).get('deviation', {}).get('avg_deviation', 0) for r in results]

    # 新增总体统计
    avg_rmse = [r.get('trajectory_analysis', {}).get('deviation', {}).get('rmse', 0) for r in results]
    avg_similarities = [r.get('trajectory_analysis', {}).get('shape_similarity', 0) for r in results]

    return {
        'total_count': total_count,
        'success_count': success_count,
        'success_rate': round(success_count / total_count * 100, 2) if total_count > 0 else 0,
        'endpoint_reached_count': endpoint_reached_count,
        'endpoint_reached_rate': round(endpoint_reached_count / total_count * 100, 2) if total_count > 0 else 0,
        'avg_completion_rate': round(np.mean(completion_rates), 2) if completion_rates else 0,
        'avg_deviation': round(np.mean(avg_deviations), 4) if avg_deviations else 0,
        'avg_rmse': round(np.mean(avg_rmse), 4) if avg_rmse else 0,
        'avg_shape_similarity': round(np.mean(avg_similarities), 4) if avg_similarities else 0
    }


def main():
    parser = argparse.ArgumentParser(description='轨迹比对工具')
    parser.add_argument('--input', '-i', default="output_qwen3_32b_all_0318_agent_log.json", help='输入JSON文件路径')
    parser.add_argument('--output', '-o', default="output_qwen3_32b_all_0318_agent_result.json",
                        help='输出JSON文件路径')
    # parser.add_argument('--input', '-i', default="output_qwen3_32b_all_0_log.json", help='输入JSON文件路径')
    # parser.add_argument('--output', '-o', default="output_qwen3_32b_all_0_result.json",
    #                     help='输出JSON文件路径')
    parser.add_argument('--start', '-s', type=int, default=0, help='开始序号（默认为0）')
    parser.add_argument('--end', '-e', type=int, default=-1, help='结束序号（-1表示到最后，默认为-1）')
    parser.add_argument('--tolerance', '-t', type=float, default=1.0, help='单步误差容差阈值（米，默认1.0）')
    parser.add_argument('--completion', '-c', type=float, default=0.8, help='完成率阈值（默认0.8）')

    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {args.input}")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件 {args.input} 不是有效的JSON格式")
        return

    if isinstance(data, dict):
        data = [data]

    if args.end == -1:
        data = data[args.start:]
    else:
        data = data[args.start:args.end + 1]

    print(f"开始处理 {len(data)} 条轨迹数据...\n")

    results = []
    for idx, item in enumerate(data, start=args.start):
        result = process_trajectory(item, args.tolerance, args.completion)
        results.append(result)

    overall_stats = calculate_overall_statistics(results)

    output_data = {
        'results': results,
        'overall_statistics': overall_stats
    }

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！结果已保存到 {args.output}")
        print(f"\n{'=' * 20} 总体统计 {'=' * 20}")
        print(f"  - 总轨迹数: {overall_stats.get('total_count', 0)}")
        print(f"  - 成功数: {overall_stats.get('success_count', 0)}")
        print(f"  - 成功率: {overall_stats.get('success_rate', 0)}%")
        print(f"  - 到达终点数: {overall_stats.get('endpoint_reached_count', 0)}")
        print(f"  - 到达终点率: {overall_stats.get('endpoint_reached_rate', 0)}%")
        print(f"  - 平均完成率: {overall_stats.get('avg_completion_rate', 0)}%")
        print(f"  - 平均偏差: {overall_stats.get('avg_deviation', 0)}米")
        print(f"  - 平均RMSE: {overall_stats.get('avg_rmse', 0)}米")
        print(f"  - 平均形状相似度: {overall_stats.get('avg_shape_similarity', 0)}")
        print("=" * 50)
    except Exception as e:
        print(f"错误：无法写入输出文件 {args.output}: {e}")


if __name__ == '__main__':
    main()
