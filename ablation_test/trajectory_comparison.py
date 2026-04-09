#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹比对工具 (贪心匹配版)
功能：读取JSON文件，对比期望路径和实际路径，
计算偏差、完成率等指标并输出结果
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
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)


def calculate_total_length(path: List[List[float]]) -> float:
    """计算路径总长度"""
    if not path or len(path) < 2:
        return 0.0
    total_len = 0.0
    for i in range(len(path) - 1):
        total_len += calculate_distance(path[i], path[i + 1])
    return total_len


def merge_actual_path(actual_path: List[Dict[str, Any]]) -> List[List[float]]:
    """智能合并实际路径点"""
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

        # if dt <= MERGE_TIME_THRESHOLD or dist <= MERGE_DIST_THRESHOLD:
        if dist <= MERGE_DIST_THRESHOLD:
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


def match_key_points_aligned(exp_path: List[List[float]], actual_path: List[Dict[str, Any]], tolerance: float = 2.0) -> \
        List[Dict[str, Any]]:
    """
    逐个匹配期望路径点与实际路径点（基于累计误差增量匹配）。
    匹配判定依据：当前累计误差 - 上一个成功匹配点的累计误差 <= tolerance
    """
    matched_points = []
    merged_actual = merge_actual_path(actual_path)
    actual_start_idx = 0  # 未匹配的实际点起始索引

    # 新增：记录上一个成功匹配的关键点的"累计误差"（即该点到其期望点的欧氏距离）
    last_matched_cum_dist = 0.0

    # 第一遍：贪心匹配
    match_results = []
    for exp_idx, exp_point in enumerate(exp_path):
        found = False
        for act_idx in range(actual_start_idx, len(merged_actual)):
            # 1. 计算当前实际点到当前期望点的绝对欧氏距离（作为当前的"累计误差"）
            curr_euclidean_dist = calculate_distance(exp_point, merged_actual[act_idx])

            # 2. 核心修改：计算相对增量 dist
            step_dist = curr_euclidean_dist - last_matched_cum_dist

            # 3. 使用增量进行判定
            # 注意：如果无人机走了捷径，step_dist 可能为负数，负数 <= tolerance 恒成立（视为匹配成功）
            if step_dist <= tolerance:
                match_results.append((exp_idx, act_idx, curr_euclidean_dist))
                actual_start_idx = act_idx + 1  # 推进实际指针
                last_matched_cum_dist = curr_euclidean_dist  # 更新基准累计误差
                found = True
                break

        if not found:
            match_results.append((exp_idx, None, float('inf')))
            # 注意：匹配失败时，last_matched_cum_dist 不变，下一个期望点依然以这个基准计算增量

    # 第二遍：计算单步误差并构建结果 (此部分无需修改，因为算出来的 step_error 和匹配时的 step_dist 逻辑一致)
    for i, (exp_idx, act_idx, dist) in enumerate(match_results):
        exp_point = exp_path[exp_idx]
        if act_idx is not None:
            actual_point = merged_actual[act_idx]
            curr_cum_dist = dist
            if i == 0:
                step_error = curr_cum_dist
            else:
                prev_dist = match_results[i - 1][2]
                step_error = curr_cum_dist - prev_dist
            matched_points.append({
                'exp_point_idx': exp_idx,
                'exp_point': exp_point,
                'matched_actual_point': actual_point,
                'cumulative_dist': curr_cum_dist,
                'step_error': step_error
            })
        else:
            matched_points.append({
                'exp_point_idx': exp_idx,
                'exp_point': exp_point,
                'matched_actual_point': None,
                'cumulative_dist': float('inf'),
                'step_error': float('inf')
            })

    return matched_points


def calculate_completion_rate(matched_points: List[Dict[str, Any]], tolerance: float = 2.0) -> float:
    """计算完成率：依据是否成功匹配到实际点"""
    if not matched_points:
        return 0.0
    reached_count = sum(1 for point in matched_points if point['matched_actual_point'] is not None)
    return reached_count / len(matched_points)


def is_endpoint_reached_cumulative(matched_points: List[Dict[str, Any]], total_length: float,
                                   threshold_ratio: float = 0.1) -> bool:
    """判断是否到达终点：依据终点累计误差是否小于总路程 * 阈值比例"""
    if not matched_points:
        return False
    last_point = matched_points[-1]

    # 如果终点没有匹配到实际点，直接返回False
    if last_point['matched_actual_point'] is None:
        return False

    max_allowed_error = total_length * threshold_ratio
    if total_length == 0:
        return bool(abs(last_point['cumulative_dist']) <= 0.1)
    return bool(abs(last_point['cumulative_dist']) <= max_allowed_error)


def calculate_path_deviation(matched_points: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    计算累计偏差统计信息。
    修改：如果当前实际点为None，使用上一个有数值的实际点计算偏差。
    """
    if not matched_points:
        return {'avg_deviation': 0.0, 'max_deviation': 0.0, 'min_deviation': 0.0, 'rmse': 0.0}

    distances = []
    last_valid_actual = None  # 记录上一个有效的实际点坐标

    for point in matched_points:
        exp_point = point['exp_point']
        act_point = point['matched_actual_point']

        if act_point is not None:
            # 如果当前点有效，更新最近有效点，并计算偏差
            last_valid_actual = act_point
            distances.append(calculate_distance(exp_point, act_point))
        else:
            # 如果当前点无效，且存在历史有效点，使用历史有效点计算偏差
            if last_valid_actual is not None:
                distances.append(calculate_distance(exp_point, last_valid_actual))
            # else: 如果从未有过有效点，暂时无法计算偏差，跳过

    if not distances:
        return {'avg_deviation': 0.0, 'max_deviation': 0.0, 'min_deviation': 0.0, 'rmse': 0.0}

    rmse = np.sqrt(np.mean(np.square(distances)))
    return {
        'avg_deviation': float(np.mean(distances)),
        'max_deviation': float(np.max(distances)),
        'min_deviation': float(np.min(distances)),
        'rmse': float(rmse)
    }


def calculate_shape_similarity(matched_points: List[Dict[str, Any]]) -> float:
    """
    计算形状相似度。
    修改：如果当前实际点为None，使用上一个有数值的实际点构建向量（即认为在该点悬停）。
    """
    if len(matched_points) < 2:
        return 1.0

    # 1. 构建补齐后的实际路径点序列
    resolved_actual_points = []
    last_valid_actual = None

    for point in matched_points:
        act_p = point['matched_actual_point']
        if act_p is not None:
            last_valid_actual = act_p
            resolved_actual_points.append(act_p)
        else:
            # 如果当前点为None，使用上一个有效点代替
            if last_valid_actual is not None:
                resolved_actual_points.append(last_valid_actual)
            else:
                # 如果一直没有有效点，无法计算，标记为None
                resolved_actual_points.append(None)

    similarities = []
    for i in range(len(matched_points) - 1):
        p_curr_exp = matched_points[i]['exp_point']
        p_next_exp = matched_points[i + 1]['exp_point']

        # 获取补齐后的实际点
        p_curr_act = resolved_actual_points[i]
        p_next_act = resolved_actual_points[i + 1]

        # 如果补齐后仍为None（说明开头一直没有匹配点），跳过
        if p_curr_act is None or p_next_act is None:
            continue

        vec_exp = np.array(p_next_exp) - np.array(p_curr_exp)
        vec_act = np.array(p_next_act) - np.array(p_curr_act)
        norm_exp = np.linalg.norm(vec_exp)
        norm_act = np.linalg.norm(vec_act)

        if norm_exp == 0 and norm_act == 0:
            similarities.append(1.0)
            continue
        if norm_exp == 0 or norm_act == 0:
            # 如果期望移动但实际没动（norm_act=0），相似度为0
            similarities.append(0.0)
            continue

        cos_theta = np.dot(vec_exp, vec_act) / (norm_exp * norm_act)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        similarities.append(cos_theta)

    if not similarities:
        return 0.0
    return float(np.mean(similarities))


def evaluate_success(matched_points: List[Dict[str, Any]], total_length: float, completion_threshold: float = 0.8,
                     endpoint_threshold_ratio: float = 0.1) -> bool:
    """综合判定是否成功"""
    endpoint_reached = is_endpoint_reached_cumulative(matched_points, total_length, endpoint_threshold_ratio)
    completion_rate = calculate_completion_rate(matched_points)
    return bool(endpoint_reached and completion_rate >= completion_threshold)


def process_trajectory(data: Dict[str, Any], tolerance: float = 2.0, completion_threshold: float = 0.8,
                       endpoint_threshold: float = 0.1) -> Dict[str, Any]:
    exp_path = data.get('exp_path', [])
    actual_path_raw = data.get('actual_path', [])
    total_length = calculate_total_length(exp_path)

    # 传入 tolerance 进行贪心匹配
    matched_points = match_key_points_aligned(exp_path, actual_path_raw, tolerance)
    endpoint_reached = is_endpoint_reached_cumulative(matched_points, total_length, endpoint_threshold)
    completion_rate = calculate_completion_rate(matched_points, tolerance)

    # 修改：判断 actual_path 是否为空
    has_valid_actual_path = actual_path_raw is not None and len(actual_path_raw) > 0

    # 只有当 actual_path 不为空时才计算偏差和相似度
    if has_valid_actual_path:
        deviation_stats = calculate_path_deviation(matched_points)
        shape_similarity = calculate_shape_similarity(matched_points)
    else:
        # actual_path 为空时，设置为默认值
        deviation_stats = {'avg_deviation': 0.0, 'max_deviation': 0.0, 'min_deviation': 0.0, 'rmse': 0.0}
        shape_similarity = 0.0

    is_success = evaluate_success(matched_points, total_length, completion_threshold, endpoint_threshold)

    # --- 打印详细信息 ---
    print(f"\n========== Case ID: {data.get('case_id', 'unknown')} ==========")
    print(
        f"【路径参数】总路程: {total_length:.2f}m, 终点允许误差: {total_length * endpoint_threshold:.2f}m ({endpoint_threshold * 100:.0f}%)")
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

        # 状态判断依据改为是否有匹配点
        status = "✓" if act_p is not None else "✗"
        if act_p is not None:
            reached_count += 1

        # 格式化偏差数值，防止inf打印溢出
        cum_str = f"{cum_dist:.4f}m" if np.isfinite(cum_dist) else "N/A"
        step_str = f"{step_err:.4f}m" if np.isfinite(step_err) else "N/A"

        print(f"{point['exp_point_idx']:<6}{exp_str:<24}{act_str:<24}{cum_str:<12}{step_str:<12}{status:<6}")

    total_count = len(matched_points)
    print("-" * 86)
    print(f"【统计结果】")
    print(f" 完成点数: {reached_count} / {total_count}")
    print(f" 完成率: {completion_rate:.2f}%")

    # 根据是否有有效实际路径，显示不同的信息
    if has_valid_actual_path:
        print(f" 平均累计偏差: {deviation_stats['avg_deviation']:.4f}m")
        print(f" 轨迹偏差(RMSE): {deviation_stats['rmse']:.4f}m")
        print(f" 形状相似度: {shape_similarity:.4f}")
    else:
        print(f" 平均累计偏差: N/A (actual_path为空)")
        print(f" 轨迹偏差(RMSE): N/A (actual_path为空)")
        print(f" 形状相似度: N/A (actual_path为空)")

    print(f" 终点累计误差达标: {'是' if endpoint_reached else '否'}")

    success_str = "成功" if is_success else "失败"
    print(f"【最终判定】: {success_str}")
    print("=" * 42 + "\n")

    result = data.copy()
    result['trajectory_analysis'] = {
        'matched_points': matched_points,
        'endpoint_reached': endpoint_reached,
        'completion_rate': round(completion_rate * 100, 2),
        'deviation': deviation_stats,
        'shape_similarity': round(shape_similarity, 4),
        'success': is_success,
        'tolerance': tolerance,
        'completion_threshold': completion_threshold,
        'endpoint_threshold': endpoint_threshold,
        'total_path_length': round(total_length, 4),
        'has_valid_actual_path': has_valid_actual_path  # 新增标记
    }
    return result


def calculate_overall_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    total_count = len(results)
    success_count = sum(1 for r in results if r.get('trajectory_analysis', {}).get('success', False))
    endpoint_reached_count = sum(1 for r in results if r.get('trajectory_analysis', {}).get('endpoint_reached', False))

    completion_rates = [r.get('trajectory_analysis', {}).get('completion_rate', 0) for r in results]

    # 修改：只统计 has_valid_actual_path 为 True 的结果
    valid_results = [
        r for r in results
        if r.get('trajectory_analysis', {}).get('has_valid_actual_path', False)
    ]
    avg_deviations = [r.get('trajectory_analysis', {}).get('deviation', {}).get('avg_deviation', 0) for r in
                      valid_results]
    avg_rmse = [r.get('trajectory_analysis', {}).get('deviation', {}).get('rmse', 0) for r in valid_results]
    avg_similarities = [r.get('trajectory_analysis', {}).get('shape_similarity', 0) for r in valid_results]
    valid_count = len(valid_results)

    return {
        'total_count': total_count,
        'valid_count': valid_count,
        'success_count': success_count,
        'success_rate': round(success_count / total_count * 100, 2) if total_count > 0 else 0,
        'endpoint_reached_count': endpoint_reached_count,
        'endpoint_reached_rate': round(endpoint_reached_count / total_count * 100, 2) if total_count > 0 else 0,
        'avg_completion_rate': round(np.mean(completion_rates), 2) if completion_rates else 0,
        'avg_deviation': round(np.mean(avg_deviations), 4) if valid_count > 0 else 0,
        'avg_rmse': round(np.mean(avg_rmse), 4) if valid_count > 0 else 0,
        'avg_shape_similarity': round(np.mean(avg_similarities), 4) if valid_count > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description='轨迹比对工具')
    parser.add_argument('--input', '-i', default="output_qwen3_32b_action_log.json", help='输入JSON文件路径')
    parser.add_argument('--output', '-o', default="output_qwen3_32b_action_result.json", help='输出JSON文件路径')
    parser.add_argument('--start', '-s', type=int, default=0, help='开始序号（默认为0）')
    parser.add_argument('--end', '-e', type=int, default=-1, help='结束序号（-1表示到最后，默认为-1）')
    parser.add_argument('--tolerance', '-t', type=float, default=1.0, help='单步误差容差阈值（米，默认1.0）')
    parser.add_argument('--completion', '-c', type=float, default=0.8, help='完成率阈值（默认0.8）')
    parser.add_argument('--endpoint_threshold', '-et', type=float, default=0.1,
                        help='终点累计误差相对于总路程的阈值比例（默认0.1）')
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
        result = process_trajectory(item, args.tolerance, args.completion, args.endpoint_threshold)
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
        print(f" - 总轨迹数: {overall_stats.get('total_count', 0)}")
        print(f" - 有效轨迹数: {overall_stats.get('valid_count', 0)} (用于计算偏差和相似度)")
        print(f" - 成功数: {overall_stats.get('success_count', 0)}")
        print(f" - 成功率: {overall_stats.get('success_rate', 0)}%")
        print(f" - 到达终点数: {overall_stats.get('endpoint_reached_count', 0)}")
        print(f" - 到达终点率: {overall_stats.get('endpoint_reached_rate', 0)}%")
        print(f" - 平均完成率: {overall_stats.get('avg_completion_rate', 0)}%")
        print(f" - 平均偏差: {overall_stats.get('avg_deviation', 0)}米")
        print(f" - 平均RMSE: {overall_stats.get('avg_rmse', 0)}米")
        print(f" - 平均形状相似度: {overall_stats.get('avg_shape_similarity', 0)}")
        print("=" * 50)
    except Exception as e:
        print(f"错误：无法写入输出文件 {args.output}: {e}")


if __name__ == '__main__':
    main()
