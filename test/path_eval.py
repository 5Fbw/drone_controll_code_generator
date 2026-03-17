from pathlib import Path
import json
import numpy as np
class Path_eval_utils:
    def to_path_ponit_array(self, path_file: str):
        """
            已知轨迹文件名称，读取文件内容并解析为轨迹点数组。
            时间戳会被转换为相对于起点的秒数。

            Args:
                path_file (str): 轨迹文件的路径（相对路径或绝对路径）

            Returns:
                list: 包含轨迹点字典的列表，每个字典包含时间戳、位置和姿态信息。
                      格式: [{'timestamp': float, 'position': (x,y,z), 'quaternion': (w,x,y,z)}, ...]
                      如果文件读取失败，返回空列表。
        """
        file_path = Path(path_file)

        # 1. 检查文件是否存在
        if not file_path.exists():
            print(f"错误：文件 {path_file} 不存在")
            return []

        trajectory_data = []

        try:
            # 2. 打开并读取文件
            lines = file_path.read_text(encoding='utf-8').splitlines()

            # 3. 遍历每一行
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                try:
                    # 提取数据并转换类型
                    timestamp = int(parts[1])
                    pos_x = float(parts[2])
                    pos_y = float(parts[3])
                    pos_z = float(parts[4])
                    q_w = float(parts[5])
                    q_x = float(parts[6])
                    q_y = float(parts[7])
                    q_z = float(parts[8])

                    point = {
                        'timestamp': timestamp,  # 暂时保存原始时间戳
                        'position': (pos_x, pos_y, pos_z),
                        'quaternion': (q_w, q_x, q_y, q_z)
                    }

                    trajectory_data.append(point)

                except (ValueError, IndexError) as e:
                    print(f"警告：跳过格式错误的行 -> {line} (错误: {e})")
                    continue

            # --- 新增逻辑：时间戳归一化处理 ---
            if trajectory_data:
                # 获取第一个点的时间戳作为基准时间
                start_timestamp = trajectory_data[0]['timestamp']

                # 遍历列表，将所有时间戳减去基准值并转换为秒
                for point in trajectory_data:
                    original_ts = point['timestamp']
                    # 计算差值并转换为秒 (除以1000)
                    point['timestamp'] = (original_ts - start_timestamp) / 1000.0
            # ---------------------------------

        except Exception as e:
            print(f"读取文件发生未知错误: {e}")
            return []

        return trajectory_data

utils = Path_eval_utils()
path_array = utils.to_path_ponit_array("2026-01-13-17-10-21/airsim_rec.txt")
print(path_array[:5])
import numpy as np
from typing import List, Dict, Tuple
import math

import numpy as np
from typing import List, Dict, Tuple
import math


class TrajectoryAnalyzer:
    def __init__(self):
        # 运动状态阈值
        self.velocity_threshold = 0.1  # m/s，用于判断静止状态
        self.direction_threshold = 30  # 度，用于判断方向变化
        self.height_threshold = 0.2  # m，用于判断高度变化

    def calculate_velocity(self, points: List[Dict]) -> List[Dict]:
        """计算每个点的瞬时速度"""
        if len(points) < 2:
            return []

        velocities = []
        for i in range(len(points)):
            if i == 0:
                # 第一个点速度设为零
                velocities.append({
                    'point': points[i],
                    'velocity': (0, 0, 0),
                    'speed': 0,
                    'direction': None,
                    'vertical_speed': 0,
                    'horizontal_speed': 0,
                    'horizontal_direction': None
                })
            else:
                # 计算时间差（毫秒转秒）
                dt = (points[i]['timestamp'] - points[i - 1]['timestamp']) / 1000.0

                if dt <= 0:
                    dt = 0.001  # 避免除零

                # 计算位置差
                dx = points[i]['position'][0] - points[i - 1]['position'][0]
                dy = points[i]['position'][1] - points[i - 1]['position'][1]
                dz = points[i]['position'][2] - points[i - 1]['position'][2]

                # 计算速度分量
                vx = dx / dt
                vy = dy / dt
                vz = dz / dt

                # 计算总速度
                speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

                # 计算垂直速度和水平速度
                vertical_speed = vz
                horizontal_speed = math.sqrt(vx ** 2 + vy ** 2)

                # 计算水平方向（角度，度）
                horizontal_direction = math.degrees(math.atan2(vy, vx)) if horizontal_speed > 0.001 else None

                # 计算三维方向（俯仰角和偏航角）
                direction = None
                if speed > 0.001:
                    pitch = math.degrees(math.asin(vz / speed))
                    yaw = math.degrees(math.atan2(vy, vx))
                    direction = (pitch, yaw)

                velocities.append({
                    'point': points[i],
                    'velocity': (vx, vy, vz),
                    'speed': speed,
                    'direction': direction,
                    'vertical_speed': vertical_speed,
                    'horizontal_speed': horizontal_speed,
                    'horizontal_direction': horizontal_direction
                })

        return velocities

    def detect_motion_state(self, velocities: List[Dict], index: int) -> str:
        """检测当前点的运动状态"""
        if index >= len(velocities):
            return 'unknown'

        v = velocities[index]

        # 根据速度判断状态
        if v['speed'] < self.velocity_threshold:
            return '静止'

        # 判断垂直运动
        if abs(v['vertical_speed']) > abs(v['horizontal_speed']):
            if v['vertical_speed'] < -self.velocity_threshold:
                return '上升'
            elif v['vertical_speed'] > self.velocity_threshold:
                return '下降'

        # 水平运动
        if v['horizontal_speed'] > self.velocity_threshold:
            return '水平移动'

        return '运动中'

    def find_state_changes(self, velocities: List[Dict]) -> List[Dict]:
        """找到状态改变的点"""
        key_points = []

        if not velocities:
            return key_points

        # 注意：这里注释掉了起点，如果需要打印起点，请取消注释
        # key_points.append({
        #     'index': 0,
        #     'point': velocities[0]['point'],
        #     'position': velocities[0]['point']['position'],
        #     'timestamp': velocities[0]['point']['timestamp'],
        #     'state': self.detect_motion_state(velocities, 0),
        #     'reason': '轨迹起点'
        # })

        prev_state = self.detect_motion_state(velocities, 0)
        prev_direction = velocities[0]['horizontal_direction']

        # 寻找状态变化点
        for i in range(1, len(velocities)):
            current_state = self.detect_motion_state(velocities, i)
            current_direction = velocities[i]['horizontal_direction']

            # 1. 状态改变
            if current_state != prev_state:
                kp = {
                    'index': i,
                    'point': velocities[i]['point'],
                    'position': velocities[i]['point']['position'],
                    'timestamp': velocities[i]['point']['timestamp'],
                    'state': current_state,
                    'reason': f'状态从{prev_state}变为{current_state}'
                }
                key_points.append(kp)

                # --- 新增：打印发现的关键点 ---
                print(
                    f"[发现关键点] 类型: 状态改变 | 时间: {kp['timestamp']:.2f}s | 原因: {kp['reason']} | 位置: {kp['position']}")
                # ------------------------------

                prev_state = current_state

            # 2. 方向改变（仅在水平移动时）
            elif current_state == '水平移动' and prev_direction is not None and current_direction is not None:
                angle_diff = abs(current_direction - prev_direction)
                # 处理角度跨越180度的情况
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff > self.direction_threshold:
                    kp = {
                        'index': i,
                        'point': velocities[i]['point'],
                        'position': velocities[i]['point']['position'],
                        'timestamp': velocities[i]['point']['timestamp'],
                        'state': current_state,
                        'reason': f'水平方向改变{angle_diff:.1f}度'
                    }
                    key_points.append(kp)

                    # --- 新增：打印发现的关键点 ---
                    print(
                        f"[发现关键点] 类型: 方向改变 | 时间: {kp['timestamp']:.2f}s | 变化角度: {angle_diff:.1f}度 | 位置: {kp['position']}")
                    # ------------------------------

                    prev_direction = current_direction

        # 注意：这里注释掉了终点，如果需要打印终点，请取消注释
        # key_points.append({
        #     'index': len(velocities) - 1,
        #     'point': velocities[-1]['point'],
        #     'position': velocities[-1]['point']['position'],
        #     'timestamp': velocities[-1]['point']['timestamp'],
        #     'state': self.detect_motion_state(velocities, len(velocities) - 1),
        #     'reason': '轨迹终点'
        # })

        return key_points

    def analyze_flight_phases(self, velocities: List[Dict]) -> List[Dict]:
        """分析飞行的各个阶段"""
        if not velocities:
            return []

        phases = []
        phase_start = 0
        phase_state = self.detect_motion_state(velocities, 0)

        for i in range(1, len(velocities)):
            current_state = self.detect_motion_state(velocities, i)

            if current_state != phase_state:
                # 记录上一个阶段
                phases.append({
                    'start_index': phase_start,
                    'end_index': i - 1,
                    'start_point': velocities[phase_start]['point'],
                    'end_point': velocities[i - 1]['point'],
                    'state': phase_state,
                    'duration': (velocities[i - 1]['point']['timestamp'] - velocities[phase_start]['point'][
                        'timestamp']) / 1000,
                    'distance': self._calculate_distance(
                        velocities[phase_start]['point']['position'],
                        velocities[i - 1]['point']['position']
                    )
                })
                phase_start = i
                phase_state = current_state

        # 添加最后一个阶段
        phases.append({
            'start_index': phase_start,
            'end_index': len(velocities) - 1,
            'start_point': velocities[phase_start]['point'],
            'end_point': velocities[-1]['point'],
            'state': phase_state,
            'duration': (velocities[-1]['point']['timestamp'] - velocities[phase_start]['point']['timestamp']) / 1000,
            'distance': self._calculate_distance(
                velocities[phase_start]['point']['position'],
                velocities[-1]['point']['position']
            )
        })

        return phases

    def _calculate_distance(self, pos1: Tuple, pos2: Tuple) -> float:
        """计算两点之间的欧氏距离"""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )

    def extract_key_points(self, trajectory: List[Dict]) -> List[Dict]:
        """主函数：提取关键位置点"""
        # 1. 计算速度
        velocities = self.calculate_velocity(trajectory)

        if not velocities:
            return []

        # 2. 找到状态改变点
        state_changes = self.find_state_changes(velocities)

        # 3. 分析飞行阶段
        phases = self.analyze_flight_phases(velocities)

        # 4. 识别特定关键点
        specific_key_points = []

        # # 起飞点：第一个点
        # specific_key_points.append({
        #     'type': '起飞点',
        #     'point': velocities[0]['point'],
        #     'position': velocities[0]['point']['position'],
        #     'timestamp': velocities[0]['point']['timestamp']
        # })

        # # 悬停点：速度为零或极小的点
        # hovering_points = []
        # for i, v in enumerate(velocities):
        #     if v['speed'] < self.velocity_threshold:
        #         # 检查是否持续静止
        #         if i > 0 and i < len(velocities) - 1:
        #             # 前后点的速度也较小
        #             if velocities[i - 1]['speed'] < self.velocity_threshold or velocities[i + 1][
        #                 'speed'] < self.velocity_threshold:
        #                 hovering_points.append(i)
        #
        # # 从悬停点中筛选关键悬停点（保留变化点）
        # if hovering_points:
        #     # 找到悬停区间的中点
        #     hover_start = hovering_points[0]
        #     for i in range(1, len(hovering_points)):
        #         if hovering_points[i] - hovering_points[i - 1] > 5:  # 不连续
        #             # 记录上一个悬停区间
        #             mid = (hover_start + hovering_points[i - 1]) // 2
        #             specific_key_points.append({
        #                 'type': '悬停点',
        #                 'point': velocities[mid]['point'],
        #                 'position': velocities[mid]['point']['position'],
        #                 'timestamp': velocities[mid]['point']['timestamp'],
        #                 'duration': (velocities[hovering_points[i - 1]]['point']['timestamp'] -
        #                              velocities[hover_start]['point']['timestamp']) / 1000
        #             })
        #             hover_start = hovering_points[i]
        #
        #     # 添加最后一个悬停区间
        #     mid = (hover_start + hovering_points[-1]) // 2
        #     specific_key_points.append({
        #         'type': '悬停点',
        #         'point': velocities[mid]['point'],
        #         'position': velocities[mid]['point']['position'],
        #         'timestamp': velocities[mid]['point']['timestamp'],
        #         'duration': (velocities[hovering_points[-1]]['point']['timestamp'] -
        #                      velocities[hover_start]['point']['timestamp']) / 1000
        #     })

        # # 高度极值点
        # height_extremes = self._find_height_extremes(velocities)
        # for ext in height_extremes:
        #     specific_key_points.append({
        #         'type': ext['type'],
        #         'point': velocities[ext['index']]['point'],
        #         'position': velocities[ext['index']]['point']['position'],
        #         'timestamp': velocities[ext['index']]['point']['timestamp'],
        #         'height': -ext['height']  # AirSim中Z向下为正
        #     })

        # # 着陆点：最后的静止点
        # for i in range(len(velocities) - 1, max(0, len(velocities) - 50), -1):
        #     if velocities[i]['speed'] < self.velocity_threshold:
        #         specific_key_points.append({
        #             'type': '着陆点',
        #             'point': velocities[i]['point'],
        #             'position': velocities[i]['point']['position'],
        #             'timestamp': velocities[i]['point']['timestamp']
        #         })
        #         break

        # 5. 合并和去重
        # all_key_points = specific_key_points + state_changes
        # === 新增：状态稳定性过滤逻辑 ===
        stable_key_points = []

        # 设定最短状态持续时间阈值（秒），小于此时间视为波动
        MIN_DURATION_THRESHOLD = 0.5

        i = 0
        n = len(state_changes)

        while i < n:
            current_kp = state_changes[i]

            # 检查当前状态是否是一个短暂的波动
            # 条件：不是最后一个点，且当前状态持续时间过短
            if i < n - 1:
                next_kp = state_changes[i + 1]
                duration = next_kp['timestamp'] - current_kp['timestamp']

                if duration < MIN_DURATION_THRESHOLD:
                    # 这是一个短暂的波动状态 (例如: 上升 -> 下降 -> 水平移动 中的"下降")
                    # 策略：跳过这个中间状态，将前后状态合并

                    # 1. 解析前一个真实状态 (从 current_kp 的 reason 中解析，如 "状态从上升变为下降" -> "上升")
                    try:
                        prev_state = current_kp['reason'].split('从')[1].split('变')[0]
                    except:
                        prev_state = current_kp.get('state', '未知')

                    # 2. 获取下一个状态 (目标状态)
                    next_state = next_kp['state']

                    # 3. 修改下一个点的信息，使其看起来像是直接从前一个状态变过来的
                    # 这样就跳过了中间的波动状态
                    # 注意：我们需要创建一个新的字典副本，避免修改原始引用导致的问题
                    merged_kp = next_kp.copy()
                    merged_kp['reason'] = f'状态从{prev_state}变为{next_state}'

                    # 4. 将合并后的点加入列表
                    stable_key_points.append(merged_kp)

                    # 5. 跳过下一个点（因为它已经被合并处理了），i 前进 2 步
                    i += 2
                    continue

            # 如果不是短暂波动，或者是最后一个点，直接添加
            stable_key_points.append(current_kp)
            i += 1

        # === 过滤逻辑结束 ===

        # 如果有 specific_key_points (悬停点等)，需要在这里合并 stable_key_points
        all_key_points = specific_key_points + stable_key_points

        # 按时间戳排序
        all_key_points.sort(key=lambda x: x.get('timestamp', 0))

        # 最终去重 (基于时间戳)
        seen = set()
        unique_key_points = []
        for kp in all_key_points:
            ts = kp.get('timestamp', 0)
            # 对时间戳做一点微小容差去重，避免极短时间内重复
            # 如果只想简单去重：
            if ts not in seen:
                seen.add(ts)
                unique_key_points.append(kp)

        return unique_key_points

# 使用示例
def analyze_trajectory_file(file_path: str):
    # 读取轨迹
    utils = Path_eval_utils()
    trajectory = utils.to_path_ponit_array(file_path)

    if not trajectory:
        print("未读取到轨迹数据")
        return

    print(f"共读取到 {len(trajectory)} 个轨迹点")

    # 分析轨迹
    analyzer = TrajectoryAnalyzer()
    key_points = analyzer.extract_key_points(trajectory)

    # 打印关键点
    print("\n=== 识别到的关键位置点 ===")
    for i, kp in enumerate(key_points, 1):
        pos = kp['position']
        ts = kp['timestamp']
        kp_type = kp.get('type', kp.get('state', '未知'))

        print(f"\n{i}. 类型: {kp_type}")
        print(f"   时间戳: {ts}")
        print(f"   位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

        if 'reason' in kp:
            print(f"   原因: {kp['reason']}")
        if 'duration' in kp:
            print(f"   持续时间: {kp['duration']:.1f}秒")
        if 'height' in kp:
            print(f"   高度: {kp['height']:.2f}米")

    # 提取纯位置列表
    print("\n=== 关键点坐标列表 ===")
    positions = [kp['position'] for kp in key_points]
    for i, pos in enumerate(positions):
        print(f"点{i + 1}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    return key_points


# 如果需要处理上传的文件
def process_uploaded_file(file_content: str):
    """处理上传的文件内容"""
    # 保存到临时文件
    temp_file = "temp_trajectory.txt"
    with open(temp_file, 'w') as f:
        f.write(file_content)

    # 分析轨迹
    return analyze_trajectory_file(temp_file)


def print_key_points_by_category(key_points):
    """按类别打印关键点"""
    if not key_points:
        print("未找到关键点")
        return

    # 1. 按类型分类
    categories = {}
    for kp in key_points:
        # 确定类型（使用 get 方法避免 KeyError）
        kp_type = kp.get('type') or kp.get('state', '未知类型')

        if kp_type not in categories:
            categories[kp_type] = []
        categories[kp_type].append(kp)

    # 2. 按类别打印
    print("\n=== 关键点分类统计 ===")
    for category, points in categories.items():
        print(f"\n【{category}】- 共 {len(points)} 个点")
        print("-" * 50)

        for i, kp in enumerate(points, 1):
            # 提取位置信息
            if 'position' in kp:
                position = kp['position']
            elif 'point' in kp and 'position' in kp['point']:
                position = kp['point']['position']
            else:
                position = None

            # 提取时间戳并格式化
            ts_val = kp.get('timestamp', '未知时间')
            if isinstance(ts_val, (int, float)):
                timestamp = f"{ts_val:.2f}秒"  # 保留两位小数并加上单位
            else:
                timestamp = ts_val

            # 打印基本信息
            print(f"  {i}. 时间: {timestamp}")
            if position:
                print(f"     位置: X={position[0]:.3f}, Y={position[1]:.3f}, Z={position[2]:.3f}")

            # 打印额外信息
            if 'reason' in kp:
                print(f"     原因: {kp['reason']}")
            if 'duration' in kp:
                print(f"     持续时间: {kp['duration']:.1f}秒")
            if 'height' in kp:
                print(f"     高度: {kp['height']:.2f}米")


# 修改主程序部分
if __name__ == "__main__":
    # 读取轨迹数据
    utils = Path_eval_utils()
    # 请确保路径正确，如果还是报错请检查文件是否存在
    path_array = utils.to_path_ponit_array("2026-01-12-20-32-58/airsim_rec.txt")

    if path_array:
        # 分析轨迹
        analyzer = TrajectoryAnalyzer()
        key_points = analyzer.extract_key_points(path_array)

        # 使用新的分类打印函数
        print_key_points_by_category(key_points)

        # 额外功能：提取关键位置坐标用于导航
        print("\n=== 导航用关键坐标 ===")

        for i, kp in enumerate(key_points, 1):
            # 提取位置
            if 'position' in kp:
                pos = kp['position']
            elif 'point' in kp and 'position' in kp['point']:
                pos = kp['point']['position']
            else:
                pos = (0, 0, 0)

            # 提取时间戳
            ts = kp.get('timestamp', 0.0)

            # 打印带时间的航点信息
            print(f"航点{i}: 时间={ts:.2f}s, 位置=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

