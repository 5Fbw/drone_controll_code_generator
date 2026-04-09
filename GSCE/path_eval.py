from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import math


# === 四元数转偏航角辅助函数 ===
def quaternion_to_yaw(w, x, y, z):
    """
    将四元数转换为偏航角
    AirSim使用NED坐标系，North(北)=0°, East(东)=90°, South(南)=180°, West(西)=-90°
    """
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return yaw


class Path_eval_utils:
    def to_path_ponit_array(self, path_file: str):
        file_path = Path(path_file)
        if not file_path.exists():
            print(f"错误：文件 {path_file} 不存在")
            return []
        trajectory_data = []
        try:
            lines = file_path.read_text(encoding='utf-8').splitlines()
            for line in lines[1:]:  # 跳过首行
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    # 对应 AirSim 记录格式:
                    # VehicleName TimeStamp POS_X POS_Y POS_Z Q_W Q_X Q_Y Q_Z ...
                    timestamp = int(parts[1])
                    pos_x = float(parts[2])
                    pos_y = float(parts[3])
                    pos_z = float(parts[4])
                    q_w = float(parts[5])
                    q_x = float(parts[6])
                    q_y = float(parts[7])
                    q_z = float(parts[8])

                    point = {
                        'timestamp': timestamp,
                        'position': (pos_x, pos_y, pos_z),
                        'quaternion': (q_w, q_x, q_y, q_z)
                    }
                    trajectory_data.append(point)
                except (ValueError, IndexError):
                    continue

            if trajectory_data:
                start_timestamp = trajectory_data[0]['timestamp']
                for point in trajectory_data:
                    point['timestamp'] = (point['timestamp'] - start_timestamp) / 1000.0
        except Exception as e:
            print(f"读取错误: {e}")
            return []
        return trajectory_data


class TrajectoryAnalyzer:
    def __init__(self):
        # 降低速度阈值，捕捉慢速移动的开始时刻
        self.velocity_threshold = 0.05
        self.direction_threshold = 30  # 方向变化阈值(度)
        self.height_threshold = 0.2

    def calculate_velocity(self, points: List[Dict]) -> List[Dict]:
        if len(points) < 2:
            return []
        velocities = []
        for i in range(len(points)):
            q = points[i]['quaternion']
            current_yaw = quaternion_to_yaw(q[0], q[1], q[2], q[3])

            if i == 0:
                velocities.append({
                    'point': points[i],
                    'velocity': (0, 0, 0),
                    'speed': 0,
                    'direction': None,
                    'vertical_speed': 0,
                    'horizontal_speed': 0,
                    'horizontal_direction': None,
                    'yaw': current_yaw
                })
            else:
                dt = (points[i]['timestamp'] - points[i - 1]['timestamp'])
                if dt <= 0:
                    dt = 0.001
                dx = points[i]['position'][0] - points[i - 1]['position'][0]
                dy = points[i]['position'][1] - points[i - 1]['position'][1]
                dz = points[i]['position'][2] - points[i - 1]['position'][2]

                vx, vy, vz = dx / dt, dy / dt, dz / dt
                speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                vertical_speed = vz
                horizontal_speed = math.sqrt(vx ** 2 + vy ** 2)

                # 计算水平运动方向 (速度矢量方向)
                horizontal_direction = math.degrees(math.atan2(vy, vx)) if horizontal_speed > 0.001 else None

                direction = None
                if speed > 0.001:
                    pitch = math.degrees(math.asin(vz / speed)) if abs(vz) <= speed else 0
                    yaw = math.degrees(math.atan2(vy, vx))
                    direction = (pitch, yaw)

                velocities.append({
                    'point': points[i],
                    'velocity': (vx, vy, vz),
                    'speed': speed,
                    'direction': direction,
                    'vertical_speed': vertical_speed,
                    'horizontal_speed': horizontal_speed,
                    'horizontal_direction': horizontal_direction,
                    'yaw': current_yaw
                })
        return velocities

    def detect_motion_state(self, velocities: List[Dict], index: int) -> str:
        if index >= len(velocities):
            return 'unknown'
        v = velocities[index]
        # 优先判断静止
        if v['speed'] < self.velocity_threshold:
            return '静止'
        # 判断垂直运动
        if abs(v['vertical_speed']) > v['horizontal_speed']:
            if v['vertical_speed'] < -self.velocity_threshold:
                return '上升'
            if v['vertical_speed'] > self.velocity_threshold:
                return '下降'
        if v['horizontal_speed'] > self.velocity_threshold:
            return '水平移动'
        return '运动中'

    def find_state_changes(self, velocities: List[Dict]) -> List[Dict]:
        key_points = []
        if not velocities:
            return key_points

        prev_state = self.detect_motion_state(velocities, 0)
        # 记录上一次有效的水平移动方向，用于检测轨迹转向
        last_trajectory_direction = velocities[0]['horizontal_direction'] if prev_state == '水平移动' else None

        for i in range(1, len(velocities)):
            current_state = self.detect_motion_state(velocities, i)
            current_direction = velocities[i]['horizontal_direction']

            # === 1. 状态变化检测 ===
            if current_state != prev_state:
                kp = {
                    'index': i,
                    'point': velocities[i]['point'],
                    'position': velocities[i]['point']['position'],
                    'timestamp': velocities[i]['point']['timestamp'],
                    'state': current_state,
                    'yaw': velocities[i]['yaw'],
                    'reason': f'状态从{prev_state}变为{current_state}'
                }
                key_points.append(kp)
                prev_state = current_state

                # 如果进入水平移动，初始化轨迹方向基准
                if current_state == '水平移动' and current_direction is not None:
                    last_trajectory_direction = current_direction
                # 如果退出水平移动，清除轨迹方向基准
                elif current_state != '水平移动':
                    last_trajectory_direction = None

            # === 2. 轨迹方向变化检测 (独立于状态变化) ===
            # 仅当当前处于水平移动状态时检测
            elif current_state == '水平移动':
                # 更新基准：如果之前没有基准，先设置当前方向为基准
                if last_trajectory_direction is None and current_direction is not None:
                    last_trajectory_direction = current_direction

                # 核心逻辑：对比当前轨迹方向与上一次记录的轨迹方向
                if last_trajectory_direction is not None and current_direction is not None:
                    angle_diff = abs(current_direction - last_trajectory_direction)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                    # 如果轨迹方向变化超过阈值，记录关键点
                    # 这能捕捉到 Q不变但 vx/vy 改变的情况 (侧飞、倒飞)
                    if angle_diff > self.direction_threshold:
                        kp = {
                            'index': i,
                            'point': velocities[i]['point'],
                            'position': velocities[i]['point']['position'],
                            'timestamp': velocities[i]['point']['timestamp'],
                            'state': '轨迹转向点',
                            'yaw': velocities[i]['yaw'],  # 记录此时机头朝向
                            'reason': f'轨迹方向改变{angle_diff:.1f}度 (侧飞/倒飞)'
                        }
                        key_points.append(kp)
                        # 更新基准方向
                        last_trajectory_direction = current_direction

        return key_points

    def extract_key_points(self, trajectory: List[Dict]) -> List[Dict]:
        velocities = self.calculate_velocity(trajectory)
        if not velocities:
            return []
        raw_key_points = self.find_state_changes(velocities)
        stable_key_points = []
        MIN_DURATION_THRESHOLD = 1.0

        i = 0
        n = len(raw_key_points)
        while i < n:
            current_kp = raw_key_points[i]
            if i < n - 1:
                next_kp = raw_key_points[i + 1]
                duration = next_kp['timestamp'] - current_kp['timestamp']
                if duration < MIN_DURATION_THRESHOLD:
                    # 短暂波动合并
                    try:
                        prev_state = current_kp['reason'].split('从')[1].split('变')[0]
                    except:
                        prev_state = current_kp.get('state', '未知')
                    next_state = next_kp['state']
                    merged_kp = next_kp.copy()
                    merged_kp['reason'] = f'状态从{prev_state}变为{next_state}'
                    stable_key_points.append(merged_kp)
                    i += 2
                    continue
            stable_key_points.append(current_kp)
            i += 1

        # 添加起点
        if velocities:
            start_point = {
                'index': 0,
                'point': velocities[0]['point'],
                'position': velocities[0]['point']['position'],
                'timestamp': velocities[0]['point']['timestamp'],
                'state': self.detect_motion_state(velocities, 0),
                'yaw': velocities[0]['yaw'],
                'reason': '起点'
            }
            stable_key_points.insert(0, start_point)

            # 添加终点
            end_point = {
                'index': len(velocities) - 1,
                'point': velocities[-1]['point'],
                'position': velocities[-1]['point']['position'],
                'timestamp': velocities[-1]['point']['timestamp'],
                'state': '终点',
                'yaw': velocities[-1]['yaw'],
                'reason': '飞行结束'
            }
            stable_key_points.append(end_point)

        stable_key_points.sort(key=lambda x: x.get('timestamp', 0))

        # === 去重逻辑 ===
        if len(stable_key_points) > 1:
            unique_points = [stable_key_points[0]]
            for i in range(1, len(stable_key_points)):
                curr = stable_key_points[i]
                prev = unique_points[-1]
                if abs(curr['timestamp'] - prev['timestamp']) < 0.001:
                    if curr['state'] == '终点':
                        unique_points[-1] = curr
                else:
                    unique_points.append(curr)
            return unique_points
        return stable_key_points


def print_key_points_by_category(key_points):
    if not key_points:
        return
    categories = {}
    for kp in key_points:
        kp_type = kp.get('type') or kp.get('state', '未知类型')
        if kp_type not in categories:
            categories[kp_type] = []
        categories[kp_type].append(kp)

    print("\n=== 关键点分类统计 ===")
    for category, points in categories.items():
        print(f"\n【{category}】- 共 {len(points)} 个点")
        print("-" * 70)
        for i, kp in enumerate(points, 1):
            if 'position' in kp:
                position = kp['position']
            elif 'point' in kp and 'position' in kp['point']:
                position = kp['point']['position']
            else:
                position = None
            ts_val = kp.get('timestamp', '未知时间')
            timestamp = f"{ts_val:.2f}s" if isinstance(ts_val, (int, float)) else ts_val
            yaw_val = kp.get('yaw', '未知')
            yaw_str = f"{yaw_val:.1f}°" if isinstance(yaw_val, (int, float)) else yaw_val
            print(f" {i}. 时间: {timestamp} | 偏航角: {yaw_str}")
            if position:
                print(f" 位置: X={position[0]:.3f}, Y={position[1]:.3f}, Z={position[2]:.3f}")
            if 'reason' in kp:
                print(f" 原因: {kp['reason']}")


if __name__ == "__main__":
    utils = Path_eval_utils()
    # 请修改为你的实际文件路径
    path_array = utils.to_path_ponit_array("D:/AirSim_Data/2026-03-19-17-09-26/airsim_rec.txt")

    if path_array:
        analyzer = TrajectoryAnalyzer()
        key_points = analyzer.extract_key_points(path_array)

        print(f"数据读取成功，共 {len(path_array)} 个点，时长 {path_array[-1]['timestamp']:.2f}s")
        print_key_points_by_category(key_points)

        print("\n=== 导航用关键坐标 ===")
        for i, kp in enumerate(key_points, 1):
            pos = kp.get('position', (0, 0, 0))
            ts = kp.get('timestamp', 0.0)
            yaw = kp.get('yaw', 0.0)
            print(f"航点{i}: 时间={ts:.2f}s, 偏航角={yaw:.1f}°, 位置=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
