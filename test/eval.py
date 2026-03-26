from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance
#衡量轨迹相似度，但思路不正确
class Utils:
    def to_path_ponit_array(self, path_file:str):
        """
            已知轨迹文件名称，读取文件内容并解析为轨迹点数组。

            Args:
                path_file (str): 轨迹文件的路径（相对路径或绝对路径）

            Returns:
                list: 包含轨迹点字典的列表，每个字典包含时间戳、位置和姿态信息。
                      格式: [{'timestamp': int, 'position': (x,y,z), 'quaternion': (w,x,y,z)}, ...]
                      如果文件读取失败，返回空列表。
        """
        file_path = Path(path_file)

        # 1. 检查文件是否存在
        if not file_path.exists():
            print(f"错误：文件 {path_file} 不存在")
            return []

        trajectory_data = []

        try:
            # 2. 打开并读取文件 (假设文件是 utf-8 编码，根据实际情况可调整 encoding)
            # 使用 readlines() 一次性读取所有行，处理小文件很方便
            lines = file_path.read_text(encoding='utf-8').splitlines()

            # 3. 遍历每一行
            # 假设第一行是表头，我们从第二行（索引1）开始处理
            for line in lines[1:]:
                line = line.strip()  # 去除首尾空白字符

                # 跳过空行
                if not line:
                    continue

                # 4. 分割字符串 (假设是以制表符或空格分隔)
                parts = line.split()

                # 简单的校验，确保列数足够
                # 格式: VehicleName, TimeStamp, POS_X, POS_Y, POS_Z, Q_W, Q_X, Q_Y, Q_Z, ImageFile...
                if len(parts) < 9:
                    continue

                try:
                    # 提取数据并转换类型
                    timestamp = int(parts[1])

                    # 位置 (POS_X, POS_Y, POS_Z)
                    pos_x = float(parts[2])
                    pos_y = float(parts[3])
                    pos_z = float(parts[4])

                    # 姿态四元数 (Q_W, Q_X, Q_Y, Q_Z)
                    q_w = float(parts[5])
                    q_x = float(parts[6])
                    q_y = float(parts[7])
                    q_z = float(parts[8])

                    # 构建轨迹点结构
                    point = {
                        'timestamp': timestamp,
                        'position': (pos_x, pos_y, pos_z),
                        'quaternion': (q_w, q_x, q_y, q_z)
                    }

                    trajectory_data.append(point)

                except (ValueError, IndexError) as e:
                    # 如果某一行数据格式不对，打印警告并跳过该行
                    print(f"警告：跳过格式错误的行 -> {line} (错误: {e})")
                    continue

        except Exception as e:
            print(f"读取文件发生未知错误: {e}")
            return []

        return trajectory_data
    def write_to_json(self,json_file_path,trajectory_data):
        actual_path_list = [list(point['position']) for point in trajectory_data]

        # 3. 读取现有的JSON文件
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"错误：文件 {json_file_path} 不存在，将创建新文件。")
            data = {}  # 如果文件不存在，初始化一个空字典
        except json.JSONDecodeError:
            print(f"错误：文件 {json_file_path} 格式不正确。")
            return

        # 4. 更新 'actual_path' 字段
        data['actual_path'] = actual_path_list

        # 5. 将更新后的数据写回文件
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"成功：已将 {len(actual_path_list)} 个轨迹点写入 {json_file_path} 的 'actual_path' 字段。")
        except IOError as e:
            print(f"错误：写入文件失败 - {e}")
    def  read_paths_from_json(self, json_file_path):
        """
            读取JSON文件，提取exp_path和actual_path字段

            Args:
                json_file_path (str): JSON文件路径

            Returns:
                tuple: (exp_path, actual_path) 两个列表
        """
        try:
            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取exp_path和actual_path字段
            exp_path = data.get('exp_path', [])
            actual_path = data.get('actual_path', [])

            # 验证数据类型
            if not isinstance(exp_path, list):
                print("警告: exp_path不是列表类型")
                exp_path = []

            if not isinstance(actual_path, list):
                print("警告: actual_path不是列表类型")
                actual_path = []

            print(f"成功读取exp_path: {len(exp_path)}个点")
            print(f"成功读取actual_path: {len(actual_path)}个点")

            return exp_path, actual_path

        except FileNotFoundError:
            print(f"错误: 文件 {json_file_path} 不存在")
            return [], []
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return [], []
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return [], []

    def calculate_metrics(self, exp_path, actual_path):
        """
        计算轨迹评估指标
        Args:
            exp_path: list of lists or np.array [[x,y,z], ...]
            actual_path: list of lists or np.array [[x,y,z], ...]
        Returns:
            dict: 包含各项指标的字典
        """
        # 转换为numpy数组以便计算
        exp_arr = np.array(exp_path)
        act_arr = np.array(actual_path)

        # 防止空数组报错
        if exp_arr.size == 0 or act_arr.size == 0:
            return {"error": "路径数据为空"}

        metrics = {}

        # --- 1. 终点状态指标 ---
        final_exp = exp_arr[-1]
        final_act = act_arr[-1]

        # 终点欧氏距离
        final_error = np.linalg.norm(final_exp - final_act)
        metrics['终点误差'] = final_error

        # 终点高度偏差
        metrics['终点高度偏差'] = abs(final_exp[2] - final_act[2])

        # --- 2. 路径长度指标 ---
        def compute_path_length(path_arr):
            diffs = np.diff(path_arr, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            return np.sum(dists)

        len_exp = compute_path_length(exp_arr)
        len_act = compute_path_length(act_arr)

        metrics['期望路径长度'] = len_exp
        metrics['实际路径长度'] = len_act
        metrics['路径长度比率'] = len_act / len_exp if len_exp > 0 else 0

        # --- 3. 轨迹偏差指标 (基于最近点匹配) ---
        # 计算实际路径中每个点到期望路径的最短距离
        # 使用 cdist 计算所有点对之间的距离矩阵
        dist_matrix = cdist(act_arr, exp_arr)
        # 取每一行的最小值，即每个实际点到期望路径集的最小距离
        min_dists = np.min(dist_matrix, axis=1)

        metrics['平均偏差'] = np.mean(min_dists)
        metrics['最大偏差'] = np.max(min_dists)
        metrics['RMSE'] = np.sqrt(np.mean(min_dists ** 2))

        # --- 4. 形状相似度指标 ---
        # Hausdorff Distance (需要安装 scipy)
        # 定义为 max(min(从A到B), min(从B到A))
        # scipy.spatial.distance.directed_hausdorff 计算单向 hausdorff
        forward_h = distance.directed_hausdorff(act_arr, exp_arr)[0]
        backward_h = distance.directed_hausdorff(exp_arr, act_arr)[0]
        metrics['Hausdorff距离'] = max(forward_h, backward_h)
        print(metrics)
        return metrics

    def read_paths_from_json(self,json_file_path):
        """
        读取JSON文件，提取exp_path和actual_path字段

        Args:
            json_file_path (str): JSON文件路径

        Returns:
            tuple: (exp_path, actual_path) 两个列表
        """
        try:
            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取exp_path和actual_path字段
            exp_path = data.get('exp_path', [])
            actual_path = data.get('actual_path', [])

            # 验证数据类型
            if not isinstance(exp_path, list):
                print("警告: exp_path不是列表类型")
                exp_path = []

            if not isinstance(actual_path, list):
                print("警告: actual_path不是列表类型")
                actual_path = []

            print(f"成功读取exp_path: {len(exp_path)}个点")
            print(f"成功读取actual_path: {len(actual_path)}个点")

            return exp_path, actual_path

        except FileNotFoundError:
            print(f"错误: 文件 {json_file_path} 不存在")
            return [], []
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return [], []
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return [], []

    def calculate_metrics(self, exp_path, actual_path, metadata=None):
        """
        计算轨迹评估指标（考虑高度偏移，并使用线性重采样计算偏差）
        修改：完成率和成功与否使用原始数据计算
        """
        # 转换为numpy数组
        exp_arr = np.array(exp_path)
        act_arr = np.array(actual_path)

        if exp_arr.size == 0 or act_arr.size == 0:
            return {"error": "路径数据为空"}

        # 获取高度偏移量
        height_offset = 0
        if metadata and 'origin_pos' in metadata:
            height_offset = metadata['origin_pos'][2] if len(metadata['origin_pos']) > 2 else 0
            print(f"检测到高度偏移: {height_offset}米")

        metrics = {}

        # --- 1. 终点状态指标 ---
        final_exp = exp_arr[-1]
        final_act = act_arr[-1]

        # 终点误差（考虑高度偏移）
        final_error = np.linalg.norm(final_exp - final_act)
        metrics['终点误差'] = final_error

        # 终点高度偏差
        metrics['终点高度偏差'] = abs(final_exp[2] - final_act[2])

        # --- 2. 路径长度指标 ---
        def compute_path_length(path_arr):
            if len(path_arr) < 2:
                return 0
            diffs = np.diff(path_arr, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            return np.sum(dists)

        len_exp = compute_path_length(exp_arr)
        len_act = compute_path_length(act_arr)

        metrics['期望路径长度'] = len_exp
        metrics['实际路径长度'] = len_act
        metrics['路径长度比率'] = len_act / len_exp if len_exp > 0 else 0

        # --- 3. 轨迹偏差指标 (使用 np.interp 进行线性重采样) ---

        # 定义重采样函数
        def resample_path(path_arr, num_points):
            if len(path_arr) < 2:
                return path_arr

                # 计算累计路径距离作为插值基准
            diffs = np.diff(path_arr, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            cum_dist = np.insert(np.cumsum(dists), 0, 0)

            # 生成新的均匀距离坐标
            new_cum_dist = np.linspace(0, cum_dist[-1], num_points)

            # 使用 np.interp 进行线性插值 (避免 scipy 版本问题)
            new_x = np.interp(new_cum_dist, cum_dist, path_arr[:, 0])
            new_y = np.interp(new_cum_dist, cum_dist, path_arr[:, 1])
            new_z = np.interp(new_cum_dist, cum_dist, path_arr[:, 2])

            # 计算新坐标
            new_path = np.vstack((new_x, new_y, new_z)).T
            return new_path

        # 确定重采样的目标数量：取两者中较大的那个
        max_points = max(len(exp_arr), len(act_arr))
        if max_points < 2:
            max_points = 2

        print(f"开始重采样：目标点数 {max_points} (Exp: {len(exp_arr)}, Act: {len(act_arr)})")

        # 执行重采样 (仅用于计算 RMSE 等指标)
        exp_resampled = resample_path(exp_arr, max_points)
        act_resampled = resample_path(act_arr, max_points)

        # 计算重采样后的偏差
        point_errors = np.linalg.norm(exp_resampled - act_resampled, axis=1)

        metrics['平均偏差'] = np.mean(point_errors)
        metrics['RMSE'] = np.sqrt(np.mean(point_errors ** 2))

        # --- 4. 完成率和成功与否计算 (使用原始数据) ---

        # 计算实际轨迹上的每个点，距离期望轨迹最近的点的距离
        # 这里使用 act_arr (原始实际数据) 和 exp_arr (原始期望数据)
        dist_matrix = cdist(act_arr, exp_arr)

        # 获取每个实际点到期望轨迹的最短距离
        min_dists_to_exp = np.min(dist_matrix, axis=0)
        # --- 新增：打印最短距离 ---
        print(f"--- 逐点最短距离检查 (共 {len(min_dists_to_exp)} 个实际点) ---")
        for i, dist in enumerate(min_dists_to_exp):
            # :.4f 保留4位小数
            print(f"实际点 {i:03d} 距期望轨迹最近点的距离: {dist:.4f} 米")
        print("-" * 20)
        # 设定容差阈值 (单位：米)
        tolerance = 1

        # 统计满足条件的点的数量
        covered_points_count = np.sum(min_dists_to_exp < tolerance)

        # 计算完成率
        if len(exp_arr) > 0:
            completion_rate = (covered_points_count / len(exp_arr)) * 100
        else:
            completion_rate = 0.0

        metrics['完成率'] = completion_rate

        # 判定成功与否
        if completion_rate >= 100.0:
            metrics['成功'] = True
        else:
            metrics['成功'] = False

        # --- 5. 形状相似度指标 (Hausdorff) ---
        # Hausdorff 距离对点数不一致不敏感，通常直接使用原始数据计算即可
        # 使用原始数据计算
        forward_h = distance.directed_hausdorff(act_resampled, exp_resampled)[0]
        backward_h = distance.directed_hausdorff(exp_resampled, act_resampled)[0]
        metrics['Hausdorff距离'] = max(forward_h, backward_h)

        print("-" * 20)
        print(f"评估结果汇总:")
        print(f"完成率: {metrics['完成率']:.2f}%")
        print(f"成功: {metrics['成功']}")
        print("-" * 20)
        print(metrics)
        return metrics

utils = Utils()
path_array = utils.to_path_ponit_array("2026-01-13-17-10-21/airsim_rec.txt")
print(path_array[:5])
json_file = "dataWrong.json"
utils.write_to_json(json_file, path_array)
exp_path, actual_path = utils.read_paths_from_json(json_file)
utils.calculate_metrics(exp_path, actual_path)

