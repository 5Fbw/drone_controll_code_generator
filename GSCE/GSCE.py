# from tkinter.scrolledtext import example
# from tkinter.scrolledtext import example

from jedi.inference import infer_expr_stmt
from openai import skills

import LLm_provider


def GSCE(llm_provider, human_instruction: str):
    print(human_instruction)
    guidelines = """
        ## Guidelines
        You are an code writer helping me with the AirSim simulator for drone control.
        You should think step by step, you can decompose task into small steps.
        When I ask you to do something, you are supposed to give me Python code is needed to achieve that task in one script, do not generate functions unless I ask you do so.
        You are only allowed to use the functions I have defined for you, you should never use any other hypothetical functions that you think might exist.
        You can use simple Python functions from libraries such as math and numpy. Do not import other python libraries.
        Do not use additional if statements or loops.
    """
    # constraints = ""
    constraints = """
        ## Constraints（回应示例）
        Important drone directional information:
        The horizontal axises are Y and X, vertical axis is Z. When rotating drone, right or clockwise means positive.
        aw.fly_to([x, y, z]) function uses NED coordinate system in world coordinates, positive X axis is North/forward, positive Y axis is East/right, positive Z axis is Down.
        In terms of drone's body frame axis conventions: forward means positive X axis, right means positive Y axis, up means positive Z axis.
        When asked to move in drones body frame, you should transform drone body frame movements to world coordinates. Because aw.fly_to([x, y, z]) function flies the drone in world coordinates.
        Below is the coordinates transforming equation: x, y, z are the world coordinate movements, x', y', z' are the drone body frame coordinate movements. Please note that we are employing a negative value for "theta".
        x = x'*cos(-theta) + y'*sin(-theta)
        y = -x'*sin(-theta) + y'*cos(-theta)
        z = -z'
    """
    skill_apis = """
        ## Skill Apis
        Here are some functions you can use to command the drone.

        aw.takeoff() - takes off the drone.
        aw.land() - lands the drone.
        aw.go_home() - return drone to home.
        aw.fly_to([x, y, z]) - flies the drone to the position specified as a list of three arguments corresponding to world XYZ coordinates. The flying speed is 2 meters per second.
        aw.get_yaw() - returns the current yaw of the drone in degrees.
        aw.set_yaw(yaw) - sets the yaw of the drone to the specified value in degrees.
        aw.get_drone_position() - returns the current position of the drone as a list of 3 floats corresponding to world XYZ coordinates.
    """
    # examples = ""
    examples = """
        ## examples
        Here are some examples:

        Query: "Fly 10 meters up."
        Answer:

        ```
        ## Step: fly 10 meters up.
        current_position = aw.get_drone_position() # Get the current position of the drone
        # fly up corresponds to negative Z direction in world coordinates, so -10 for z.
        aw.fly_to([current_position[0], current_position[1], current_position[2] - 10])
        ```


        Query: "Turn 90 degrees left, then fly 5 meters drone's left in the drone's body frame."
        Answer:

        ```
        ## Step: turn 90 degrees left.
        current_yaw = aw.get_yaw()  # Get the current yaw of the drone
        new_yaw = current_yaw - 90  # Calculate the new yaw for a 90 degrees left turn
        aw.set_yaw(new_yaw)  # Set the drone's yaw to turn 90 degrees left

        ## Step: fly 5 meters drone's left in the drone's body frame.
        current_position = aw.get_drone_position() # Retrieve the current XYZ position of the drone
        # Calculate movement in the drone's body frame
        body_x = 0  # move 0 in drone's x axis
        body_y = -5 # move left means -5 in drone's y axis
        body_z = 0 # move 0 in drone's z axis
        # Use transform equation to transform the drone's body frame movements to the world frame movments
        # x = x'*cos(-theta) + y'*sin(-theta), y = -x'*sin(-theta) + y'*cos(-theta), z = -z'
        current_yaw = aw.get_yaw() # get current yaw angle
        radian_yaw = math.radians(-current_yaw)
        world_x = current_position[0] + body_x * math.cos(radian_yaw) + body_y * math.sin(radian_yaw) # x = x'*cos(-theta) + y'*sin(-theta)
        world_y = current_position[1] - body_x * math.sin(radian_yaw) + body_y * math.cos(radian_yaw) # y = -x'*sin(-theta) + y'*cos(-theta)
        world_z = current_position[2] - body_z # z = -z'
        # Command the drone to the new position
        aw.fly_to([world_x, world_y, world_z])
        ```


        Query: "Fly the drone in the top-right direction at an angle of 45 degrees from the horizontal axis, in the YZ plane of drone's body frame for a distance of 10 meters."
        Answer:

        ```
        ## Step: fly the drone in the top-right direction at an angle of 45 degrees from the horizontal axis, in the YZ plane of drone's body frame for a distance of 10 meters. Predicted state change [0,7.07,-7.07,0].
        current_position = aw.get_drone_position()  # get current XYZ coordinates
        distance = 10  # distance to fly

        # Calculate movement in Y and Z directions
        # the horizontal axis is Y, so the angle is 45 from the Y axis.
        # thus, Y = d * cos(angle), Z = d * sin(angle)
        angle_degrees = 45
        angle_radians = math.radians(angle_degrees)
        delta_y = distance * math.cos(angle_radians) # Y = d * cos(angle)
        delta_z = distance * math.sin(angle_radians) # Z = d * sin(angle)

        # We are moving up and right for top-right direction.
        # Calculate movement in the drone's body frame
        body_x = 0  # move 0 in drone's x axis
        body_y = delta_y # move right means positive in drone's y axis
        body_z = delta_z # move top means positive in drone's z axis

        # Use transform equation to transform the drone's body frame movements to the world frame movments
        # x = x'*cos(-theta) + y'*sin(-theta), y = -x'*sin(-theta) + y'*cos(-theta), z = -z'
        current_yaw = aw.get_yaw() # get current yaw angle
        radian_yaw = math.radians(-current_yaw)
        world_x = current_position[0] + body_x * math.cos(radian_yaw) + body_y * math.sin(radian_yaw) # x = x'*cos(-theta) + y'*sin(-theta)
        world_y = current_position[1] - body_x * math.sin(radian_yaw) + body_y * math.cos(radian_yaw) # y = -x'*sin(-theta) + y'*cos(-theta)
        world_z = current_position[2] - body_z # z = -z'
        # Command the drone to the new position
        aw.fly_to([world_x, world_y, world_z])
    """
    # guidelines = """
    #         ## 指南
    #         你是一名代码编写员，协助我使用 AirSim 模拟器进行无人机控制。
    #         你应该一步步思考，可以将任务分解为小步骤。
    #         当我要求你做某事时，你应该在一个脚本中提供实现该任务所需的 Python 代码，除非我要求，否则不要生成函数。
    #         你只允许使用我为你定义的函数，绝对不要使用任何你认为可能存在的假设性函数。
    #         你可以使用 math 和 numpy 等库中的简单 Python 函数。不要导入其他 python 库。
    #         不要使用额外的 if 语句或循环。
    #     """
    # constraints = """
    #         ## 约束条件（回应示例）
    #         重要的无人机方向信息：
    #         水平轴是 Y 和 X，垂直轴是 Z。旋转无人机时，向右或顺时针表示正值。
    #         aw.fly_to([x, y, z]) 函数在世界坐标系中使用 NED 坐标系，X 轴正方向为北/前，Y 轴正方向为东/右，Z 轴正方向为下。
    #         关于无人机机体坐标轴约定：前方表示 X 轴正方向，右侧表示 Y 轴正方向，上方表示 Z 轴正方向。
    #         当被要求在无人机机体坐标系中移动时，你应该将无人机机体坐标系的运动转换为世界坐标系。因为 aw.fly_to([x, y, z]) 函数是在世界坐标系中飞行无人机的。
    #         下面是坐标变换方程：x, y, z 是世界坐标系移动值，x', y', z' 是无人机机体坐标系移动值。请注意，我们使用的是负的“theta”值。
    #         x = x'*cos(-theta) + y'*sin(-theta)
    #         y = -x'*sin(-theta) + y'*cos(-theta)
    #         z = -z'
    #     """
    # skill_apis = """
    #         ## 技能接口
    #         这里有一些你可以用来控制无人机的函数。
    #
    #         aw.takeoff() - 无人机起飞。
    #         aw.land() - 无人机降落。
    #         aw.go_home() - 无人机返航。
    #         aw.fly_to([x, y, z]) - 将无人机飞到指定位置，该位置由对应世界 XYZ 坐标的三个参数组成的列表指定。飞行速度为每秒 2 米。
    #         aw.get_yaw() - 返回无人机当前的偏航角（以度为单位）。
    #         aw.set_yaw(yaw) - 将无人机的偏航角设置为指定值（以度为单位）。
    #         aw.get_drone_position() - 返回无人机当前位置，作为对应世界 XYZ 坐标的 3 个浮点数列表。
    #     """
    # examples = """
    # ## 示例
    #     这里有一些示例：
    #
    #     查询: "向上飞 10 米。"
    #     回答:
    #
    #      ## 步骤：向上飞 10 米。
    #     current_position = aw.get_drone_position() # 获取无人机当前位置
    #     # 向上飞对应世界坐标系中的 Z 轴负方向，所以 z 为 -10。
    #     aw.fly_to([current_position[0], current_position[1], current_position[2] - 10])
    #
    #
    #     查询: "向左转 90 度，然后在无人机机体坐标系中向无人机左侧飞 5 米。"
    #     回答:
    #
    #      ## 步骤：向左转 90 度。
    #     current_yaw = aw.get_yaw() # 获取无人机当前的偏航角
    #     new_yaw = current_yaw - 90 # 计算向左转 90 度的新偏航角
    #     aw.set_yaw(new_yaw) # 设置无人机的偏航角向左转 90 度
    #
    #     ## 步骤：在无人机机体坐标系中向无人机左侧飞 5 米。
    #     current_position = aw.get_drone_position() # 获取无人机当前的 XYZ 位置
    #     # 计算无人机机体坐标系中的移动
    #     body_x = 0  # 在无人机 x 轴移动 0
    #     body_y = -5 # 向左移动意味着无人机 y 轴为 -5
    #     body_z = 0 # 在无人机 z 轴移动 0
    #     # 使用变换方程将无人机机体坐标系的移动转换为世界坐标系的移动
    #     # x = x'*cos(-theta) + y'*sin(-theta), y = -x'*sin(-theta) + y'*cos(-theta), z = -z'
    #     current_yaw = aw.get_yaw() # 获取当前偏航角
    #     radian_yaw = math.radians(-current_yaw)
    #     world_x = current_position[0] + body_x * math.cos(radian_yaw) + body_y * math.sin(radian_yaw) # x = x'*cos(-theta) + y'*sin(-theta)
    #     world_y = current_position[1] - body_x * math.sin(radian_yaw) + body_y * math.cos(radian_yaw) # y = -x'*sin(-theta) + y'*cos(-theta)
    #     world_z = current_position[2] - body_z # z = -z'
    #     # 控制无人机飞向新位置
    #     aw.fly_to([world_x, world_y, world_z])
    #
    #
    #     查询: "在无人机机体坐标系的 YZ 平面内，沿与水平轴成 45 度角的右上方向飞行 10 米。"
    #     回答:
    #
    #      ## 步骤：在无人机机体坐标系的 YZ 平面内，沿与水平轴成 45 度角的右上方向飞行 10 米。预测状态变化 [0, 7.07, -7.07, 0]。
    #     current_position = aw.get_drone_position() # 获取当前 XYZ 坐标
    #     distance = 10 # 飞行距离
    #
    #     # 计算 Y 和 Z 方向的移动
    #     # 水平轴是 Y，所以角度是从 Y 轴算起的 45 度。
    #     # 因此，Y = d * cos(angle), Z = d * sin(angle)
    #     angle_degrees = 45
    #     angle_radians = math.radians(angle_degrees)
    #     delta_y = distance * math.cos(angle_radians) # Y = d * cos(angle)
    #     delta_z = distance * math.sin(angle_radians) # Z = d * sin(angle)
    #
    #     # 我们向右上方移动。
    #     # 计算无人机机体坐标系中的移动
    #     body_x = 0  # 在无人机 x 轴移动 0
    #     body_y = delta_y # 向右移动意味着无人机 y 轴为正值
    #     body_z = delta_z # 向上移动意味着无人机 z 轴为正值
    #
    #     # 使用变换方程将无人机机体坐标系的移动转换为世界坐标系的移动
    #     # x = x'*cos(-theta) + y'*sin(-theta), y = -x'*sin(-theta) + y'*cos(-theta), z = -z'
    #     current_yaw = aw.get_yaw() # 获取当前偏航角
    #     radian_yaw = math.radians(-current_yaw)
    #     world_x = current_position[0] + body_x * math.cos(radian_yaw) + body_y * math.sin(radian_yaw) # x = x'*cos(-theta) + y'*sin(-theta)
    #     world_y = current_position[1] - body_x * math.sin(radian_yaw) + body_y * math.cos(radian_yaw) # y = -x'*sin(-theta) + y'*cos(-theta)
    #     world_z = current_position[2] - body_z # z = -z'
    #     # 控制无人机飞向新位置
    #     aw.fly_to([world_x, world_y, world_z])
    # """
    prompt = f"""
        {guidelines}
        {constraints}
        {skill_apis}
        {examples}
        ## query
        {human_instruction}
        """
    # [修改点] 接收 qwen_api 返回的元组 (文本, Token信息)
    response, usage_info = llm_provider.qwen_api(prompt, stream=False)

    # [修改点] 一并返回
    return response, usage_info
# query ="Turn 60 degrees clockwise, then fly 10 meters forward in the drone's body frame."
# llm_provider = LLm_provider.LLMProvider(model_name="qwen3-8b")
# GSCE(llm_provider,query)