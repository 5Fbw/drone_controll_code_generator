
language_name = "python"
lib_name = "airsim"
model_name = "Qwen"
output_file_path = "airsim01.py"
seed_code = '''
    -向上/向下移动 moveToZAsync(-3, 2).join() 以2m/s的速度移动到全局坐标系3m高度处
    -以机身坐标系以指定速度移动指定时间 moveByVelocityBodyFrameAsync(1, 0, 0, 10).join() 当x = 1时以机身坐标向前以1m/s的速度飞行10s即10m
                                moveByVelocityBodyFrameAsync(0, 1, 0, 5).join() 当y = 1时以机身坐标向右以1m/s的速度飞行5s即5m
    -以全局坐标系以指定速度移动指定时间 moveToPositionAsync(-10, 0, 0, 1).join() 在全局坐标以1m/s速度南移动10m的速度
    -转向 rotateToYawAsync(float yaw = 90, float margin = 5f, float timeout_sec = 60).join()转向正东方 
'''
security_check_flag = True
security_constraints = {
    "max_velocity": 3,
    "max_height": 15
}

from enum import Enum

class Status(Enum):
    """定义代码审核状态的枚举"""
    PENDING = "待审核"
    CORRECT = "正确"
    ERROR = "错误"

api_key_new = "sk-f25779b5a1464f97bd5395cb954154d3"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"