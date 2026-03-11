# 基于LangGraph的无人机控制代码生成多智能体系统

## 项目概述

本项目使用LangGraph框架实现了handoffs模式的多智能体交互系统，用于生成无人机控制代码。系统包含以下智能体：

1. **Movement Extractor Agent**：负责将自然语言指令解析为结构化的动作序列
2. **Code Generator Agent**：负责将结构化动作序列转换为可执行的AirSim代码
3. **Code Checker Agent**：负责检查生成的代码是否符合目标动作序列
4. **Supervisor Agent**：负责协调各个智能体之间的交互

## 系统架构

```
自然语言指令
    ↓
[Movement Extractor Agent] → 解析指令为动作序列
    ↓
[Code Generator Agent] → 生成AirSim代码
    ↓
[Code Checker Agent] → 检查代码正确性
    ↓
[Supervisor Agent] → 判断是否需要修正
    ↓ (如果需要修正)
[Code Checker Agent] → 重新检查
    ↓ (如果通过)
最终代码输出
```

## 安装依赖

```bash
pip install langgraph langchain-core langchain-openai
```

## 使用方法

### 基本使用

```python
from adapt_langgraph.drone_code_agent import run_langgraph_agent

instruction = "起飞至10m高度，向右飞行5m，顺时针旋转45度，向前飞行10m"
result = run_langgraph_agent(instruction)

print(f"生成的代码:\n{result['code_output']['code']}")
```

### 自定义迭代次数

```python
result = run_langgraph_agent(instruction, max_iterations=5)
```

## 核心功能

### 1. 指令解析
- 将自然语言指令转换为结构化动作序列
- 支持起飞、降落、移动、转向等动作
- 自动计算位置和朝向变化

### 2. 代码生成
- 根据动作序列生成AirSim控制代码
- 自动添加控制解锁和结束代码
- 支持多种移动方式（全局坐标系、机身坐标系）

### 3. 代码检查
- 验证生成的代码是否符合目标动作序列
- 检查速度、距离等参数是否正确
- 自动修正不符合要求的代码

### 4. 多轮迭代
- 支持多轮代码检查和修正
- 可配置最大迭代次数
- 自动终止条件判断

## 与原系统的对比

### 原系统（main.py）
- 线性流程：指令解析 → 代码生成 → 代码检查
- 固定的执行顺序
- 缺乏灵活的智能体交互

### 新系统（LangGraph）
- 灵活的智能体交互
- 支持条件分支和循环
- 可扩展的架构设计
- 更好的错误处理和重试机制

## 配置说明

系统使用与原项目相同的配置文件（config.py）：
- API密钥和基础URL
- 语言和库名称
- 安全约束参数

## 示例

```python
# 示例1：简单指令
instruction = "起飞至10m高度"
result = run_langgraph_agent(instruction)

# 示例2：复杂指令
instruction = "起飞，以每秒1米的速度升空至10米高度。前进5米,然后转向正西，并以每秒1米的速度前进10米。然后降落。"
result = run_langgraph_agent(instruction, max_iterations=5)

# 示例3：多步动作
instruction = "起飞至10m高度，向右飞行5m，顺时针旋转45度，向前飞行10m"
result = run_langgraph_agent(instruction)
```

## 注意事项

1. 确保已安装所有依赖项
2. 确保config.py中的API密钥有效
3. 建议设置合理的max_iterations值（默认为3）
4. 对于复杂指令，可能需要更多的迭代次数

## 扩展性

系统设计具有良好的扩展性，可以轻松添加新的智能体：
- 添加安全检查智能体
- 添加路径规划智能体
- 添加仿真验证智能体
- 添加日志记录智能体

## 许可证

本项目遵循与主项目相同的许可证。
