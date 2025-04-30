# 智能电网安全管理系统

## 项目简介

本项目是一个基于深度学习的智能电网安全管理系统，能够根据历史数据预测电网的过载状态和功率消耗。通过分析电压、电流、功率因数等多种指标，结合天气等环境因素，实现对电网状态的准确预测，从而提前发现潜在风险，保障电网安全稳定运行。

## 数据说明

项目使用了包含以下字段的智能电网数据集：

- 电压 (Voltage)
- 电流 (Current)
- 功率因数 (Power Factor)
- 无功功率 (Reactive Power)
- 电压波动 (Voltage Fluctuation)
- 电价 (Electricity Price)
- 过载状态 (Overload Condition)
- 变压器故障状态 (Transformer Fault)
- 功率消耗 (Power Consumption)
- 温度 (Temperature)
- 湿度 (Humidity)

## 功能特点

1. **数据预处理**：将原始电网数据转换为适用于大语言模型训练的SFT格式
2. **过载预测**：基于历史数据预测当前电网是否处于过载状态
3. **功率预测**：准确预测电网的功率消耗，为能源调度提供依据
4. **训练测试分离**：自动将数据集分为训练集和测试集，便于模型评估

## 安装说明

1. 克隆项目到本地:

```bash
git clone https://github.com/your-username/chat_ev.git
cd chat_ev
```

2. 安装依赖:

```bash
pip install -r requirements.txt
```

## 使用方法

### 数据预处理

执行以下命令进行数据预处理:

```bash
python preprocess_data.py
```

预处理将生成两个文件:
- `sft_data_train.json`: 用于模型训练的数据
- `sft_data_test.json`: 用于模型评估的数据

### 模型训练

```bash
python train_llama_lora.py
```

### 模型评估

```bash
python evaluate_model.py
```

## 项目结构

```
chat_ev/
├── original_data/           # 原始数据集
│   └── smart_grid_dataset.csv
├── preprocess_data.py       # 数据预处理脚本
├── train_model.py           # 模型训练脚本 
├── llama_inference.py       # 模型推理脚本 
├── evaluate_model.py        # 模型评估脚本
├── sft_data_train.json      # 生成的训练数据
├── sft_data_test.json       # 生成的测试数据
└── README.md                # 项目说明文档
```

## 数据格式

示例提示格式:

SYSTEM_PROMPT:
```
You are a smart grid security management expert, skilled in understanding and predicting grid security conditions
```

USER_PROMPT:
```
Please determine the current overload condition and power consumption (kW) based on the following information.\nThe current weather temperature is 19.1913 Celsius, and humidity is 38.3004.\nGiven the following historical charging data time series:\nVoltage (past|current) = [236.9291, 239.9761, 224.4519, 230.5807] | 234.5784 V;\nCurrent (past|current) = [24.5292, 39.2458, 19.6493, 41.844] | 21.2455 A;\nPower Factor (past|current) = [0.913, 0.9672, 0.8311, 0.9675] | 0.8326;\nReactive Power (past|current) = [1.174, 3.4639, 0.8904, 1.0849] | 1.6016 kVAR;\nVoltage Fluctuation (past|current) = [3.8385, 4.0509, 2.8421, -0.286] | -2.6559%;\nElectricity Price (past|current) = [0.2478, 0.4932, 0.1003, 0.4814] | 0.1806;\nOverload history status = [0, 0, 0, 0] (0 means no, 1 means yes);\nTransformer fault status (past|current) = [0, 0, 0, 0] | 0 (0 means normal, 1 means faulty);\nPower consumption in the past 1 hour = [5.8117, 9.4181, 4.4103, 9.6484] kW.\nPlease note!\nYour task is to determine the current overload condition (0 means no, 1 means yes) and predict power consumption (kW) by analyzing the given information and using your common sense.\nIn your answer, just provide your determination and predicted value.\n### Answer:
```

### 微调说明

- 机器: RTX 5880 48G x 4
- 原始模型: meta-llama/Llama-3.2-1B-Instruct
- 微调数据: 40000条样本
- 微调方法: 采用LoRA微调方法做SFT微调

### 评测指标

这是原始模型和微调后模型指标对比：

| 模型 | 电网状态预测准确率 | 功率预测RMSE (kW) | 功率预测MAE (kW) |
|------|----------------|-------------------|------------------|
| 原始模型 | 29.3% | 58.219 | 20.287 |
| 少样本微调 | 71.6% | 4.644 | 3.772 |
| 全样本微调 | 90.05% | 0.102 | 0.052 |

*注：RMSE (均方根误差) 和 MAE (平均绝对误差) 越低表示预测越准确*
