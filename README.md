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
python train_model.py
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
├── evaluate_model.py        # 模型评估脚本
├── sft_data_train.json      # 生成的训练数据
├── sft_data_test.json       # 生成的测试数据
└── README.md                # 项目说明文档
```

## 数据格式

示例提示格式:

```
Please determine the current overload condition and power consumption (kW) based on the following information.
The current weather temperature is 25.5 Celsius, and humidity is 65.2.
Given the following historical charging data time series:
Voltage (past|current) = [220.1, 219.8, 221.2, 220.5] | 221.3 V;
Current (past|current) = [30.2, 32.1, 31.5, 33.0] | 34.2 A;
...
```

## 贡献指南

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request

## 联系方式

如有任何问题或建议，请通过以下方式联系我们:
- 邮箱: your-email@example.com
- GitHub Issues: [https://github.com/your-username/chat_ev/issues](https://github.com/your-username/chat_ev/issues)

## 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件
