import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# 读取原始数据
print("正在读取原始数据...")
df = pd.read_csv("original_data/smart_grid_dataset.csv")
print(f"成功读取数据，共 {len(df)} 条记录")

# 创建提示模板
system_prompt = "You are a smart grid security management expert, skilled in understanding and predicting grid security conditions"

prompt_template = """
Please determine the current overload condition and power consumption (kW) based on the following information.
The current weather temperature is {temperature} Celsius, and humidity is {humidity}.
Given the following historical charging data time series:
Voltage (past|current) = {voltage_past} | {voltage_current} V;
Current (past|current) = {current_past} | {current_current} A;
Power Factor (past|current) = {power_factor_past} | {power_factor_current};
Reactive Power (past|current) = {reactive_power_past} | {reactive_power_current} kVAR;
Voltage Fluctuation (past|current) = {voltage_fluctuation_past} | {voltage_fluctuation_current}%;
Electricity Price (past|current) = {electricity_price_past} | {electricity_price_current};
Overload history status = {overload_history} (0 means no, 1 means yes);
Transformer fault status (past|current) = {transformer_fault_past} | {transformer_fault_current} (0 means normal, 1 means faulty);
Power consumption in the past 1 hour = {power_consumption_past} kW.
Please note!
Your task is to determine the current overload condition (0 means no, 1 means yes) and predict power consumption (kW) by analyzing the given information and using your common sense.
In your answer, just provide your determination and predicted value.
### Answer:
"""

def prepare_sft_data(df, lookback=4):
    """准备SFT格式的数据"""
    
    all_samples = []
    
    # 从lookback+1开始，以便有足够的历史数据
    for idx in tqdm(range(lookback+1, len(df))):
        current_row = df.iloc[idx]
        history_rows = df.iloc[idx-lookback:idx]
        
        # 提取历史数据
        voltage_past = [round(v, 4) for v in history_rows['Voltage (V)'].values]
        voltage_current = round(current_row['Voltage (V)'], 4)
        
        current_past = [round(c, 4) for c in history_rows['Current (A)'].values]
        current_current = round(current_row['Current (A)'], 4)
        
        power_factor_past = [round(pf, 4) for pf in history_rows['Power Factor'].values]
        power_factor_current = round(current_row['Power Factor'], 4)
        
        reactive_power_past = [round(rp, 4) for rp in history_rows['Reactive Power (kVAR)'].values]
        reactive_power_current = round(current_row['Reactive Power (kVAR)'], 4)
        
        voltage_fluctuation_past = [round(vf, 4) for vf in history_rows['Voltage Fluctuation (%)'].values]
        voltage_fluctuation_current = round(current_row['Voltage Fluctuation (%)'], 4)
        
        electricity_price_past = [round(ep, 4) for ep in history_rows['Electricity Price (USD/kWh)'].values]
        electricity_price_current = round(current_row['Electricity Price (USD/kWh)'], 4)
        
        overload_history = history_rows['Overload Condition'].tolist()
        
        transformer_fault_past = history_rows['Transformer Fault'].tolist()
        transformer_fault_current = int(current_row['Transformer Fault'])
        
        power_consumption_past = [round(pc, 4) for pc in history_rows['Power Consumption (kW)'].values]
        
        temperature = round(current_row['Temperature (°C)'], 4)
        humidity = round(current_row['Humidity (%)'], 4)
        
        # 获取当前实际的过载状态和功率消耗（作为标签）
        overload_condition = int(current_row['Overload Condition'])
        power_consumption = round(current_row['Power Consumption (kW)'], 4)
        
        # 格式化提示
        user_prompt = prompt_template.format(
            temperature=temperature,
            humidity=humidity,
            voltage_past=voltage_past,
            voltage_current=voltage_current,
            current_past=current_past,
            current_current=current_current,
            power_factor_past=power_factor_past,
            power_factor_current=power_factor_current,
            reactive_power_past=reactive_power_past,
            reactive_power_current=reactive_power_current,
            voltage_fluctuation_past=voltage_fluctuation_past,
            voltage_fluctuation_current=voltage_fluctuation_current,
            electricity_price_past=electricity_price_past,
            electricity_price_current=electricity_price_current,
            overload_history=overload_history,
            transformer_fault_past=transformer_fault_past,
            transformer_fault_current=transformer_fault_current,
            power_consumption_past=power_consumption_past
        )
        
        # 助手回答（目标输出）
        assistant_response = f"{overload_condition}, {power_consumption}"
        
        # 创建SFT样本
        sample = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt.strip()
                },
                {
                    "role": "user",
                    "content": user_prompt.strip()
                },
                {
                    "role": "assistant",
                    "content": assistant_response.strip()
                }
            ]
        }
        
        all_samples.append(sample)
    
    return all_samples

# 生成训练数据
print("正在生成SFT训练数据...")
sft_data = prepare_sft_data(df)
print(f"成功生成 {len(sft_data)} 条SFT训练样本")

# 保存为JSON文件
with open("sft_data_train.json", "w") as f:
    json.dump(sft_data[:49000], f, ensure_ascii=False,indent=2)

with open("sft_data_test.json", "w") as f:
    json.dump(sft_data[49000:], f, ensure_ascii=False,indent=2)

print("SFT训练数据已保存到 sft_data_train.json")
print("SFT测试数据已保存到 sft_data_test.json")

# 打印示例
print("\n示例SFT数据:")
print("用户提示:", sft_data[0]["messages"][1]["content"])
print("\n助手回答:", sft_data[0]["messages"][2]["content"])