#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import torch
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_json_test_data(json_test_path):
    """
    加载JSON格式的测试数据
    """
    print(f"正在读取JSON测试数据: {json_test_path}...")
    with open(json_test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"成功读取JSON测试数据，共 {len(test_data)} 条记录")
    return test_data

def prepare_json_prompts(json_test_data):
    """
    准备用于评估的提示和标签（从JSON数据）
    """
    prompts = []
    labels = []
    
    # 用于从模型回复中解析真实标签的正则表达式
    label_pattern = r"(\d+)(?:\s*,\s*|\s+)(\d+(?:\.\d+)?)"
    
    for item in tqdm(json_test_data):
        messages = item["messages"]
        if len(messages) >= 2:
            # 用户查询作为提示
            user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
            if user_message:
                prompts.append(user_message)
            
            # 如果有助手回复，从中提取标签
            assistant_message = next((msg["content"] for msg in messages if msg["role"] == "assistant"), None)
            if assistant_message:
                matches = re.findall(label_pattern, assistant_message)
                if matches:
                    # 提取第一个匹配的结果
                    overload, power = matches[0]
                    labels.append((int(overload), float(power)))
                else:
                    # 如果没有匹配标准格式，尝试直接查找数字
                    numbers = re.findall(r"\d+(?:\.\d+)?", assistant_message)
                    if len(numbers) >= 2:
                        labels.append((int(float(numbers[0])), float(numbers[1])))
                    else:
                        # 如果仍找不到，添加默认值
                        print(f"警告: 无法从回复中解析标签: {assistant_message}")
                        labels.append((0, 0.0))
    
    return prompts, labels

def generate_predictions(model, tokenizer, prompts, max_new_tokens=50, batch_size=8):
    """
    使用模型生成预测结果
    """
    predictions = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )
            
        for j, output in enumerate(outputs):
            # 解码生成的文本
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # 从生成的文本中截取模型的回答部分
            response = generated_text[len(batch_prompts[j]):]
            predictions.append(response.strip())
    
    return predictions

def parse_predictions(predictions):
    """
    从模型的文本输出中解析过载状态和功率消耗的预测值
    """
    parsed_results = []
    
    for pred in predictions:
        # 使用正则表达式匹配数字
        matches = re.findall(r"(\d+)(?:\s*,\s*|\s+)(\d+(?:\.\d+)?)", pred)
        if matches:
            # 提取第一个匹配的结果
            overload, power = matches[0]
            parsed_results.append((int(overload), float(power)))
        else:
            # 如果没有匹配到格式，尝试直接查找数字
            numbers = re.findall(r"\d+(?:\.\d+)?", pred)
            if len(numbers) >= 2:
                parsed_results.append((int(float(numbers[0])), float(numbers[1])))
            else:
                # 如果仍找不到，添加默认值
                parsed_results.append((0, 0.0))
    
    return parsed_results

def calculate_metrics(true_values, predicted_values):
    """
    计算评估指标：准确率、召回率、F1分数、RMSE、MAE和R平方分数
    """
    # 提取过载状态和功率消耗
    y_true_overload = np.array([x[0] for x in true_values])
    y_pred_overload = np.array([x[0] for x in predicted_values])
    
    y_true_power = np.array([x[1] for x in true_values])
    y_pred_power = np.array([x[1] for x in predicted_values])
    
    # 计算过载状态的准确率
    accuracy = np.mean(y_true_overload == y_pred_overload)
    
    # 计算过载状态的召回率和F1分数
    # 过载状态为1的情况下的召回率（真正例 / (真正例 + 假负例)）
    true_positives = np.sum((y_true_overload == 1) & (y_pred_overload == 1))
    false_negatives = np.sum((y_true_overload == 1) & (y_pred_overload == 0))
    false_positives = np.sum((y_true_overload == 0) & (y_pred_overload == 1))
    
    # 避免除零错误
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0
    
    # 计算精确率 (真正例 / (真正例 + 假正例))
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0
    
    # 计算F1分数
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    # 计算功率消耗的RMSE和MAE
    rmse = np.sqrt(mean_squared_error(y_true_power, y_pred_power))
    mae = mean_absolute_error(y_true_power, y_pred_power)
    
    # 计算R平方分数（决定系数）
    # R² = 1 - (残差平方和 / 总平方和)
    y_true_mean = np.mean(y_true_power)
    ss_total = np.sum((y_true_power - y_true_mean) ** 2)  # 总平方和
    ss_residual = np.sum((y_true_power - y_pred_power) ** 2)  # 残差平方和
    
    # 避免除零错误
    if ss_total > 0:
        r_squared = 1 - (ss_residual / ss_total)
    else:
        r_squared = 0.0
    
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1_score,
        "rmse": rmse,
        "mae": mae,
        "r_squared": r_squared
    }

def evaluate_model(model_name, tokenizer_name, test_prompts, test_labels, is_peft=False, adapter_path=None):
    """
    评估模型性能
    """
    print(f"\n评估模型: {model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.cuda()
    
    # 如果是微调后的模型，加载LoRA适配器
    if is_peft and adapter_path:
        print(f"加载LoRA适配器: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    # 生成预测
    print("生成预测...")
    predictions = generate_predictions(model, tokenizer, test_prompts)
    
    # 解析预测结果
    print("解析预测结果...")
    parsed_predictions = parse_predictions(predictions)
    
    # 计算评估指标
    print("计算评估指标...")
    metrics = calculate_metrics(test_labels, parsed_predictions)
    
    # 清理GPU内存
    del model
    torch.cuda.empty_cache()
    
    return metrics, predictions, parsed_predictions

def plot_evaluation_metrics(original_metrics, checkpoint_50_metrics, checkpoint_100_metrics, checkpoint_200_metrics, checkpoint_300_metrics, checkpoint_400_metrics):
    """
    绘制评估指标的柱状图对比
    """
    print("\n正在绘制评估指标柱状图...")
    
    # 设置中文字体，以正确显示中文
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")
    
    # 分两组绘制图表：过载状态指标和功率消耗指标
    # 1. 过载状态指标（准确率、召回率、精确率、F1分数）
    overload_metrics = ['准确率', '召回率', '精确率', 'F1分数']
    original_overload_values = [
        original_metrics['accuracy'], 
        original_metrics['recall'], 
        original_metrics['precision'], 
        original_metrics['f1_score']
    ]
    checkpoint_50_overload_values = [
        checkpoint_50_metrics['accuracy'], 
        checkpoint_50_metrics['recall'], 
        checkpoint_50_metrics['precision'], 
        checkpoint_50_metrics['f1_score']
    ]
    checkpoint_100_overload_values = [
        checkpoint_100_metrics['accuracy'], 
        checkpoint_100_metrics['recall'], 
        checkpoint_100_metrics['precision'], 
        checkpoint_100_metrics['f1_score']
    ]
    checkpoint_200_overload_values = [
        checkpoint_200_metrics['accuracy'], 
        checkpoint_200_metrics['recall'], 
        checkpoint_200_metrics['precision'], 
        checkpoint_200_metrics['f1_score']
    ]
    checkpoint_300_overload_values = [
        checkpoint_300_metrics['accuracy'], 
        checkpoint_300_metrics['recall'], 
        checkpoint_300_metrics['precision'], 
        checkpoint_300_metrics['f1_score']
    ]
    checkpoint_400_overload_values = [
        checkpoint_400_metrics['accuracy'], 
        checkpoint_400_metrics['recall'], 
        checkpoint_400_metrics['precision'], 
        checkpoint_400_metrics['f1_score']
    ]
    
    # 2. 功率消耗指标（RMSE、MAE和R平方分数）
    power_metrics = ['RMSE', 'MAE', 'R²']
    original_power_values = [original_metrics['rmse'], original_metrics['mae'], original_metrics['r_squared']]
    checkpoint_50_power_values = [checkpoint_50_metrics['rmse'], checkpoint_50_metrics['mae'], checkpoint_50_metrics['r_squared']]
    checkpoint_100_power_values = [checkpoint_100_metrics['rmse'], checkpoint_100_metrics['mae'], checkpoint_100_metrics['r_squared']]
    checkpoint_200_power_values = [checkpoint_200_metrics['rmse'], checkpoint_200_metrics['mae'], checkpoint_200_metrics['r_squared']]
    checkpoint_300_power_values = [checkpoint_300_metrics['rmse'], checkpoint_300_metrics['mae'], checkpoint_300_metrics['r_squared']]
    checkpoint_400_power_values = [checkpoint_400_metrics['rmse'], checkpoint_400_metrics['mae'], checkpoint_400_metrics['r_squared']]
    
    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # 绘制过载状态指标
    x1 = np.arange(len(overload_metrics))  # 标签位置
    width = 0.14  # 柱状图宽度
    
    rects1_1 = ax1.bar(x1 - 2.5*width, original_overload_values, width, label='base', color='skyblue')
    rects1_2 = ax1.bar(x1 - 1.5*width, checkpoint_50_overload_values, width, label='checkpoint-50', color='orange')
    rects1_3 = ax1.bar(x1 - 0.5*width, checkpoint_100_overload_values, width, label='checkpoint-100', color='lightgreen')
    rects1_4 = ax1.bar(x1 + 0.5*width, checkpoint_200_overload_values, width, label='checkpoint-200', color='salmon')
    rects1_5 = ax1.bar(x1 + 1.5*width, checkpoint_300_overload_values, width, label='checkpoint-300', color='mediumpurple')
    rects1_6 = ax1.bar(x1 + 2.5*width, checkpoint_400_overload_values, width, label='checkpoint-400', color='gold')
    
    # 添加标题和坐标轴标签
    ax1.set_title('过载状态预测指标对比', fontsize=16)
    ax1.set_ylabel('指标值', fontsize=14)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(overload_metrics, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 1.0)  # 设置y轴范围为0-1，因为这些指标都是比率
    
    # 绘制功率消耗指标
    x2 = np.arange(len(power_metrics))  # 标签位置
    
    rects2_1 = ax2.bar(x2 - 2.5*width, original_power_values, width, label='base', color='skyblue')
    rects2_2 = ax2.bar(x2 - 1.5*width, checkpoint_50_power_values, width, label='checkpoint-50', color='orange')
    rects2_3 = ax2.bar(x2 - 0.5*width, checkpoint_100_power_values, width, label='checkpoint-100', color='lightgreen')
    rects2_4 = ax2.bar(x2 + 0.5*width, checkpoint_200_power_values, width, label='checkpoint-200', color='salmon')
    rects2_5 = ax2.bar(x2 + 1.5*width, checkpoint_300_power_values, width, label='checkpoint-300', color='mediumpurple')
    rects2_6 = ax2.bar(x2 + 2.5*width, checkpoint_400_power_values, width, label='checkpoint-400', color='gold')
    
    # 添加标题和坐标轴标签
    ax2.set_title('功率消耗预测指标对比', fontsize=16)
    ax2.set_ylabel('指标值', fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(power_metrics, fontsize=12)
    ax2.legend(fontsize=12)
    
    # 在柱状图上添加数值标签
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1_1, ax1)
    autolabel(rects1_2, ax1)
    autolabel(rects1_3, ax1)
    autolabel(rects1_4, ax1)
    autolabel(rects1_5, ax1)
    autolabel(rects1_6, ax1)
    autolabel(rects2_1, ax2)
    autolabel(rects2_2, ax2)
    autolabel(rects2_3, ax2)
    autolabel(rects2_4, ax2)
    autolabel(rects2_5, ax2)
    autolabel(rects2_6, ax2)
    
    # 调整布局
    fig.tight_layout()
    
    # 保存图表
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("评估指标柱状图已保存为 model_comparison.png")
    
    # 绘制训练步数与性能指标的关系曲线图
    plt.figure(figsize=(20, 15))
    
    # 设置训练步数
    steps = [0, 50, 100, 200, 300, 400]  # 0代表原始模型
    
    # 1. 过载状态准确率曲线
    plt.subplot(3, 2, 1)
    accuracy_values = [
        original_metrics['accuracy'],
        checkpoint_50_metrics['accuracy'],
        checkpoint_100_metrics['accuracy'],
        checkpoint_200_metrics['accuracy'],
        checkpoint_300_metrics['accuracy'],
        checkpoint_400_metrics['accuracy']
    ]
    plt.plot(steps, accuracy_values, 'o-', linewidth=2, markersize=8)
    plt.title('训练步数与过载状态准确率关系', fontsize=14)
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.grid(True)
    
    # 2. 过载状态F1分数曲线
    plt.subplot(3, 2, 2)
    f1_values = [
        original_metrics['f1_score'],
        checkpoint_50_metrics['f1_score'],
        checkpoint_100_metrics['f1_score'],
        checkpoint_200_metrics['f1_score'],
        checkpoint_300_metrics['f1_score'],
        checkpoint_400_metrics['f1_score']
    ]
    plt.plot(steps, f1_values, 'o-', linewidth=2, markersize=8, color='orange')
    plt.title('训练步数与过载状态F1分数关系', fontsize=14)
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('F1分数', fontsize=12)
    plt.grid(True)
    
    # 3. 功率消耗RMSE曲线
    plt.subplot(3, 2, 3)
    rmse_values = [
        original_metrics['rmse'],
        checkpoint_50_metrics['rmse'],
        checkpoint_100_metrics['rmse'],
        checkpoint_200_metrics['rmse'],
        checkpoint_300_metrics['rmse'],
        checkpoint_400_metrics['rmse']
    ]
    plt.plot(steps, rmse_values, 'o-', linewidth=2, markersize=8, color='green')
    plt.title('训练步数与功率消耗RMSE关系', fontsize=14)
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True)
    
    # 4. 功率消耗MAE曲线
    plt.subplot(3, 2, 4)
    mae_values = [
        original_metrics['mae'],
        checkpoint_50_metrics['mae'],
        checkpoint_100_metrics['mae'],
        checkpoint_200_metrics['mae'],
        checkpoint_300_metrics['mae'],
        checkpoint_400_metrics['mae']
    ]
    plt.plot(steps, mae_values, 'o-', linewidth=2, markersize=8, color='red')
    plt.title('训练步数与功率消耗MAE关系', fontsize=14)
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.grid(True)
    
    # 5. 功率消耗R平方分数曲线
    plt.subplot(3, 2, 5)
    r_squared_values = [
        original_metrics['r_squared'],
        checkpoint_50_metrics['r_squared'],
        checkpoint_100_metrics['r_squared'],
        checkpoint_200_metrics['r_squared'],
        checkpoint_300_metrics['r_squared'],
        checkpoint_400_metrics['r_squared']
    ]
    plt.plot(steps, r_squared_values, 'o-', linewidth=2, markersize=8, color='purple')
    plt.title('训练步数与功率消耗R平方分数关系', fontsize=14)
    plt.xlabel('训练步数', fontsize=12)
    plt.ylabel('R平方分数', fontsize=12)
    plt.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("训练进度曲线图已保存为 training_progress.png")

def main():
    # 配置参数
    original_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 与训练脚本中使用的模型一致
    checkpoint_50_adapter_path = "checkpoints/checkpoint-50"  # 50步检查点模型的保存路径
    checkpoint_100_adapter_path = "checkpoints/checkpoint-100"  # 100步检查点模型的保存路径
    checkpoint_200_adapter_path = "checkpoints/checkpoint-200"  # 200步检查点模型的保存路径
    checkpoint_300_adapter_path = "checkpoints/checkpoint-300"  # 300步检查点模型的保存路径
    checkpoint_400_adapter_path = "checkpoints/checkpoint-400"  # 400步检查点模型的保存路径
    json_test_data_path = "sft_data_test.json"  # JSON测试数据路径
    
    # 打印评估配置
    print("=== 电网安全预测模型评估 ===")
    print(f"原始模型: {original_model_name}")
    print(f"检查点-50步适配器: {checkpoint_50_adapter_path}")
    print(f"检查点-100步适配器: {checkpoint_100_adapter_path}")
    print(f"检查点-200步适配器: {checkpoint_200_adapter_path}")
    print(f"检查点-300步适配器: {checkpoint_300_adapter_path}")
    print(f"检查点-400步适配器: {checkpoint_400_adapter_path}")
    

    test_data = load_json_test_data(json_test_data_path)
    test_prompts, test_labels = prepare_json_prompts(test_data)
    
    print(f"生成 {len(test_prompts)} 个测试样本")
    
    # 评估原始模型
    original_metrics, original_predictions, original_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels
    )
    
    # 评估50步检查点模型
    checkpoint_50_metrics, checkpoint_50_predictions, checkpoint_50_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_50_adapter_path
    )
    
    # 评估100步检查点模型
    checkpoint_100_metrics, checkpoint_100_predictions, checkpoint_100_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_100_adapter_path
    )
    
    # 评估200步检查点模型
    checkpoint_200_metrics, checkpoint_200_predictions, checkpoint_200_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_200_adapter_path
    )
    
    # 评估300步检查点模型
    checkpoint_300_metrics, checkpoint_300_predictions, checkpoint_300_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_300_adapter_path
    )
    
    # 评估400步检查点模型
    checkpoint_400_metrics, checkpoint_400_predictions, checkpoint_400_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_400_adapter_path
    )
    
    # 打印评估结果
    print("\n=== 评估结果 ===")
    print("原始模型:")
    print(f"  过载状态准确率: {original_metrics['accuracy']:.4f}")
    print(f"  过载状态召回率: {original_metrics['recall']:.4f}")
    print(f"  过载状态精确率: {original_metrics['precision']:.4f}")
    print(f"  过载状态F1分数: {original_metrics['f1_score']:.4f}")
    print(f"  功率消耗RMSE: {original_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {original_metrics['mae']:.4f}")
    print(f"  功率消耗R平方分数: {original_metrics['r_squared']:.4f}")
    
    print("\n检查点-50步模型:")
    print(f"  过载状态准确率: {checkpoint_50_metrics['accuracy']:.4f}")
    print(f"  过载状态召回率: {checkpoint_50_metrics['recall']:.4f}")
    print(f"  过载状态精确率: {checkpoint_50_metrics['precision']:.4f}")
    print(f"  过载状态F1分数: {checkpoint_50_metrics['f1_score']:.4f}")
    print(f"  功率消耗RMSE: {checkpoint_50_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {checkpoint_50_metrics['mae']:.4f}")
    print(f"  功率消耗R平方分数: {checkpoint_50_metrics['r_squared']:.4f}")
    
    print("\n检查点-100步模型:")
    print(f"  过载状态准确率: {checkpoint_100_metrics['accuracy']:.4f}")
    print(f"  过载状态召回率: {checkpoint_100_metrics['recall']:.4f}")
    print(f"  过载状态精确率: {checkpoint_100_metrics['precision']:.4f}")
    print(f"  过载状态F1分数: {checkpoint_100_metrics['f1_score']:.4f}")
    print(f"  功率消耗RMSE: {checkpoint_100_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {checkpoint_100_metrics['mae']:.4f}")
    print(f"  功率消耗R平方分数: {checkpoint_100_metrics['r_squared']:.4f}")
    
    print("\n检查点-200步模型:")
    print(f"  过载状态准确率: {checkpoint_200_metrics['accuracy']:.4f}")
    print(f"  过载状态召回率: {checkpoint_200_metrics['recall']:.4f}")
    print(f"  过载状态精确率: {checkpoint_200_metrics['precision']:.4f}")
    print(f"  过载状态F1分数: {checkpoint_200_metrics['f1_score']:.4f}")
    print(f"  功率消耗RMSE: {checkpoint_200_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {checkpoint_200_metrics['mae']:.4f}")
    print(f"  功率消耗R平方分数: {checkpoint_200_metrics['r_squared']:.4f}")
    
    print("\n检查点-300步模型:")
    print(f"  过载状态准确率: {checkpoint_300_metrics['accuracy']:.4f}")
    print(f"  过载状态召回率: {checkpoint_300_metrics['recall']:.4f}")
    print(f"  过载状态精确率: {checkpoint_300_metrics['precision']:.4f}")
    print(f"  过载状态F1分数: {checkpoint_300_metrics['f1_score']:.4f}")
    print(f"  功率消耗RMSE: {checkpoint_300_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {checkpoint_300_metrics['mae']:.4f}")
    print(f"  功率消耗R平方分数: {checkpoint_300_metrics['r_squared']:.4f}")
    
    print("\n检查点-400步模型:")
    print(f"  过载状态准确率: {checkpoint_400_metrics['accuracy']:.4f}")
    print(f"  过载状态召回率: {checkpoint_400_metrics['recall']:.4f}")
    print(f"  过载状态精确率: {checkpoint_400_metrics['precision']:.4f}")
    print(f"  过载状态F1分数: {checkpoint_400_metrics['f1_score']:.4f}")
    print(f"  功率消耗RMSE: {checkpoint_400_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {checkpoint_400_metrics['mae']:.4f}")
    print(f"  功率消耗R平方分数: {checkpoint_400_metrics['r_squared']:.4f}")
    
    # 计算各个检查点模型相对于原始模型的性能提升百分比
    # 50步检查点
    checkpoint_50_accuracy_improvement = ((checkpoint_50_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_50_recall_improvement = ((checkpoint_50_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_50_precision_improvement = ((checkpoint_50_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_50_f1_improvement = ((checkpoint_50_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_50_rmse_improvement = ((original_metrics['rmse'] - checkpoint_50_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_50_mae_improvement = ((original_metrics['mae'] - checkpoint_50_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_50_r_squared_improvement = ((checkpoint_50_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # 100步检查点
    checkpoint_100_accuracy_improvement = ((checkpoint_100_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_100_recall_improvement = ((checkpoint_100_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_100_precision_improvement = ((checkpoint_100_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_100_f1_improvement = ((checkpoint_100_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_100_rmse_improvement = ((original_metrics['rmse'] - checkpoint_100_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_100_mae_improvement = ((original_metrics['mae'] - checkpoint_100_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_100_r_squared_improvement = ((checkpoint_100_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # 200步检查点
    checkpoint_200_accuracy_improvement = ((checkpoint_200_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_200_recall_improvement = ((checkpoint_200_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_200_precision_improvement = ((checkpoint_200_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_200_f1_improvement = ((checkpoint_200_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_200_rmse_improvement = ((original_metrics['rmse'] - checkpoint_200_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_200_mae_improvement = ((original_metrics['mae'] - checkpoint_200_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_200_r_squared_improvement = ((checkpoint_200_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # 300步检查点
    checkpoint_300_accuracy_improvement = ((checkpoint_300_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_300_recall_improvement = ((checkpoint_300_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_300_precision_improvement = ((checkpoint_300_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_300_f1_improvement = ((checkpoint_300_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_300_rmse_improvement = ((original_metrics['rmse'] - checkpoint_300_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_300_mae_improvement = ((original_metrics['mae'] - checkpoint_300_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_300_r_squared_improvement = ((checkpoint_300_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # 400步检查点
    checkpoint_400_accuracy_improvement = ((checkpoint_400_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_400_recall_improvement = ((checkpoint_400_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_400_precision_improvement = ((checkpoint_400_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_400_f1_improvement = ((checkpoint_400_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_400_rmse_improvement = ((original_metrics['rmse'] - checkpoint_400_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_400_mae_improvement = ((original_metrics['mae'] - checkpoint_400_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_400_r_squared_improvement = ((checkpoint_400_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    print("\n各检查点模型性能提升:")
    print("检查点-50步模型:")
    print(f"  过载状态准确率提升: {checkpoint_50_accuracy_improvement:.2f}%")
    print(f"  过载状态召回率提升: {checkpoint_50_recall_improvement:.2f}%")
    print(f"  过载状态精确率提升: {checkpoint_50_precision_improvement:.2f}%")
    print(f"  过载状态F1分数提升: {checkpoint_50_f1_improvement:.2f}%")
    print(f"  功率消耗RMSE降低: {checkpoint_50_rmse_improvement:.2f}%")
    print(f"  功率消耗MAE降低: {checkpoint_50_mae_improvement:.2f}%")
    print(f"  功率消耗R平方分数提升: {checkpoint_50_r_squared_improvement:.2f}%")
    
    print("\n检查点-100步模型:")
    print(f"  过载状态准确率提升: {checkpoint_100_accuracy_improvement:.2f}%")
    print(f"  过载状态召回率提升: {checkpoint_100_recall_improvement:.2f}%")
    print(f"  过载状态精确率提升: {checkpoint_100_precision_improvement:.2f}%")
    print(f"  过载状态F1分数提升: {checkpoint_100_f1_improvement:.2f}%")
    print(f"  功率消耗RMSE降低: {checkpoint_100_rmse_improvement:.2f}%")
    print(f"  功率消耗MAE降低: {checkpoint_100_mae_improvement:.2f}%")
    print(f"  功率消耗R平方分数提升: {checkpoint_100_r_squared_improvement:.2f}%")
    
    print("\n检查点-200步模型:")
    print(f"  过载状态准确率提升: {checkpoint_200_accuracy_improvement:.2f}%")
    print(f"  过载状态召回率提升: {checkpoint_200_recall_improvement:.2f}%")
    print(f"  过载状态精确率提升: {checkpoint_200_precision_improvement:.2f}%")
    print(f"  过载状态F1分数提升: {checkpoint_200_f1_improvement:.2f}%")
    print(f"  功率消耗RMSE降低: {checkpoint_200_rmse_improvement:.2f}%")
    print(f"  功率消耗MAE降低: {checkpoint_200_mae_improvement:.2f}%")
    print(f"  功率消耗R平方分数提升: {checkpoint_200_r_squared_improvement:.2f}%")
    
    print("\n检查点-300步模型:")
    print(f"  过载状态准确率提升: {checkpoint_300_accuracy_improvement:.2f}%")
    print(f"  过载状态召回率提升: {checkpoint_300_recall_improvement:.2f}%")
    print(f"  过载状态精确率提升: {checkpoint_300_precision_improvement:.2f}%")
    print(f"  过载状态F1分数提升: {checkpoint_300_f1_improvement:.2f}%")
    print(f"  功率消耗RMSE降低: {checkpoint_300_rmse_improvement:.2f}%")
    print(f"  功率消耗MAE降低: {checkpoint_300_mae_improvement:.2f}%")
    print(f"  功率消耗R平方分数提升: {checkpoint_300_r_squared_improvement:.2f}%")
    
    print("\n检查点-400步模型:")
    print(f"  过载状态准确率提升: {checkpoint_400_accuracy_improvement:.2f}%")
    print(f"  过载状态召回率提升: {checkpoint_400_recall_improvement:.2f}%")
    print(f"  过载状态精确率提升: {checkpoint_400_precision_improvement:.2f}%")
    print(f"  过载状态F1分数提升: {checkpoint_400_f1_improvement:.2f}%")
    print(f"  功率消耗RMSE降低: {checkpoint_400_rmse_improvement:.2f}%")
    print(f"  功率消耗MAE降低: {checkpoint_400_mae_improvement:.2f}%")
    print(f"  功率消耗R平方分数提升: {checkpoint_400_r_squared_improvement:.2f}%")
    
    # 保存评估结果
    results = {
        "original_model": {
            "metrics": original_metrics,
            "predictions": original_predictions,
            "parsed_predictions": original_parsed
        },
        "checkpoint_50": {
            "metrics": checkpoint_50_metrics,
            "predictions": checkpoint_50_predictions,
            "parsed_predictions": checkpoint_50_parsed
        },
        "checkpoint_100": {
            "metrics": checkpoint_100_metrics,
            "predictions": checkpoint_100_predictions,
            "parsed_predictions": checkpoint_100_parsed
        },
        "checkpoint_200": {
            "metrics": checkpoint_200_metrics,
            "predictions": checkpoint_200_predictions,
            "parsed_predictions": checkpoint_200_parsed
        },
        "checkpoint_300": {
            "metrics": checkpoint_300_metrics,
            "predictions": checkpoint_300_predictions,
            "parsed_predictions": checkpoint_300_parsed
        },
        "checkpoint_400": {
            "metrics": checkpoint_400_metrics,
            "predictions": checkpoint_400_predictions,
            "parsed_predictions": checkpoint_400_parsed
        },
        "improvements": {
            "checkpoint_50": {
                "accuracy": checkpoint_50_accuracy_improvement,
                "recall": checkpoint_50_recall_improvement,
                "precision": checkpoint_50_precision_improvement,
                "f1_score": checkpoint_50_f1_improvement,
                "rmse": checkpoint_50_rmse_improvement,
                "mae": checkpoint_50_mae_improvement,
                "r_squared": checkpoint_50_r_squared_improvement
            },
            "checkpoint_100": {
                "accuracy": checkpoint_100_accuracy_improvement,
                "recall": checkpoint_100_recall_improvement,
                "precision": checkpoint_100_precision_improvement,
                "f1_score": checkpoint_100_f1_improvement,
                "rmse": checkpoint_100_rmse_improvement,
                "mae": checkpoint_100_mae_improvement,
                "r_squared": checkpoint_100_r_squared_improvement
            },
            "checkpoint_200": {
                "accuracy": checkpoint_200_accuracy_improvement,
                "recall": checkpoint_200_recall_improvement,
                "precision": checkpoint_200_precision_improvement,
                "f1_score": checkpoint_200_f1_improvement,
                "rmse": checkpoint_200_rmse_improvement,
                "mae": checkpoint_200_mae_improvement,
                "r_squared": checkpoint_200_r_squared_improvement
            },
            "checkpoint_300": {
                "accuracy": checkpoint_300_accuracy_improvement,
                "recall": checkpoint_300_recall_improvement,
                "precision": checkpoint_300_precision_improvement,
                "f1_score": checkpoint_300_f1_improvement,
                "rmse": checkpoint_300_rmse_improvement,
                "mae": checkpoint_300_mae_improvement,
                "r_squared": checkpoint_300_r_squared_improvement
            },
            "checkpoint_400": {
                "accuracy": checkpoint_400_accuracy_improvement,
                "recall": checkpoint_400_recall_improvement,
                "precision": checkpoint_400_precision_improvement,
                "f1_score": checkpoint_400_f1_improvement,
                "rmse": checkpoint_400_rmse_improvement,
                "mae": checkpoint_400_mae_improvement,
                "r_squared": checkpoint_400_r_squared_improvement
            }
        }
    }
    
    with open("evaluation_results.json", "w") as f:
        # 转换不可序列化的numpy值为Python原生类型
        json_results = {
            "original_model": {
                "metrics": {
                    "accuracy": float(original_metrics["accuracy"]),
                    "recall": float(original_metrics["recall"]),
                    "precision": float(original_metrics["precision"]),
                    "f1_score": float(original_metrics["f1_score"]),
                    "rmse": float(original_metrics["rmse"]),
                    "mae": float(original_metrics["mae"]),
                    "r_squared": float(original_metrics["r_squared"])
                },
                "predictions": original_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in original_parsed]
            },
            "checkpoint_50": {
                "metrics": {
                    "accuracy": float(checkpoint_50_metrics["accuracy"]),
                    "recall": float(checkpoint_50_metrics["recall"]),
                    "precision": float(checkpoint_50_metrics["precision"]),
                    "f1_score": float(checkpoint_50_metrics["f1_score"]),
                    "rmse": float(checkpoint_50_metrics["rmse"]),
                    "mae": float(checkpoint_50_metrics["mae"]),
                    "r_squared": float(checkpoint_50_metrics["r_squared"])
                },
                "predictions": checkpoint_50_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in checkpoint_50_parsed]
            },
            "checkpoint_100": {
                "metrics": {
                    "accuracy": float(checkpoint_100_metrics["accuracy"]),
                    "recall": float(checkpoint_100_metrics["recall"]),
                    "precision": float(checkpoint_100_metrics["precision"]),
                    "f1_score": float(checkpoint_100_metrics["f1_score"]),
                    "rmse": float(checkpoint_100_metrics["rmse"]),
                    "mae": float(checkpoint_100_metrics["mae"]),
                    "r_squared": float(checkpoint_100_metrics["r_squared"])
                },
                "predictions": checkpoint_100_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in checkpoint_100_parsed]
            },
            "checkpoint_200": {
                "metrics": {
                    "accuracy": float(checkpoint_200_metrics["accuracy"]),
                    "recall": float(checkpoint_200_metrics["recall"]),
                    "precision": float(checkpoint_200_metrics["precision"]),
                    "f1_score": float(checkpoint_200_metrics["f1_score"]),
                    "rmse": float(checkpoint_200_metrics["rmse"]),
                    "mae": float(checkpoint_200_metrics["mae"]),
                    "r_squared": float(checkpoint_200_metrics["r_squared"])
                },
                "predictions": checkpoint_200_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in checkpoint_200_parsed]
            },
            "checkpoint_300": {
                "metrics": {
                    "accuracy": float(checkpoint_300_metrics["accuracy"]),
                    "recall": float(checkpoint_300_metrics["recall"]),
                    "precision": float(checkpoint_300_metrics["precision"]),
                    "f1_score": float(checkpoint_300_metrics["f1_score"]),
                    "rmse": float(checkpoint_300_metrics["rmse"]),
                    "mae": float(checkpoint_300_metrics["mae"]),
                    "r_squared": float(checkpoint_300_metrics["r_squared"])
                },
                "predictions": checkpoint_300_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in checkpoint_300_parsed]
            },
            "checkpoint_400": {
                "metrics": {
                    "accuracy": float(checkpoint_400_metrics["accuracy"]),
                    "recall": float(checkpoint_400_metrics["recall"]),
                    "precision": float(checkpoint_400_metrics["precision"]),
                    "f1_score": float(checkpoint_400_metrics["f1_score"]),
                    "rmse": float(checkpoint_400_metrics["rmse"]),
                    "mae": float(checkpoint_400_metrics["mae"]),
                    "r_squared": float(checkpoint_400_metrics["r_squared"])
                },
                "predictions": checkpoint_400_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in checkpoint_400_parsed]
            }
        }
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print("\n评估结果已保存到 evaluation_results.json")
    
    # 绘制评估指标柱状图
    plot_evaluation_metrics(original_metrics, checkpoint_50_metrics, checkpoint_100_metrics, checkpoint_200_metrics, checkpoint_300_metrics, checkpoint_400_metrics)

if __name__ == "__main__":
    main()