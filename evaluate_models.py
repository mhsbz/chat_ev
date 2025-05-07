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

def plot_evaluation_metrics(original_metrics, few_shot_metrics, finetuned_metrics):
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
    few_shot_overload_values = [
        few_shot_metrics['accuracy'], 
        few_shot_metrics['recall'], 
        few_shot_metrics['precision'], 
        few_shot_metrics['f1_score']
    ]
    finetuned_overload_values = [
        finetuned_metrics['accuracy'], 
        finetuned_metrics['recall'], 
        finetuned_metrics['precision'], 
        finetuned_metrics['f1_score']
    ]
    
    # 2. 功率消耗指标（RMSE、MAE和R平方分数）
    power_metrics = ['RMSE', 'MAE', 'R²']
    original_power_values = [original_metrics['rmse'], original_metrics['mae'], original_metrics['r_squared']]
    few_shot_power_values = [few_shot_metrics['rmse'], few_shot_metrics['mae'], few_shot_metrics['r_squared']]
    finetuned_power_values = [finetuned_metrics['rmse'], finetuned_metrics['mae'], finetuned_metrics['r_squared']]
    
    # 创建一个包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 绘制过载状态指标
    x1 = np.arange(len(overload_metrics))  # 标签位置
    width = 0.25  # 柱状图宽度
    
    rects1_1 = ax1.bar(x1 - width, original_overload_values, width, label='base', color='skyblue')
    rects1_2 = ax1.bar(x1, few_shot_overload_values, width, label='few-shot (checkpoint-50)', color='orange')
    rects1_3 = ax1.bar(x1 + width, finetuned_overload_values, width, label='finetuned (checkpoint-625)', color='lightgreen')
    
    # 添加标题和坐标轴标签
    ax1.set_title('过载状态预测指标对比', fontsize=16)
    ax1.set_ylabel('指标值', fontsize=14)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(overload_metrics, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 1.0)  # 设置y轴范围为0-1，因为这些指标都是比率
    
    # 绘制功率消耗指标
    x2 = np.arange(len(power_metrics))  # 标签位置
    
    rects2_1 = ax2.bar(x2 - width, original_power_values, width, label='base', color='skyblue')
    rects2_2 = ax2.bar(x2, few_shot_power_values, width, label='few-shot (checkpoint-50)', color='orange')
    rects2_3 = ax2.bar(x2 + width, finetuned_power_values, width, label='finetuned (checkpoint-625)', color='lightgreen')
    
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
                        ha='center', va='bottom')
    
    autolabel(rects1_1, ax1)
    autolabel(rects1_2, ax1)
    autolabel(rects1_3, ax1)
    autolabel(rects2_1, ax2)
    autolabel(rects2_2, ax2)
    autolabel(rects2_3, ax2)
    
    # 调整布局
    fig.tight_layout()
    
    # 保存图表
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("评估指标柱状图已保存为 model_comparison.png")

def main():
    # 配置参数
    original_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 与训练脚本中使用的模型一致
    few_shot_model_adapter_path = "checkpoints/checkpoint-50"  # 少样本微调模型的保存路径
    finetuned_model_adapter_path = "checkpoints/checkpoint-300"  # 微调模型的保存路径
    json_test_data_path = "sft_data_test.json"  # JSON测试数据路径
    
    # 打印评估配置
    print("=== 电网安全预测模型评估 ===")
    print(f"原始模型: {original_model_name}")
    print(f"少样本微调模型适配器: {few_shot_model_adapter_path}")
    print(f"完全微调模型适配器: {finetuned_model_adapter_path}")
    

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
    
    # 评估少样本微调模型 (checkpoint-50)
    few_shot_metrics, few_shot_predictions, few_shot_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=few_shot_model_adapter_path
    )
    
    # 评估完全微调后的模型 (checkpoint-625)
    finetuned_metrics, finetuned_predictions, finetuned_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=finetuned_model_adapter_path
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
    
    print("\n少样本微调模型 (checkpoint-50):")
    print(f"  过载状态准确率: {few_shot_metrics['accuracy']:.4f}")
    print(f"  过载状态召回率: {few_shot_metrics['recall']:.4f}")
    print(f"  过载状态精确率: {few_shot_metrics['precision']:.4f}")
    print(f"  过载状态F1分数: {few_shot_metrics['f1_score']:.4f}")
    print(f"  功率消耗RMSE: {few_shot_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {few_shot_metrics['mae']:.4f}")
    print(f"  功率消耗R平方分数: {few_shot_metrics['r_squared']:.4f}")
    
    print("\n完全微调模型 (checkpoint-625):")
    print(f"  过载状态准确率: {finetuned_metrics['accuracy']:.4f}")
    print(f"  过载状态召回率: {finetuned_metrics['recall']:.4f}")
    print(f"  过载状态精确率: {finetuned_metrics['precision']:.4f}")
    print(f"  过载状态F1分数: {finetuned_metrics['f1_score']:.4f}")
    print(f"  功率消耗RMSE: {finetuned_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {finetuned_metrics['mae']:.4f}")
    print(f"  功率消耗R平方分数: {finetuned_metrics['r_squared']:.4f}")
    
    # 少样本模型相对于原始模型的性能提升百分比
    few_shot_accuracy_improvement = ((few_shot_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    few_shot_recall_improvement = ((few_shot_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    few_shot_precision_improvement = ((few_shot_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    few_shot_f1_improvement = ((few_shot_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    few_shot_rmse_improvement = ((original_metrics['rmse'] - few_shot_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    few_shot_mae_improvement = ((original_metrics['mae'] - few_shot_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    few_shot_r_squared_improvement = ((few_shot_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # 完全微调模型相对于原始模型的性能提升百分比
    finetuned_accuracy_improvement = ((finetuned_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    finetuned_recall_improvement = ((finetuned_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    finetuned_precision_improvement = ((finetuned_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    finetuned_f1_improvement = ((finetuned_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    finetuned_rmse_improvement = ((original_metrics['rmse'] - finetuned_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    finetuned_mae_improvement = ((original_metrics['mae'] - finetuned_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    finetuned_r_squared_improvement = ((finetuned_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    print("\n少样本微调模型性能提升:")
    print(f"  过载状态准确率提升: {few_shot_accuracy_improvement:.2f}%")
    print(f"  过载状态召回率提升: {few_shot_recall_improvement:.2f}%")
    print(f"  过载状态精确率提升: {few_shot_precision_improvement:.2f}%")
    print(f"  过载状态F1分数提升: {few_shot_f1_improvement:.2f}%")
    print(f"  功率消耗RMSE降低: {few_shot_rmse_improvement:.2f}%")
    print(f"  功率消耗MAE降低: {few_shot_mae_improvement:.2f}%")
    print(f"  功率消耗R平方分数提升: {few_shot_r_squared_improvement:.2f}%")
    
    print("\n完全微调模型性能提升:")
    print(f"  过载状态准确率提升: {finetuned_accuracy_improvement:.2f}%")
    print(f"  过载状态召回率提升: {finetuned_recall_improvement:.2f}%")
    print(f"  过载状态精确率提升: {finetuned_precision_improvement:.2f}%")
    print(f"  过载状态F1分数提升: {finetuned_f1_improvement:.2f}%")
    print(f"  功率消耗RMSE降低: {finetuned_rmse_improvement:.2f}%")
    print(f"  功率消耗MAE降低: {finetuned_mae_improvement:.2f}%")
    print(f"  功率消耗R平方分数提升: {finetuned_r_squared_improvement:.2f}%")
    
    # 保存评估结果
    results = {
        "original_model": {
            "metrics": original_metrics,
            "predictions": original_predictions,
            "parsed_predictions": original_parsed
        },
        "few_shot_model": {
            "metrics": few_shot_metrics,
            "predictions": few_shot_predictions,
            "parsed_predictions": few_shot_parsed
        },
        "finetuned_model": {
            "metrics": finetuned_metrics,
            "predictions": finetuned_predictions,
            "parsed_predictions": finetuned_parsed
        },
        "improvements": {
            "few_shot": {
                "accuracy": few_shot_accuracy_improvement,
                "recall": few_shot_recall_improvement,
                "precision": few_shot_precision_improvement,
                "f1_score": few_shot_f1_improvement,
                "rmse": few_shot_rmse_improvement,
                "mae": few_shot_mae_improvement,
                "r_squared": few_shot_r_squared_improvement
            },
            "finetuned": {
                "accuracy": finetuned_accuracy_improvement,
                "recall": finetuned_recall_improvement,
                "precision": finetuned_precision_improvement,
                "f1_score": finetuned_f1_improvement,
                "rmse": finetuned_rmse_improvement,
                "mae": finetuned_mae_improvement,
                "r_squared": finetuned_r_squared_improvement
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
            "few_shot_model": {
                "metrics": {
                    "accuracy": float(few_shot_metrics["accuracy"]),
                    "recall": float(few_shot_metrics["recall"]),
                    "precision": float(few_shot_metrics["precision"]),
                    "f1_score": float(few_shot_metrics["f1_score"]),
                    "rmse": float(few_shot_metrics["rmse"]),
                    "mae": float(few_shot_metrics["mae"]),
                    "r_squared": float(few_shot_metrics["r_squared"])
                },
                "predictions": few_shot_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in few_shot_parsed]
            },
            "finetuned_model": {
                "metrics": {
                    "accuracy": float(finetuned_metrics["accuracy"]),
                    "recall": float(finetuned_metrics["recall"]),
                    "precision": float(finetuned_metrics["precision"]),
                    "f1_score": float(finetuned_metrics["f1_score"]),
                    "rmse": float(finetuned_metrics["rmse"]),
                    "mae": float(finetuned_metrics["mae"]),
                    "r_squared": float(finetuned_metrics["r_squared"])
                },
                "predictions": finetuned_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in finetuned_parsed]
            }
        }
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print("\n评估结果已保存到 evaluation_results.json")
    
    # 绘制评估指标柱状图
    plot_evaluation_metrics(original_metrics, few_shot_metrics, finetuned_metrics)

if __name__ == "__main__":
    main()