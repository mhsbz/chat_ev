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
    计算评估指标：RMSE和MAE
    """
    # 提取过载状态和功率消耗
    y_true_overload = np.array([x[0] for x in true_values])
    y_pred_overload = np.array([x[0] for x in predicted_values])
    
    y_true_power = np.array([x[1] for x in true_values])
    y_pred_power = np.array([x[1] for x in predicted_values])
    
    # 计算过载状态的准确率
    accuracy = np.mean(y_true_overload == y_pred_overload)
    
    # 计算功率消耗的RMSE和MAE
    rmse = np.sqrt(mean_squared_error(y_true_power, y_pred_power))
    mae = mean_absolute_error(y_true_power, y_pred_power)
    
    return {
        "accuracy": accuracy,
        "rmse": rmse,
        "mae": mae
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

def plot_evaluation_metrics(original_metrics, finetuned_metrics):
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
    
    metrics = ['overload_accuracy)', 'RMSE', 'MAE']
    original_values = [original_metrics['accuracy'], original_metrics['rmse'], original_metrics['mae']]
    finetuned_values = [finetuned_metrics['accuracy'], finetuned_metrics['rmse'], finetuned_metrics['mae']]
    
    x = np.arange(len(metrics))  # 标签位置
    width = 0.35  # 柱状图宽度
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, original_values, width, label='base', color='skyblue')
    rects2 = ax.bar(x + width/2, finetuned_values, width, label='finetuned', color='lightgreen')
    
    # 添加标题和坐标轴标签
    ax.set_title('原始模型与微调模型性能对比', fontsize=16)
    ax.set_ylabel('指标值', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    
    # 在柱状图上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # 调整布局
    fig.tight_layout()
    
    # 保存图表
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("评估指标柱状图已保存为 model_comparison.png")

def main():
    # 配置参数
    original_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # 与训练脚本中使用的模型一致
    finetuned_model_adapter_path = "checkpoints/checkpoint-625"  # 微调模型的保存路径
    test_data_path = "original_data/smart_grid_dataset.csv"  # 原始数据集路径
    json_test_data_path = "sft_data_test.json"  # JSON测试数据路径
    use_json_test = True  # 设置为True表示使用JSON测试数据
    
    # 打印评估配置
    print("=== 电网安全预测模型评估 ===")
    print(f"原始模型: {original_model_name}")
    print(f"微调模型适配器: {finetuned_model_adapter_path}")
    print(f"测试数据: {'JSON测试数据' if use_json_test else '原始数据集随机划分'}")
    

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
    
    # 评估微调后的模型
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
    print(f"  功率消耗RMSE: {original_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {original_metrics['mae']:.4f}")
    
    print("\n微调后模型:")
    print(f"  过载状态准确率: {finetuned_metrics['accuracy']:.4f}")
    print(f"  功率消耗RMSE: {finetuned_metrics['rmse']:.4f}")
    print(f"  功率消耗MAE: {finetuned_metrics['mae']:.4f}")
    
    # 性能提升百分比
    accuracy_improvement = ((finetuned_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    rmse_improvement = ((original_metrics['rmse'] - finetuned_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    mae_improvement = ((original_metrics['mae'] - finetuned_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    
    print("\n性能提升:")
    print(f"  过载状态准确率提升: {accuracy_improvement:.2f}%")
    print(f"  功率消耗RMSE降低: {rmse_improvement:.2f}%")
    print(f"  功率消耗MAE降低: {mae_improvement:.2f}%")
    
    # 保存评估结果
    results = {
        "original_model": {
            "metrics": original_metrics,
            "predictions": original_predictions,
            "parsed_predictions": original_parsed
        },
        "finetuned_model": {
            "metrics": finetuned_metrics,
            "predictions": finetuned_predictions,
            "parsed_predictions": finetuned_parsed
        },
        "improvements": {
            "accuracy": accuracy_improvement,
            "rmse": rmse_improvement,
            "mae": mae_improvement
        }
    }
    
    with open("evaluation_results.json", "w") as f:
        # 转换不可序列化的numpy值为Python原生类型
        json_results = {
            "original_model": {
                "metrics": {
                    "accuracy": float(original_metrics["accuracy"]),
                    "rmse": float(original_metrics["rmse"]),
                    "mae": float(original_metrics["mae"])
                },
                "predictions": original_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in original_parsed]
            },
            "finetuned_model": {
                "metrics": {
                    "accuracy": float(finetuned_metrics["accuracy"]),
                    "rmse": float(finetuned_metrics["rmse"]),
                    "mae": float(finetuned_metrics["mae"])
                },
                "predictions": finetuned_predictions,
                "parsed_predictions": [(int(o), float(p)) for o, p in finetuned_parsed]
            }
        }
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print("\n评估结果已保存到 evaluation_results.json")
    
    # 绘制评估指标柱状图
    plot_evaluation_metrics(original_metrics, finetuned_metrics)

if __name__ == "__main__":
    main()