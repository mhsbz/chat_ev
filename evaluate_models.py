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
    Load JSON format test data
    """
    print(f"Reading JSON test data: {json_test_path}...")
    with open(json_test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Successfully read JSON test data, total {len(test_data)} records")
    return test_data

def prepare_json_prompts(json_test_data):
    """
    Prepare prompts and labels for evaluation (from JSON data)
    """
    prompts = []
    labels = []
    
    # Regular expression to parse true labels from model responses
    label_pattern = r"(\d+)(?:\s*,\s*|\s+)(\d+(?:\.\d+)?)"
    
    for item in tqdm(json_test_data):
        messages = item["messages"]
        if len(messages) >= 2:
            # User query as prompt
            user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
            if user_message:
                prompts.append(user_message)
            
            # If there's an assistant reply, extract labels from it
            assistant_message = next((msg["content"] for msg in messages if msg["role"] == "assistant"), None)
            if assistant_message:
                matches = re.findall(label_pattern, assistant_message)
                if matches:
                    # Extract the first match
                    overload, power = matches[0]
                    labels.append((int(overload), float(power)))
                else:
                    # If no standard format match, try to find numbers directly
                    numbers = re.findall(r"\d+(?:\.\d+)?", assistant_message)
                    if len(numbers) >= 2:
                        labels.append((int(float(numbers[0])), float(numbers[1])))
                    else:
                        # If still can't find, add default values
                        print(f"Warning: Unable to parse labels from reply: {assistant_message}")
                        labels.append((0, 0.0))
    
    return prompts, labels

def generate_predictions(model, tokenizer, prompts, max_new_tokens=50, batch_size=8):
    """
    Generate predictions using the model
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
            # Decode generated text
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            # Extract the model's answer part from the generated text
            response = generated_text[len(batch_prompts[j]):]
            predictions.append(response.strip())
    
    return predictions

def parse_predictions(predictions):
    """
    Parse overload status and power consumption predictions from model's text output
    """
    parsed_results = []
    
    for pred in predictions:
        # Use regular expression to match numbers
        matches = re.findall(r"(\d+)(?:\s*,\s*|\s+)(\d+(?:\.\d+)?)", pred)
        if matches:
            # Extract the first match
            overload, power = matches[0]
            parsed_results.append((int(overload), float(power)))
        else:
            # If no format match, try to find numbers directly
            numbers = re.findall(r"\d+(?:\.\d+)?", pred)
            if len(numbers) >= 2:
                parsed_results.append((int(float(numbers[0])), float(numbers[1])))
            else:
                # If still can't find, add default values
                parsed_results.append((0, 0.0))
    
    return parsed_results

def calculate_metrics(true_values, predicted_values):
    """
    Calculate evaluation metrics: Accuracy, Recall, Precision, F1 Score, RMSE, MAE, and R-squared
    """
    # Extract overload status and power consumption
    y_true_overload = np.array([x[0] for x in true_values])
    y_pred_overload = np.array([x[0] for x in predicted_values])
    
    y_true_power = np.array([x[1] for x in true_values])
    y_pred_power = np.array([x[1] for x in predicted_values])
    
    # Calculate accuracy for overload status
    accuracy = np.mean(y_true_overload == y_pred_overload)
    
    # Calculate recall and precision for overload status
    # True positives: cases where both true and predicted values are 1
    true_positives = np.sum((y_true_overload == 1) & (y_pred_overload == 1))
    # False negatives: cases where true value is 1 but predicted value is 0
    false_negatives = np.sum((y_true_overload == 1) & (y_pred_overload == 0))
    # False positives: cases where true value is 0 but predicted value is 1
    false_positives = np.sum((y_true_overload == 0) & (y_pred_overload == 1))
    # True negatives: cases where both true and predicted values are 0
    true_negatives = np.sum((y_true_overload == 0) & (y_pred_overload == 0))
    
    # Calculate recall (true positives / (true positives + false negatives))
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0
    
    # Calculate precision (true positives / (true positives + false positives))
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0
    
    # Calculate F1 score (2 * (precision * recall) / (precision + recall))
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    # Calculate RMSE and MAE for power consumption
    rmse = np.sqrt(mean_squared_error(y_true_power, y_pred_power))
    mae = mean_absolute_error(y_true_power, y_pred_power)
    
    # Calculate R-squared (coefficient of determination)
    # R² = 1 - (sum of squared residuals / total sum of squares)
    y_true_mean = np.mean(y_true_power)
    ss_total = np.sum((y_true_power - y_true_mean) ** 2)  # Total sum of squares
    ss_residual = np.sum((y_true_power - y_pred_power) ** 2)  # Sum of squared residuals
    
    # Avoid division by zero
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
    Evaluate model performance
    """
    print(f"\nEvaluating model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model.cuda()
    
    # If it's a fine-tuned model, load the LoRA adapter
    if is_peft and adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = generate_predictions(model, tokenizer, test_prompts)
    
    # Parse prediction results
    print("Parsing prediction results...")
    parsed_predictions = parse_predictions(predictions)
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    metrics = calculate_metrics(test_labels, parsed_predictions)
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return metrics, predictions, parsed_predictions

def plot_evaluation_metrics(original_metrics, checkpoint_50_metrics, checkpoint_100_metrics, checkpoint_200_metrics, checkpoint_300_metrics, checkpoint_400_metrics):
    """
    Plot bar charts comparing evaluation metrics
    """
    print("\nPlotting evaluation metrics bar charts...")
    
    # Set font for displaying text
    try:
        plt.rcParams['axes.unicode_minus'] = False  # For displaying minus sign correctly
    except:
        print("Warning: Unable to set font, text may not display correctly")
    
    # Plot two groups of charts: overload status metrics and power consumption metrics
    # 1. Overload status metrics (Accuracy, Recall, Precision, F1 Score)
    overload_metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
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
    
    # 2. Power consumption metrics (RMSE, MAE, and R-squared)
    power_metrics = ['RMSE', 'MAE', 'R²']
    original_power_values = [original_metrics['rmse'], original_metrics['mae'], original_metrics['r_squared']]
    checkpoint_50_power_values = [checkpoint_50_metrics['rmse'], checkpoint_50_metrics['mae'], checkpoint_50_metrics['r_squared']]
    checkpoint_100_power_values = [checkpoint_100_metrics['rmse'], checkpoint_100_metrics['mae'], checkpoint_100_metrics['r_squared']]
    checkpoint_200_power_values = [checkpoint_200_metrics['rmse'], checkpoint_200_metrics['mae'], checkpoint_200_metrics['r_squared']]
    checkpoint_300_power_values = [checkpoint_300_metrics['rmse'], checkpoint_300_metrics['mae'], checkpoint_300_metrics['r_squared']]
    checkpoint_400_power_values = [checkpoint_400_metrics['rmse'], checkpoint_400_metrics['mae'], checkpoint_400_metrics['r_squared']]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Plot overload status metrics
    x1 = np.arange(len(overload_metrics))  # Label positions
    width = 0.14  # Width of the bars
    
    rects1_1 = ax1.bar(x1 - 2.5*width, original_overload_values, width, label='base', color='skyblue')
    rects1_2 = ax1.bar(x1 - 1.5*width, checkpoint_50_overload_values, width, label='checkpoint-50', color='orange')
    rects1_3 = ax1.bar(x1 - 0.5*width, checkpoint_100_overload_values, width, label='checkpoint-100', color='lightgreen')
    rects1_4 = ax1.bar(x1 + 0.5*width, checkpoint_200_overload_values, width, label='checkpoint-200', color='salmon')
    rects1_5 = ax1.bar(x1 + 1.5*width, checkpoint_300_overload_values, width, label='checkpoint-300', color='mediumpurple')
    rects1_6 = ax1.bar(x1 + 2.5*width, checkpoint_400_overload_values, width, label='checkpoint-400', color='gold')
    
    # Add titles and axis labels
    ax1.set_title('Overload Status Prediction Metrics Comparison', fontsize=16)
    ax1.set_ylabel('Metric Value', fontsize=14)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(overload_metrics, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 1.0)  # Set y-axis range to 0-1, as these metrics are ratios
    
    # Plot power consumption metrics
    x2 = np.arange(len(power_metrics))  # Label positions
    
    rects2_1 = ax2.bar(x2 - 2.5*width, original_power_values, width, label='base', color='skyblue')
    rects2_2 = ax2.bar(x2 - 1.5*width, checkpoint_50_power_values, width, label='checkpoint-50', color='orange')
    rects2_3 = ax2.bar(x2 - 0.5*width, checkpoint_100_power_values, width, label='checkpoint-100', color='lightgreen')
    rects2_4 = ax2.bar(x2 + 0.5*width, checkpoint_200_power_values, width, label='checkpoint-200', color='salmon')
    rects2_5 = ax2.bar(x2 + 1.5*width, checkpoint_300_power_values, width, label='checkpoint-300', color='mediumpurple')
    rects2_6 = ax2.bar(x2 + 2.5*width, checkpoint_400_power_values, width, label='checkpoint-400', color='gold')
    
    # Add titles and axis labels
    ax2.set_title('Power Consumption Prediction Metrics Comparison', fontsize=16)
    ax2.set_ylabel('Metric Value', fontsize=14)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(power_metrics, fontsize=12)
    ax2.legend(fontsize=12)
    
    # Add value labels on the bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.4f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
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
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the chart
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Evaluation metrics bar chart saved as model_comparison.png")
    
    # Plot training steps vs. performance metrics line charts
    plt.figure(figsize=(20, 15))
    
    # Set training steps
    steps = [0, 50, 100, 200, 300, 400]  # 0 represents the original model
    
    # 1. Overload status accuracy curve
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
    plt.title('Training Steps vs. Overload Status Accuracy', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True)
    
    # 2. Overload status F1 score curve
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
    plt.title('Training Steps vs. Overload Status F1 Score', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True)
    
    # 3. Power consumption RMSE curve
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
    plt.title('Training Steps vs. Power Consumption RMSE', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True)
    
    # 4. Power consumption MAE curve
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
    plt.title('Training Steps vs. Power Consumption MAE', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.grid(True)
    
    # 5. Power consumption R-squared curve
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
    plt.title('Training Steps vs. Power Consumption R-squared', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('R-squared', fontsize=12)
    plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("Training progress line chart saved as training_progress.png")

def main():
    # Configuration parameters
    original_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Same as the model used in the training script
    checkpoint_50_adapter_path = "checkpoints/checkpoint-50"  # Save path for 50-step checkpoint model
    checkpoint_100_adapter_path = "checkpoints/checkpoint-100"  # Save path for 100-step checkpoint model
    checkpoint_200_adapter_path = "checkpoints/checkpoint-200"  # Save path for 200-step checkpoint model
    checkpoint_300_adapter_path = "checkpoints/checkpoint-300"  # Save path for 300-step checkpoint model
    checkpoint_400_adapter_path = "checkpoints/checkpoint-400"  # Save path for 400-step checkpoint model
    json_test_data_path = "sft_data_test.json"  # JSON test data path
    
    # Print evaluation configuration
    print("=== Power Grid Safety Prediction Model Evaluation ===")
    print(f"Original model: {original_model_name}")
    print(f"Checkpoint-50 adapter: {checkpoint_50_adapter_path}")
    print(f"Checkpoint-100 adapter: {checkpoint_100_adapter_path}")
    print(f"Checkpoint-200 adapter: {checkpoint_200_adapter_path}")
    print(f"Checkpoint-300 adapter: {checkpoint_300_adapter_path}")
    print(f"Checkpoint-400 adapter: {checkpoint_400_adapter_path}")
    

    test_data = load_json_test_data(json_test_data_path)
    test_prompts, test_labels = prepare_json_prompts(test_data)
    
    print(f"Generated {len(test_prompts)} test samples")
    
    # Evaluate original model
    original_metrics, original_predictions, original_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels
    )
    
    # Evaluate 50-step checkpoint model
    checkpoint_50_metrics, checkpoint_50_predictions, checkpoint_50_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_50_adapter_path
    )
    
    # Evaluate 100-step checkpoint model
    checkpoint_100_metrics, checkpoint_100_predictions, checkpoint_100_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_100_adapter_path
    )
    
    # Evaluate 200-step checkpoint model
    checkpoint_200_metrics, checkpoint_200_predictions, checkpoint_200_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_200_adapter_path
    )
    
    # Evaluate 300-step checkpoint model
    checkpoint_300_metrics, checkpoint_300_predictions, checkpoint_300_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_300_adapter_path
    )
    
    # Evaluate 400-step checkpoint model
    checkpoint_400_metrics, checkpoint_400_predictions, checkpoint_400_parsed = evaluate_model(
        original_model_name, 
        original_model_name, 
        test_prompts, 
        test_labels, 
        is_peft=True, 
        adapter_path=checkpoint_400_adapter_path
    )
    
    # Print evaluation metrics
    print("\n=== Evaluation Metrics ===")
    print("Original model:")
    print(f"  Overload Status Accuracy: {original_metrics['accuracy']:.4f}")
    print(f"  Overload Status Recall: {original_metrics['recall']:.4f}")
    print(f"  Overload Status Precision: {original_metrics['precision']:.4f}")
    print(f"  Overload Status F1 Score: {original_metrics['f1_score']:.4f}")
    print(f"  Power Consumption RMSE: {original_metrics['rmse']:.4f}")
    print(f"  Power Consumption MAE: {original_metrics['mae']:.4f}")
    print(f"  Power Consumption R-squared: {original_metrics['r_squared']:.4f}")
    
    print("\nCheckpoint-50 model:")
    print(f"  Overload Status Accuracy: {checkpoint_50_metrics['accuracy']:.4f}")
    print(f"  Overload Status Recall: {checkpoint_50_metrics['recall']:.4f}")
    print(f"  Overload Status Precision: {checkpoint_50_metrics['precision']:.4f}")
    print(f"  Overload Status F1 Score: {checkpoint_50_metrics['f1_score']:.4f}")
    print(f"  Power Consumption RMSE: {checkpoint_50_metrics['rmse']:.4f}")
    print(f"  Power Consumption MAE: {checkpoint_50_metrics['mae']:.4f}")
    print(f"  Power Consumption R-squared: {checkpoint_50_metrics['r_squared']:.4f}")
    
    print("\nCheckpoint-100 model:")
    print(f"  Overload Status Accuracy: {checkpoint_100_metrics['accuracy']:.4f}")
    print(f"  Overload Status Recall: {checkpoint_100_metrics['recall']:.4f}")
    print(f"  Overload Status Precision: {checkpoint_100_metrics['precision']:.4f}")
    print(f"  Overload Status F1 Score: {checkpoint_100_metrics['f1_score']:.4f}")
    print(f"  Power Consumption RMSE: {checkpoint_100_metrics['rmse']:.4f}")
    print(f"  Power Consumption MAE: {checkpoint_100_metrics['mae']:.4f}")
    print(f"  Power Consumption R-squared: {checkpoint_100_metrics['r_squared']:.4f}")
    
    print("\nCheckpoint-200 model:")
    print(f"  Overload Status Accuracy: {checkpoint_200_metrics['accuracy']:.4f}")
    print(f"  Overload Status Recall: {checkpoint_200_metrics['recall']:.4f}")
    print(f"  Overload Status Precision: {checkpoint_200_metrics['precision']:.4f}")
    print(f"  Overload Status F1 Score: {checkpoint_200_metrics['f1_score']:.4f}")
    print(f"  Power Consumption RMSE: {checkpoint_200_metrics['rmse']:.4f}")
    print(f"  Power Consumption MAE: {checkpoint_200_metrics['mae']:.4f}")
    print(f"  Power Consumption R-squared: {checkpoint_200_metrics['r_squared']:.4f}")
    
    print("\nCheckpoint-300 model:")
    print(f"  Overload Status Accuracy: {checkpoint_300_metrics['accuracy']:.4f}")
    print(f"  Overload Status Recall: {checkpoint_300_metrics['recall']:.4f}")
    print(f"  Overload Status Precision: {checkpoint_300_metrics['precision']:.4f}")
    print(f"  Overload Status F1 Score: {checkpoint_300_metrics['f1_score']:.4f}")
    print(f"  Power Consumption RMSE: {checkpoint_300_metrics['rmse']:.4f}")
    print(f"  Power Consumption MAE: {checkpoint_300_metrics['mae']:.4f}")
    print(f"  Power Consumption R-squared: {checkpoint_300_metrics['r_squared']:.4f}")
    
    print("\nCheckpoint-400 model:")
    print(f"  Overload Status Accuracy: {checkpoint_400_metrics['accuracy']:.4f}")
    print(f"  Overload Status Recall: {checkpoint_400_metrics['recall']:.4f}")
    print(f"  Overload Status Precision: {checkpoint_400_metrics['precision']:.4f}")
    print(f"  Overload Status F1 Score: {checkpoint_400_metrics['f1_score']:.4f}")
    print(f"  Power Consumption RMSE: {checkpoint_400_metrics['rmse']:.4f}")
    print(f"  Power Consumption MAE: {checkpoint_400_metrics['mae']:.4f}")
    print(f"  Power Consumption R-squared: {checkpoint_400_metrics['r_squared']:.4f}")
    
    # Calculate performance improvement percentages relative to the original model
    # Checkpoint-50
    checkpoint_50_accuracy_improvement = ((checkpoint_50_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_50_recall_improvement = ((checkpoint_50_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_50_precision_improvement = ((checkpoint_50_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_50_f1_improvement = ((checkpoint_50_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_50_rmse_improvement = ((original_metrics['rmse'] - checkpoint_50_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_50_mae_improvement = ((original_metrics['mae'] - checkpoint_50_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_50_r_squared_improvement = ((checkpoint_50_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # Checkpoint-100
    checkpoint_100_accuracy_improvement = ((checkpoint_100_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_100_recall_improvement = ((checkpoint_100_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_100_precision_improvement = ((checkpoint_100_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_100_f1_improvement = ((checkpoint_100_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_100_rmse_improvement = ((original_metrics['rmse'] - checkpoint_100_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_100_mae_improvement = ((original_metrics['mae'] - checkpoint_100_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_100_r_squared_improvement = ((checkpoint_100_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # Checkpoint-200
    checkpoint_200_accuracy_improvement = ((checkpoint_200_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_200_recall_improvement = ((checkpoint_200_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_200_precision_improvement = ((checkpoint_200_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_200_f1_improvement = ((checkpoint_200_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_200_rmse_improvement = ((original_metrics['rmse'] - checkpoint_200_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_200_mae_improvement = ((original_metrics['mae'] - checkpoint_200_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_200_r_squared_improvement = ((checkpoint_200_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # Checkpoint-300
    checkpoint_300_accuracy_improvement = ((checkpoint_300_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_300_recall_improvement = ((checkpoint_300_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_300_precision_improvement = ((checkpoint_300_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_300_f1_improvement = ((checkpoint_300_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_300_rmse_improvement = ((original_metrics['rmse'] - checkpoint_300_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_300_mae_improvement = ((original_metrics['mae'] - checkpoint_300_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_300_r_squared_improvement = ((checkpoint_300_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    # Checkpoint-400
    checkpoint_400_accuracy_improvement = ((checkpoint_400_metrics['accuracy'] - original_metrics['accuracy']) / original_metrics['accuracy']) * 100 if original_metrics['accuracy'] > 0 else float('inf')
    checkpoint_400_recall_improvement = ((checkpoint_400_metrics['recall'] - original_metrics['recall']) / original_metrics['recall']) * 100 if original_metrics['recall'] > 0 else float('inf')
    checkpoint_400_precision_improvement = ((checkpoint_400_metrics['precision'] - original_metrics['precision']) / original_metrics['precision']) * 100 if original_metrics['precision'] > 0 else float('inf')
    checkpoint_400_f1_improvement = ((checkpoint_400_metrics['f1_score'] - original_metrics['f1_score']) / original_metrics['f1_score']) * 100 if original_metrics['f1_score'] > 0 else float('inf')
    checkpoint_400_rmse_improvement = ((original_metrics['rmse'] - checkpoint_400_metrics['rmse']) / original_metrics['rmse']) * 100 if original_metrics['rmse'] > 0 else float('inf')
    checkpoint_400_mae_improvement = ((original_metrics['mae'] - checkpoint_400_metrics['mae']) / original_metrics['mae']) * 100 if original_metrics['mae'] > 0 else float('inf')
    checkpoint_400_r_squared_improvement = ((checkpoint_400_metrics['r_squared'] - original_metrics['r_squared']) / abs(original_metrics['r_squared'])) * 100 if original_metrics['r_squared'] != 0 else float('inf')
    
    print("\nPerformance Improvement of Checkpoint Models:")
    print("Checkpoint-50 model:")
    print(f"  Overload Status Accuracy improvement: {checkpoint_50_accuracy_improvement:.2f}%")
    print(f"  Overload Status Recall improvement: {checkpoint_50_recall_improvement:.2f}%")
    print(f"  Overload Status Precision improvement: {checkpoint_50_precision_improvement:.2f}%")
    print(f"  Overload Status F1 Score improvement: {checkpoint_50_f1_improvement:.2f}%")
    print(f"  Power Consumption RMSE reduction: {checkpoint_50_rmse_improvement:.2f}%")
    print(f"  Power Consumption MAE reduction: {checkpoint_50_mae_improvement:.2f}%")
    print(f"  Power Consumption R-squared improvement: {checkpoint_50_r_squared_improvement:.2f}%")
    
    print("\nCheckpoint-100 model:")
    print(f"  Overload Status Accuracy improvement: {checkpoint_100_accuracy_improvement:.2f}%")
    print(f"  Overload Status Recall improvement: {checkpoint_100_recall_improvement:.2f}%")
    print(f"  Overload Status Precision improvement: {checkpoint_100_precision_improvement:.2f}%")
    print(f"  Overload Status F1 Score improvement: {checkpoint_100_f1_improvement:.2f}%")
    print(f"  Power Consumption RMSE reduction: {checkpoint_100_rmse_improvement:.2f}%")
    print(f"  Power Consumption MAE reduction: {checkpoint_100_mae_improvement:.2f}%")
    print(f"  Power Consumption R-squared improvement: {checkpoint_100_r_squared_improvement:.2f}%")
    
    print("\nCheckpoint-200 model:")
    print(f"  Overload Status Accuracy improvement: {checkpoint_200_accuracy_improvement:.2f}%")
    print(f"  Overload Status Recall improvement: {checkpoint_200_recall_improvement:.2f}%")
    print(f"  Overload Status Precision improvement: {checkpoint_200_precision_improvement:.2f}%")
    print(f"  Overload Status F1 Score improvement: {checkpoint_200_f1_improvement:.2f}%")
    print(f"  Power Consumption RMSE reduction: {checkpoint_200_rmse_improvement:.2f}%")
    print(f"  Power Consumption MAE reduction: {checkpoint_200_mae_improvement:.2f}%")
    print(f"  Power Consumption R-squared improvement: {checkpoint_200_r_squared_improvement:.2f}%")
    
    print("\nCheckpoint-300 model:")
    print(f"  Overload Status Accuracy improvement: {checkpoint_300_accuracy_improvement:.2f}%")
    print(f"  Overload Status Recall improvement: {checkpoint_300_recall_improvement:.2f}%")
    print(f"  Overload Status Precision improvement: {checkpoint_300_precision_improvement:.2f}%")
    print(f"  Overload Status F1 Score improvement: {checkpoint_300_f1_improvement:.2f}%")
    print(f"  Power Consumption RMSE reduction: {checkpoint_300_rmse_improvement:.2f}%")
    print(f"  Power Consumption MAE reduction: {checkpoint_300_mae_improvement:.2f}%")
    print(f"  Power Consumption R-squared improvement: {checkpoint_300_r_squared_improvement:.2f}%")
    
    print("\nCheckpoint-400 model:")
    print(f"  Overload Status Accuracy improvement: {checkpoint_400_accuracy_improvement:.2f}%")
    print(f"  Overload Status Recall improvement: {checkpoint_400_recall_improvement:.2f}%")
    print(f"  Overload Status Precision improvement: {checkpoint_400_precision_improvement:.2f}%")
    print(f"  Overload Status F1 Score improvement: {checkpoint_400_f1_improvement:.2f}%")
    print(f"  Power Consumption RMSE reduction: {checkpoint_400_rmse_improvement:.2f}%")
    print(f"  Power Consumption MAE reduction: {checkpoint_400_mae_improvement:.2f}%")
    print(f"  Power Consumption R-squared improvement: {checkpoint_400_r_squared_improvement:.2f}%")
    
    # Save evaluation results
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
        # Convert non-serializable numpy values to Python native types
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
    
    print("\nEvaluation results saved to evaluation_results.json")
    
    # Plot evaluation metrics bar charts
    plot_evaluation_metrics(original_metrics, checkpoint_50_metrics, checkpoint_100_metrics, checkpoint_200_metrics, checkpoint_300_metrics, checkpoint_400_metrics)

if __name__ == "__main__":
    main()