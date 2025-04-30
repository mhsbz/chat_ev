#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

def load_dataset_from_json(file_path):
    """从JSON文件加载数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为Hugging Face Dataset格式
    formatted_data = []
    for item in data:
        formatted_data.append({
            "messages": item["messages"]
        })
    
    return Dataset.from_list(formatted_data)

def main():
    # 模型参数
    model_name = "meta-llama/Meta-Llama-3.2-1B-Instruct"
    dataset_path = "sft_training_data.json"
    output_dir = "./llama-3.2-1b-ev-lora"
    max_seq_length = 2048
    
    # 打印训练配置
    print(f"加载模型: {model_name}")
    print(f"加载数据集: {dataset_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 加载训练数据集
    dataset = load_dataset_from_json(dataset_path)
    print(f"加载数据集成功，包含 {len(dataset)} 个训练样本")
    
    # 设置训练参数
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3.0,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        weight_decay=0.001,
        warmup_ratio=0.03,
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        report_to="tensorboard",
        remove_unused_columns=False,
    )
    
    # 配置SFT训练器
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=dataset.select(range(min(100, len(dataset)))),
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_text_field="messages",
        max_seq_length=max_seq_length,
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print(f"训练完成，保存模型到 {output_dir}")
    trainer.save_model(output_dir)
    
if __name__ == "__main__":
    main()