import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer,SFTConfig
from peft import LoraConfig, get_peft_model

def main():
    # 加载模型和分词器
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 配置 LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "embed_tokens"
            ],
        lora_dropout=0.2,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用 LoRA
    model = get_peft_model(model, lora_config)

    # 检测CUDA可用性
    if torch.cuda.is_available():
        model = model.cuda()
        print("检测到CUDA设备，已启用GPU加速")
    else:
        print("未检测到CUDA设备，使用CPU训练")

    # 加载数据集
    # 加载完整数据集
    full_dataset = load_dataset('json', data_files='sft_data_train.json', split='train')
    
    # 随机切分数据集为训练集(80%)、验证集(10%)和测试集(10%)
    splits = full_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = splits['train']
    valid_dataset = splits['test']

    # 训练参数
    training_args = SFTConfig(
        output_dir="checkpoints",
        num_train_epochs=1,
        max_grad_norm=0.3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_gpu_eval_batch_size=4,
        eval_accumulation_steps=4,
        learning_rate=6e-5,
        bf16=torch.cuda.is_available(),  # 自动根据CUDA可用性启用bf16
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        max_steps=300,
        save_total_limit=10,
        optim="adamw_torch",
        warmup_ratio=0.1,
        ddp_find_unused_parameters=False,
        report_to="wandb",
    )

    # 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        # tokenizer=tokenizer,
        # packing=False,
        # max_seq_length=2048,
    )

    # 开始训练
    trainer.train()
    
    # 合并LoRA权重到基础模型
    print("正在合并LoRA权重到基础模型...")
    # 指定adapter和base_model的保存路径
    model = model.merge_and_unload()
    
    # 保存合并后的模型和tokenizer
    output_dir = "llama_finetuned"
    print(f"正在保存完整模型到 {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("模型保存完成！")

### CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 train_llama_lora.py
if __name__ == "__main__":
    main()