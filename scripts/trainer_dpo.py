# scripts/train_dpo.py
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import os

# 路径配置
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_path = "../data/processed/dpo_data.jsonl"
output_dir = "../models/dpo-tinyllama"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # 避免 pad 报错

model = AutoModelForCausalLM.from_pretrained(model_name)

# 训练参数配置
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    learning_rate=5e-5,
    fp16=True,  # T4支持fp16
    report_to="none"
)

# DPO配置
dpo_config = DPOConfig(
    beta=0.1,  # 重要的 DPO 强度参数
    max_prompt_length=512,
    max_length=1024,
)

# 初始化 trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 如果不设置，默认复制当前 model
    args=training_args,
    dpo_config=dpo_config,
    tokenizer=tokenizer,
    train_dataset=dataset_path,  # 直接传jsonl路径
)

trainer.train()