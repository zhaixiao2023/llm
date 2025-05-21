import json
import pandas as pd

# 读取 TSV 文件
df = pd.read_csv("llm-classification-finetuning/train.csv")

# 用于存储 DPO 格式的数据
output_data = []

# 遍历 DataFrame
for _, row in df.iterrows():
    try:
        # 将 prompt 列（字符串形式的列表）转为真实列表后拼成字符串
        prompts = json.loads(row['prompt'])
        full_prompt = "\n\n".join(prompts)

        # 判断谁是 chosen 谁是 rejected
        if row['winner_model_a'] == 1:
            chosen = row['response_a']
            rejected = row['response_b']
        elif row['winner_model_b'] == 1:
            chosen = row['response_b']
            rejected = row['response_a']
        elif row['winner_tie'] == 1:
            # 若是平局，DPO可以复制一份反过来的对
            chosen = row['response_a']
            rejected = row['response_b']
            output_data.append({
                "prompt": full_prompt,
                "chosen": chosen,
                "rejected": rejected
            })
            chosen = row['response_b']
            rejected = row['response_a']
        else:
            continue  # 跳过没有 winner 的行

        output_data.append({
            "prompt": full_prompt,
            "chosen": chosen,
            "rejected": rejected
        })

    except KeyError as e:
        print(f"KeyError: {e} in row {row}")
        continue
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e} in row {row}")
        continue

# 保存为 JSONL 文件
output_file = "processed/dpo_data.jsonl"  # 修改为相对路径
with open(output_file, "w", encoding="utf-8") as f:
    for item in output_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Processed data saved to {output_file}")