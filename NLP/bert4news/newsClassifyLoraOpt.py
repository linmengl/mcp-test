import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding


def split_label_text(example):
    parts = example['text'].split('\t', 1)  # 只按第一个空格拆分
    return {
        "label": parts[0],
        "text": parts[1] if len(parts) > 1 else ""
    }


# 1. 加载今日头条数据集
# 应用到每个子集
dataset = load_dataset("spiritx2023/ThuCnews", cache_dir="./data")
# 拆分文本为 text label
dataset = dataset.map(split_label_text)

# 构造标签映射（将 label 字符串转为数字 id）
label_list = sorted(list(set(dataset['train']['label'])))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(label_list)


# 将 label 字符串映射为整数 id
def convert_label(example):
    example["label"] = label2id[example["label"]]
    return example


# 将 label 字符串映射为整数 id
dataset = dataset.map(convert_label)

# 3. 加载预训练的 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels)

# 配置 LoRA
config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=16,  # 缩放因子
    target_modules=["query", "value"],  # 需要应用 LoRA 的模块
    lora_dropout=0.1,  # Dropout 概率
    bias="none",  # 是否训练偏置项
    modules_to_save=["classifier"],  # 需要保存的模块
)

# 将 LoRA 配置应用到模型
model = get_peft_model(model, config)
model = model.to(device)
model.print_trainable_parameters()


# 定义分词函数
def tokenize_fc(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)


# 对数据集的每个子集应用分词函数
dataset = dataset.map(tokenize_fc, batched=True)

# 4. 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 训练参数
training_args = TrainingArguments(
    output_dir="./output",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)


# 评估函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro")
    }


# 8. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print('模型训练开始')
# 9. 训练
trainer.train()
print('模型训练完成')

# 10. 保存模型
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# 11. 在验证集上评估
eval_result = trainer.evaluate()
print("Eval result:", eval_result)

# 12. 示例预测
test_text = "据新华社消息，北京将举办2025年国际人工智能大会"
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True).to(device)
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = torch.argmax(logits).item()
print("预测类别:", id2label[predicted_class_id])
