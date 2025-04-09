from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm, trange
import torch

from bertSimple1 import BertForSequenceClassification

# 创建分词器
tokenizer = get_tokenizer("basic_english")
print(tokenizer("here is the an example!"))

input_ids = tokenizer.encode(
                  example.text_a,
                  add_special_tokens=True,
                  max_length=min(max_length, tokenizer.max_len),
              )

attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

model = BertForSequenceClassification.from_pretrained()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def training_step(batch):
    input_ids, token_type_ids, attention_mask, labels = batch
    # 将数据发送到GPU
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels_voc.to(device)

    logits = model(input_ids,
                   token_type_ids=token_type_ids,
                   attention_mask=attention_mask,
                   labels=labels)
    loss_fct = BCEWithLogitsLoss()
    loss = loss_fct(logits.view(-1, num_labels_cate), labels.view(-1, num_labels_cate).float())
    loss.backward()

for epoch in trange(0, args.num_train_epochs):
  # 把模型设置为训练状态。
  model.train()
  for step, batch in enumerate(tqdm(train_dataLoader, desc='Iteration')):
      # 训练的核心环节
    step_loss = training_step(batch)
    tr_loss += step_loss[0]
    optimizer.step()
    optimizer.zero_grad()