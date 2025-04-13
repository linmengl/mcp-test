# 方案 1：使用 HuggingFace Datasets（推荐）
# dataset = load_dataset("imdb", split="train", cache_dir="./data")
# print('dataset[0]=',dataset[0])
# print('dataset[1]=',dataset[1])
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# print('bert-base-uncased分词结果=',tokenizer("here is the an example!"))
#
# encodings = tokenizer(dataset[0]["text"], truncation=True, padding=True)
# print('encodings结果=',encodings["input_ids"])
import torchtext
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

torchtext.disable_torchtext_deprecation_warning()

train_iter = load_dataset(path="imdb", cache_dir="./data", split="train")
print(train_iter)

# 创建分词器
tokenizer = get_tokenizer("basic_english")
print(tokenizer("here is the an example!"))


# 构建词汇表
def yield_tokens(data_iter):
    for item in data_iter:
        # print('is text',item['text'],'label',item['label'])
        yield tokenizer(item['text'])


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(vocab(tokenizer('here is the an example <pad> <pad>')))

# 数据处理pipelines
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0

print(text_pipeline('here is the an example'))
print(label_pipeline('pos'))