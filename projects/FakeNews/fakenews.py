from IPython import get_ipython
import os
import pandas as pd
import glob
import torch
import pysnooper
from torch.utils.data import Dataset
import torch.optim as optim
from transformers import BertTokenizer
from IPython.display import clear_output
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
glob.glob('./data/*.csv.zip')

os.system('unzip ./data/train.csv.zip')
df_train = pd.read_csv('train.csv')
df_train.head()

empty_title = ((df_train['title1_zh'].isnull())
               | (df_train['title2_zh'].isnull())
               | (df_train['title2_zh'] == '')
               | (df_train['title2_zh'] == '0'))

df_train = df_train[~empty_title]

MAX_LENGTH = 30
df_train = df_train[~(df_train.title1_zh.apply(lambda x: len(x)) > MAX_LENGTH)]
df_train = df_train[~(df_train.title2_zh.apply(lambda x: len(x)) > MAX_LENGTH)]

SAMPLE_RATE = 0.01
df_train = df_train.sample(frac=SAMPLE_RATE, random_state=2020)
df_train = df_train.reset_index()
df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
df_train.columns = ['text_a', 'text_b', 'label']
df_train.head()

df_train.to_csv('train.tsv', sep='\t', index=False)
print('train samples: ', len(df_train))

df_train.label.value_counts() / len(df_train)

os.system('unzip ./data/test.csv.zip')
df_test = pd.read_csv('test.csv')
df_test = df_test.loc[:, ['title1_zh', 'title2_zh', 'id']]
df_test.columns = ['text_a', 'text_b', 'Id']
df_test.to_csv('test.tsv', sep='\t', index=False)

print('test samples: ', len(df_test))
df_test.head()

ratio = len(df_test) / len(df_train)
print("測試集樣本數 / 訓練集樣本數 = {:.1f} 倍".format(ratio))
"""
實作一個可以用來讀取訓練 / 測試集的 Dataset，這是你需要徹底了解的部分。
此 Dataset 每次將 tsv 裡的一筆成對句子轉換成 BERT 相容的格式，並回傳 3 個 tensors：
- tokens_tensor：兩個句子合併後的索引序列，包含 [CLS] 與 [SEP]
- segments_tensor：可以用來識別兩個句子界限的 binary tensor
- label_tensor：將分類標籤轉換成類別索引的 tensor, 如果是測試集則回傳 None
"""


class FakeNewsDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ['train', 'dev', 'test']
        self.mode = mode

        self.df = pd.read_csv('%s.tsv' % mode, sep='\t').fillna('')
        self.len = len(self.df)
        self.label_map = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
        self.tokenizer = tokenizer

    # @pysnooper.snoop()
    def __getitem__(self, idx):
        if self.mode == 'test':
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)

        word_pieces = ['[CLS]']
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ['[SEP]']
        len_a = len(tokens_a)

        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ['[SEP]']
        len_b = len(word_pieces) - len_a

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segments_tensor = torch.tensor([0] * len_a + [1] * len_b,
                                       dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.len


PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
trainset = FakeNewsDataset('train', tokenizer)

sample_idx = 0
text_a, text_b, label = trainset.df.iloc[sample_idx, :].values
tokens_tensor, segments_tensor, label_tensor = trainset[sample_idx]

tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
combined_text = ''.join(tokens)

print(f"""[原始文本]
句子 1：{text_a}
句子 2：{text_b}
分類  ：{label}

--------------------

[Dataset 回傳的 tensors]
tokens_tensor  ：{tokens_tensor}

segments_tensor：{segments_tensor}

label_tensor   ：{label_tensor}

--------------------

[還原 tokens_tensors]
{combined_text}
""")
"""
實作可以一次回傳一個 mini-batch 的 DataLoader
這個 DataLoader 吃我們上面定義的 `FakeNewsDataset`，
回傳訓練 BERT 時會需要的 4 個 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""


# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
# 剛剛定義的 `FakeNewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids


BATCH_SIZE = 64
trainloader = DataLoader(trainset,
                         batch_size=BATCH_SIZE,
                         collate_fn=create_mini_batch)

data = next(iter(trainloader))
tokens_tensors, segments_tensors, masks_tensors, label_ids = data
print(f"""
tokens_tensors.shape   = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
segments_tensors.shape = {segments_tensors.shape}
{segments_tensors}
------------------------
masks_tensors.shape    = {masks_tensors.shape}
{masks_tensors}
------------------------
label_ids.shape        = {label_ids.shape}
{label_ids}
""")

PRETRAINED_MODEL_PATH = './bert-base-chinese/'
NUM_LABELS = 3

model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH,
                                                      num_labels=NUM_LABELS)
clear_output()

print('{0:10}{1:15}'.format('name', 'module'))
for name, module in model.named_children():
    if name == 'bert':
        for n, _ in module.named_children():
            print(f'{name}:{n}')
    else:
        print('{:15}{}'.format(name, module))
"""
定義一個可以針對特定 DataLoader 取得模型預測結果以及分類準確度的函式
在將 `tokens`、`segments_tensors` 等 tensors
丟入模型時，強力建議指定每個 tensor 對應的參數名稱，以避免 HuggingFace
更新 repo 程式碼並改變參數順序時影響到我們的結果。
"""


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                data = [t.to('cuda:0') for t in data if t is not None]

            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors,
                            token_type_ids=segments_tensors,
                            attention_mask=masks_tensors)
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)

            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

        if compute_acc:
            acc = correct / total
            return predictions, acc

        return predictions


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
model = model.to(device)
_, acc = get_predictions(model, trainloader, compute_acc=True)
print('classification acc: ', acc)


def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


model_params = get_learnable_params(model)
clf_params = get_learnable_params(model.classifier)

print(f"""
整個分類模型的參數量：{sum(p.numel() for p in model_params)}
線性分類器的參數量：{sum(p.numel() for p in clf_params)}
""")

model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

EPOCHS = 6
for epoch in range(EPOCHS):
    running_loss = 0.0
    for data in trainloader:
        tokens_tensors, segments_tensors, masks_tensors, labels = [
            t.to(device) for t in data
        ]

        optimizer.zero_grad()
        outputs = model(input_ids=tokens_tensors,
                        token_type_ids=segments_tensors,
                        attention_mask=masks_tensors,
                        labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

        running_loss += loss

    _, acc = get_predictions(model, trainloader, compute_acc=True)

    print('[epoch %d] loss: %.3f, acc: %.3f' % (epoch + 1, running_loss, acc))

testset = FakeNewsDataset('test', tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)

predictions = get_predictions(model, testloader)
index_map = {v: k for k, v in testset.label_map.items()}

df = pd.DataFrame({"Category": predictions.tolist()})
df['Category'] = df.Category.apply(lambda x: index_map[x])
df_pred = pd.concat([testset.df.loc[:, 'Id'], df.loc[:, 'Category']], axis=1)
df_pred.to_csv('bert_1_prec_training_samples.csv', index=False)
df_pred.head()

predictions = get_predictions(model, trainloader)
df = pd.DataFrame({'predicted': predictions.tolist()})
df['predicted'] = df.predicted.apply(lambda x: index_map[x])
df1 = pd.concat([trainset.df, df.loc[:, 'predicted']], axis=1)
disagreed_tp = ((df1.label == 'disagreed') & \
     (df1.label == df1.predicted) & \
     (df1.text_a.apply(lambda x: True if len(x) < 10 else False)))
df1[disagreed_tp].head()
