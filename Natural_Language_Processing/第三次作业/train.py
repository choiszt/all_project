import paddle
import paddle.nn as nn
import paddlenlp
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator
import paddle.nn.initializer as I
paddle.set_device("gpu:0")
# 下载并解压数据集

def convert_tokens_to_ids(tokens, vocab, oov_token=None):
    token_ids = []
    oov_id = vocab.get(oov_token) if oov_token else None
    for token in tokens:
        token_id = vocab.get(token, oov_id)
        token_ids.append(token_id)
    return token_ids

def build_map(lists): 
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps

word_lists,tag_lists=[],[]
with open("/mnt/ve_share/liushuai/lstm/train.txt",encoding="utf-8")as f:
    with open("/mnt/ve_share/liushuai/lstm/train_TAG.txt",encoding='utf-8')as t:
        lines=f.readlines()
        labels=t.readlines()
        for line in lines:
            word_lists.append(line.strip().split(' '))
        for label in labels:
            tag_lists.append(label.strip().split(' '))
        word_vocab = build_map(word_lists)
        label_vocab = build_map(tag_lists)
word_vocab.update({'OOV':len(word_vocab)})
def load_dataset(datafiles):
    def read(data_path):
        with open(data_path,encoding="utf-8")as f:
            if data_path.split('/')[-1]!='test.txt':
                with open(f'{data_path.split(".txt")[0]}_TAG.txt',encoding='utf-8')as t:
                    lines=f.readlines()
                    labels=t.readlines()
                assert len(lines)==len(labels)
                for num in range(len(lines)):
                    line=lines[num].strip().split(' ')
                    label=labels[num].strip().split(' ')              
                    yield line, label

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]
train_ds, dev_ds = load_dataset(datafiles=('/mnt/ve_share/liushuai/lstm/train.txt', '/mnt/ve_share/liushuai/lstm/dev.txt'))



def convert_example(example):
        tokens, labels = example
        token_ids = convert_tokens_to_ids(tokens, word_vocab, 'OOV')
        label_ids = convert_tokens_to_ids(labels, label_vocab, 'O')
        return token_ids, len(token_ids), label_ids

convert_example(train_ds.__getitem__(1))
train_ds.map(convert_example)
dev_ds.map(convert_example)

batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=word_vocab.get('OOV')),  # token_ids
        Stack(),  # seq_len
        Pad(axis=0, pad_val=label_vocab.get('O'))  # label_ids
    ): fn(samples)

train_loader = paddle.io.DataLoader(
        dataset=train_ds,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)

dev_loader = paddle.io.DataLoader(
        dataset=dev_ds,
        batch_size=32,
        drop_last=True,
        return_list=True,
        collate_fn=batchify_fn)



####----------------model--------------####
class BiLSTMWithCRF(nn.Layer):
    def __init__(
        self,
        embed_dim, #300
        hidden_size, #300
        word_num, #len
        label_num, #num_classes
        num_layers=1,
        dropout_prob=0.0,
        init_scale=0.1,#权重初始化
    ):
        super(BiLSTMWithCRF, self).__init__()
        # 定义词向量层，将输入的词语索引映射为词向量
        self.embedder = nn.Embedding(word_num, embed_dim)
        # 定义 LSTM 层，用于处理输入序列，并提取特征
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, "bidirectional", dropout=dropout_prob)
        # 全连接层输入维度，即 LSTM 输出的维度乘以 2（因为是双向 LSTM）
        self.fc = nn.Linear(
            hidden_size * 2,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(low=-init_scale, high=init_scale)),
        )
        self.output_layer = nn.Linear(
            hidden_size,
            label_num+2,
            weight_attr=paddle.ParamAttr(initializer=I.Uniform(low=-init_scale, high=init_scale)),
        )
        # 定义 CRF 模型，用于对序列标注结果进行建模和预测
        self.crf = LinearChainCrf(label_num,with_start_stop_tag=True)
         # 定义解码器，用于利用 CRF 模型的转移矩阵进行标注结果的预测
        self.decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, x_1, seq_len_1):
        # 输入 x_1 是一个形状为 [batch_size, seq_len] 的整数张量，表示输入的词语索引序列
        # seq_len_1 是一个形状为 [batch_size] 的整数张量，表示每个输入序列的真实长度
        x_embed_1 = self.embedder(x_1)
        # 下面的代码将输入的词语索引映射为词向量，并输入到 LSTM 层中进行处理
        lstm_out_1, (_, _) = self.lstm(x_embed_1, sequence_length=seq_len_1)
        out = paddle.tanh(self.fc(lstm_out_1))
        logits = self.output_layer(out)
         # 利用 CRF 模型和解码器对标注结果进行预测，并返回预测结果、输入序列长度和标签概率分布
        _, pred = self.decoder(logits, seq_len_1)
        return logits, seq_len_1, pred

# Define the model netword and its loss
network = BiLSTMWithCRF(300, 300, len(word_vocab), len(label_vocab))
model = paddle.Model(network)

optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
crf_loss = LinearChainCrfLoss(network.crf)
chunk_evaluator = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)

visualcallback = paddle.callbacks.VisualDL(log_dir='/mnt/ve_share/liushuai/lstm/n2e300')
lrcallback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
stopcallback = paddle.callbacks.EarlyStopping(
    'loss',
    mode='min',
    patience=6,
    verbose=1,
    min_delta=0,
    baseline=None,
    save_best_model=True)
callbacks=[visualcallback,lrcallback,stopcallback]
model.prepare(optimizer, crf_loss, chunk_evaluator)
model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=3,
              save_dir='/mnt/ve_share/liushuai/lstm/n2e300',
              log_freq=1,
              callbacks=callbacks)