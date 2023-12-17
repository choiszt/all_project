import paddle
import paddle.nn as nn
import paddlenlp
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder, LinearChainCrfLoss
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.embeddings import TokenEmbedding
import paddle.nn.initializer as I
paddle.set_device("gpu:1")
# 下载并解压数据集

def convert_tokens_to_ids(tokens, vocab, oov_token=None):
    token_ids = []
    oov_id = vocab.get(oov_token) if oov_token else None
    for token in tokens:
        token_id = vocab.get(token, oov_id)
        token_ids.append(token_id)
    return token_ids


def load_dict(dict_path):
    vocab = {}
    i = 0
    for line in open(dict_path, 'r', encoding='utf-8'):
        key = line.strip('\n')
        vocab[key] = i
        i += 1
    return vocab

def build_map(lists): #
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
                    lines=f.readlines()
                    for num in range(len(lines)):
                        line=lines[num].strip().split(' ')
                        yield line,[]

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]
test_ds = load_dataset(datafiles=('/mnt/ve_share/liushuai/lstm/test.txt'))


# label_vocab = load_dict('data/tag.dic')
# word_vocab = load_dict('data/word.dic')
def convert_example(example):
        tokens, labels = example
        token_ids = convert_tokens_to_ids(tokens, word_vocab, 'OOV')
        label_ids = convert_tokens_to_ids(labels, label_vocab, 'O')
        return token_ids, len(token_ids), label_ids

test_ds.map(convert_example)

batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=word_vocab.get('OOV')),  # token_ids
        Stack(),  # seq_len
        Pad(axis=0, pad_val=label_vocab.get('O'))  # label_ids
    ): fn(samples)

test_loader = paddle.io.DataLoader(
        dataset=test_ds,
        batch_size=32,
        drop_last=False,
        return_list=True,
        collate_fn=batchify_fn,
    	shuffle=False)


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
        init_scale=0.1,
    ):
        super(BiLSTMWithCRF, self).__init__()
        self.embedder = nn.Embedding(word_num, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, "bidirectional", dropout=dropout_prob)
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
        self.crf = LinearChainCrf(label_num,with_start_stop_tag=True)
        self.decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, x_1, seq_len_1):
        x_embed_1 = self.embedder(x_1)
        lstm_out_1, (hidden_1, _) = self.lstm(x_embed_1, sequence_length=seq_len_1)
        out_1 = paddle.concat((hidden_1[-2, :, :], hidden_1[-1, :, :]), axis=1)
        # out = paddle.tanh(self.fc(out_1))
        # out = self.fc(out_1)
        out = paddle.tanh(self.fc(lstm_out_1))
        logits = self.output_layer(out)
        _, pred = self.decoder(logits, seq_len_1)
        return logits, seq_len_1, pred

class BiGRUWithCRF(nn.Layer):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 word_num,
                 label_num,
                 use_w2v_emb=False):
        super(BiGRUWithCRF, self).__init__()
        if use_w2v_emb:
            self.word_emb = TokenEmbedding(
                extended_vocab_path='./conf/word.dic', unknown_token='OOV')
        else:
            self.word_emb = nn.Embedding(word_num, emb_size)
        self.gru = nn.GRU(emb_size,
                          hidden_size,
                          num_layers=2,
                          direction='bidirectional')
        self.fc = nn.Linear(hidden_size * 2, label_num)  # BOS EOS
        self.crf = LinearChainCrf(label_num)
        self.decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, x, lens):
        embs = self.word_emb(x)
        output, _ = self.gru(embs)
        output = self.fc(output)
        _, pred = self.decoder(output, lens)
        return output, lens, pred

# Define the model netword and its loss
network = BiLSTMWithCRF(300, 300, len(word_vocab), len(label_vocab))
model = paddle.Model(network)
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
crf_loss = LinearChainCrfLoss(network.crf)
chunk_evaluator = ChunkEvaluator(label_list=label_vocab.keys(), suffix=True)

id2label=dict(enumerate(label_vocab))

model.prepare(optimizer, crf_loss, chunk_evaluator)
model.load("/mnt/ve_share/liushuai/lstm/results/final.pdparams")
output, lens, pred=model.predict(test_data=test_loader)
length=len(lens)
import tqdm
for num in tqdm.tqdm(range(length)):
    for num2 in range(len(list(lens[num]))):
        index=lens[num][num2]
        templist=list(pred[num][num2][:index])
        with open("/mnt/ve_share/liushuai/lstm/2020212267.txt",'a+')as f:
        	for ele in templist:
                    f.write(id2label[ele].replace("'","")+" ")
                f.write('\n')