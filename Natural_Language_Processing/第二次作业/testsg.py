import sys
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Embedding
import numpy as np
sys.path.append("/mnt/ve_share/liushuai/skip-gram/")
weightpath="/mnt/ve_share/liushuai/skip-gram/weight/skip_gram_model1024789.pdparams"
class SkipGram(fluid.dygraph.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        #vocab_size定义了这个skipgram这个模型的词表大小
        #embedding_size定义了词向量的维度是多少
        #init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        #使用paddle.fluid.dygraph提供的Embedding函数，构造一个词向量参数
        #这个参数的大小为：[self.vocab_size, self.embedding_size]
        #数据类型为：float32
        #这个参数的名称为：embedding_para
        #这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = Embedding(
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5/embedding_size, high=0.5/embedding_size)))

        #使用paddle.fluid.dygraph提供的Embedding函数，构造另外一个词向量参数
        #这个参数的大小为：[self.vocab_size, self.embedding_size]
        #数据类型为：float32
        #这个参数的名称为：embedding_para_out
        #这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        #跟上面不同的是，这个参数的名称跟上面不同，因此，
        #embedding_para_out和embedding_para虽然有相同的shape，但是权重不共享
        self.embedding_out = Embedding(
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_out_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5/embedding_size, high=0.5/embedding_size)))

    #定义网络的前向计算逻辑
    #center_words是一个tensor（mini-batch），表示中心词
    #target_words是一个tensor（mini-batch），表示目标词
    #label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
    #用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果
    def forward(self, center_words, target_words, label):
        #首先，通过embedding_para（self.embedding）参数，将mini-batch中的词转换为词向量
        #这里center_words和eval_words_emb查询的是一个相同的参数
        #而target_words_emb查询的是另一个参数
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        #center_words_emb = [batch_size, embedding_size]
        #target_words_emb = [batch_size, embedding_size]
        #我们通过点乘的方式计算中心词到目标词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率。
        word_sim = fluid.layers.elementwise_mul(center_words_emb, target_words_emb)
        word_sim = fluid.layers.reduce_sum(word_sim, dim = -1)
        word_sim = fluid.layers.reshape(word_sim, shape=[-1])
        pred = fluid.layers.sigmoid(word_sim)

        #通过估计的输出概率定义损失函数，注意我们使用的是sigmoid_cross_entropy_with_logits函数
        #将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred
        
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(word_sim, label)
        loss = fluid.layers.reduce_mean(loss)

        #返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
        return pred, loss
def load_text8():
    with open("./text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus
corpus = load_text8()

def build_dict(corpus):
    #首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    #将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    #一般来说，出现频率高的高频词往往是：I，the，you这种代词，而出现频率低的词，往往是一些名词，如：nlp
    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)
    
    #构造3个不同的词典，分别存储，
    #每个词到id的映射关系：word2id_dict
    #每个id出现的频率：word2id_freq
    #每个id到词的映射关系：id2word_dict
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    #按照频率，从高到低，开始遍历每个单词，并为这个单词构造一个独一无二的id
    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[curr_id] = freq #以id为标识，把标识加到freq中
        id2word_dict[curr_id] = word #

    return word2id_freq, word2id_dict, id2word_dict

def get_similar_tokens(token1, token2, embed):
    W = embed.numpy()
    x = W[word2id_dict[token1]]
    y = W[word2id_dict[token2]]
    cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    min_val = -1.0
    max_val = 1.0
    cos_sim_mapped = (cos_sim - min_val) / (max_val - min_val) * 10
    return cos_sim_mapped
word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)

vocab_size,embedding_size=len(word2id_dict),200
skip_gram_model = SkipGram(vocab_size, embedding_size)
model_state_dict = paddle.load('/mnt/ve_share/liushuai/skip-gram/weight/skip_gram_model1024789.pdparams')
skip_gram_model.set_state_dict(model_state_dict)
with open("./output.txt","w")as outf:
    pass
with open ("./wordsim353_agreed.txt")as f:
    datas=f.readlines()
    tmplist=[]
    origindata=[]
    scoredata=[]
    for data in datas:
        data=data.strip().split('\t')
        token1=data[-3]
        token2=data[-2]
        origindata.append((data[-1]))

        with open("./output.txt","a+")as outf:
            if token1 not in word2id_dict or token2 not in word2id_dict:
                score=0
            else:
                score=get_similar_tokens(token1,token2,skip_gram_model.embedding.weight)
                score="{:.2f}".format(score)
            scoredata.append(score)
            new_data = "\t".join(data) + "\t" + str(score) + "\n" 
            outf.write(new_data)
        