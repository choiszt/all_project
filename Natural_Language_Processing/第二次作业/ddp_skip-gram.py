import io
import os
import sys
import requests
from collections import OrderedDict 
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid
import tqdm
from paddle.fluid.dygraph.nn import Embedding
import paddle.distributed as dist
from paddle.io import Dataset
# 一般来说，在自然语言处理中，需要先对语料进行切词。对于英文来说，可以比较简单地直接使用空格进行切词，代码如下：

#对语料进行预处理（分词）
# 在经过切词后，需要对语料进行统计，为每个词构造ID。一般来说，可以根据每个词在语料中出现的频次构造ID，频次越高，ID越小，便于对词典进行管理。代码如下：
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
def convert_corpus_to_id(corpus, word2id_dict):
    #使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus] #一个text中每个位置的词用id来表示#TODO ls
    return corpus
def subsampling(corpus, word2id_freq):
    
    #这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    #如果一个词的频率很大，那么它被遗弃的概率就很大
    # def discard(word_id):
    #     return random.uniform(0, 1) < 1 - math.sqrt(
    #         1e-4 / word2id_freq[word_id] * len(corpus))
    discard=lambda word_id:random.uniform(0, 1) < 1 - math.sqrt(1e-4 / word2id_freq[word_id] * len(corpus))
    corpus = [word for word in corpus if not discard(word)] #随机清除其中的一些词，如果频率很大，则很可能被遗弃
    return corpus
def build_data(corpus, word2id_dict, word2id_freq, max_window_size = 3, negative_sample_num = 4):
    
    #使用一个list存储处理好的数据
    dataset = []

    #从左到右，开始枚举每个中心点的位置
    for center_word_idx in tqdm.tqdm(range(len(corpus))):
        #以max_window_size为上限，随机采样一个window_size，这样会使得训练更加稳定
        window_size = random.randint(1, max_window_size)
        #当前的中心词就是center_word_idx所指向的词
        center_word = corpus[center_word_idx] #第i个词就是此时的centerword

        #以当前中心词为中心，左右两侧在window_size内的词都可以看成是正样本
        positive_word_range = (max(0, center_word_idx - window_size), min(len(corpus) - 1, center_word_idx + window_size))
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0], positive_word_range[1]+1) if idx != center_word_idx]
        #中心词附近的正样本
        #对于每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for positive_word in positive_word_candidates:
            #首先把（中心词，正样本，label=1）的三元组数据放入dataset中，
            #这里label=1表示这个样本是个正样本
            dataset.append((center_word, positive_word, 1))

            #开始负采样
            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, len(word2id_freq)-1)

                if negative_word_candidate not in positive_word_candidates:
                    #把（中心词，正样本，label=0）的三元组数据放入dataset中，
                    #这里label=0表示这个样本是个负样本
                    dataset.append((center_word, negative_word_candidate, 0))
                    i += 1
    
    return dataset
class text8data(Dataset):
    def __init__(self, data_dir):
        super(text8data, self).__init__()
        self.data_list = []
        #load data from text
        with open(data_dir, "r") as f:
            corpus= f.read().strip("\n")
        f.close()
        #data formating...
        corpus = corpus.strip().lower()
        corpus = corpus.split(" ")
        
        self.word2id_freq, self.word2id_dict, self.id2word_dict = build_dict(corpus)
        corpus = convert_corpus_to_id(corpus, self.word2id_dict)
        corpus = subsampling(corpus, self.word2id_freq)#subsampling
        self.data_list=build_data(corpus, self.word2id_dict, self.word2id_freq)
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
    def __getitem__(self, index):
        # 根据索引，从列表中取出一个图像
        center_word,target_word,label = self.data_list[index]
        return center_word, target_word,label
    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

text8dataset = text8data("/mnt/ve_share/liushuai/skip-gram/1.txt")
print(f"the scale of dataset is {len(text8dataset)}")
#定义skip-gram训练网络结构
#这里我们使用的是paddlepaddle的1.8.0版本
#一般来说，在使用fluid训练的时候，我们需要通过一个类来定义网络结构，这个类继承了fluid.dygraph.Layer
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
            size=[self.vocab_size, self.embedding_size], #253854,200
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


#开始训练，定义一些训练过程中需要使用的超参数
batch_size = 512
batch_size = 1
epoch_num = 3
embedding_size = 200
step = 0
learning_rate = 0.001
train_loader = paddle.io.DataLoader(text8dataset,batch_size=batch_size, num_workers=4,drop_last=False)
#定义一个使用word-embedding查询同义词的函数
#这个函数query_token是要查询的词，k表示要返回多少个最相似的词，embed是我们学习到的word-embedding参数
#我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
#具体实现如下，x代表要查询词的Embedding，Embedding参数矩阵W代表所有词的Embedding
#两者计算Cos得出所有词对查询词的相似度得分向量，排序取top_k放入indices列表
def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[text8dataset.word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(text8dataset.id2word_dict[i])))
#------------distributed_training----------------#
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler
paddle.set_device('gpu')
fleet.init(is_collective=True)
skip_gram_model = SkipGram(len(text8dataset.word2id_freq), embedding_size)
skip_gram_model = fleet.distributed_model(skip_gram_model)
adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list = skip_gram_model.parameters())
adam = fleet.distributed_optimizer(adam)
train_sampler = DistributedBatchSampler(text8dataset, batch_size, shuffle=True, drop_last=True)
train_loader = DataLoader(text8dataset, batch_sampler=train_sampler, num_workers=4)
for epoch in range(epoch_num): 
    for _,(center_words, target_words, label) in tqdm.tqdm(enumerate(train_loader)):
        print('alkj',center_words,target_words,label)
        #使用fluid.dygraph.to_variable函数，将一个numpy的tensor，转换为飞桨可计算的tensor
        center_words_var = fluid.dygraph.to_variable(center_words)
        target_words_var = fluid.dygraph.to_variable(target_words)
        label_var = fluid.dygraph.to_variable(label)
        pred, loss = skip_gram_model(center_words_var, target_words_var, label_var)
        loss.backward()
        #通过minimize函数，让程序根据loss，完成一步对参数的优化更新
        adam.minimize(loss)
        #使用clear_gradients函数清空模型中的梯度，以便于下一个mini-batch进行更新
        skip_gram_model.clear_gradients()
        best_loss=100.
        no_optim=0
        step += 1
        #每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
        if step % 100 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))
            if loss>=best_loss:
                no_optim+=1
            else: #此时的loss较小
                no_optim=0
                best_loss=loss
            if no_optim>6:
                print(f'early stop at {step} epoch')
                fluid.save_dygraph(skip_gram_model.state_dict(), f'./ddpweight/skip_gram_model{step}')
        #经过10000个mini-batch，打印一次模型对eval_words中的10个词计算的同义词
        #这里我们使用词和词之间的向量点积作为衡量相似度的方法
        #我们只打印了5个最相似的词
        if step % 10000 == 0:
            get_similar_tokens('one', 5, skip_gram_model.embedding.weight)
            get_similar_tokens('she', 5, skip_gram_model.embedding.weight)
            get_similar_tokens('chip', 5, skip_gram_model.embedding.weight)   
    fluid.save_dygraph(skip_gram_model.state_dict(), f'./weight/skip_gram_model{step}')
    print('finish!')