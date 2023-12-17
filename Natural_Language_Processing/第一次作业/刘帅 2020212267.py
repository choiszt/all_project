import re
import collections
import logging
import tqdm
logging.basicConfig(filename='mylog.log', level=logging.INFO) #log日志文件初始化


######################
#前处理
######################
with open('mylog.log', 'w'): #对之前的log先清空
    pass
def get_subwords(data):
    #统计子词以及对应的词频
    subwords = collections.defaultdict(int)
    for word, freq in data.items():
        for subword in word.split():
            subwords[subword] += freq
    return subwords
def init_traindata(path):
    with open(path,'r')as f:
        lines=f.readlines() #readline将所有数据读进来
        # for i in range(1000):
        #     lines.append(f.readline())
    train_data={}
    for data in lines: #针对每行句子
        for i in re.split(r"[,。、《》：“”()\[\]【】]", data): #提取出该行的子句
            if i not in train_data.keys():
                train_data[i]=0
            else:
                train_data[i]+=1
    return train_data

def get_testdata(path):
    with open(path,'r')as f:
        lines=f.readlines()
        # lines=[f.readline()]
        # for i in range(10):
        #     lines.append(f.readline())
    results=[]
    for data in lines:
        results.append(data.replace(' ',''))
    return results

trainpath="train_BPE.txt" #数据集路径
testpath="test_BPE.txt"

train_data=init_traindata(trainpath) #训练集预处理
######################
#训练初始化
######################
subwords = get_subwords(train_data)
# 获取初始化的子词词表
bpe_vocab = set(subwords.keys())
print("初始词表个数为",len(bpe_vocab))
# print("词表：", bpe_vocab)

######################
#构建词表
######################
def get_pair_with_frequency(data):
    """
    获取子词对以及子词集合
    """
    pairs = collections.defaultdict(int)
    for word, freq in data.items():
        sub_words = word.split() 
        for i in range(len(sub_words)-1):
            pair = (sub_words[i],sub_words[i+1])#每两个相邻的词组成pair
            pairs[pair] += freq#结果存入字典
    return pairs


def merge_data_with_pair(pair, data):
    """
    将语料中的最高频子词对进行合并
    输入：
        - pair: 最高频子词词对
        - data: 字典形式，统计好的输入语料
    """
    result = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') #pair中的两元素进行合并 例如("社会","主义"),并且两者前后不能有其他穿插（保证连续）
    for word in data:
        merged_word = p.sub(''.join(pair), word)#将word中匹配到的子字符串替换为pair中的两个元素合并后的字符串
        result[merged_word] = data[word]
    return result

def build_vocab(train_data, num_merges,size_bpe):

    # 初始化词表
    subwords = get_subwords(train_data)
    bpe_vocab = set(subwords.keys())
    # print(bpe_vocab, len(bpe_vocab))
    i = 1
    # 逐步生成词表
    for _ in tqdm.tqdm(range(num_merges),mininterval=10): #每十个轮次进行一次可视化
        # 根据语料统计相邻子词对的词频
        pairs = get_pair_with_frequency(train_data)
        # 取频率最大的子词对, 如果pairs 为空或子词对的最大频次为1，则停止
        if not pairs: #如果pairs为空
            break
        best_pair = max(pairs, key=pairs.get)
        if pairs[best_pair] == 1:
            break
        # 合并语料
        train_data = merge_data_with_pair(best_pair, train_data)
        # 将子词加入词表中
        merged_word = "".join(best_pair)
        bpe_vocab.add(merged_word)
        # 删除子词
        subwords = get_subwords(train_data)
        if best_pair[0] not in subwords:
            bpe_vocab.remove(best_pair[0])
        if best_pair[1] not in subwords:#若两词相同，第一次删除后在subwords中则不存在对应的子词
            if best_pair[1] in bpe_vocab: #防止例如(瑟、瑟)的pair 
                bpe_vocab.remove(best_pair[1])
        i += 1
        if i%10==0:
            tqdm.tqdm.write(f"第{i}个iter：当前词表词数:{len(bpe_vocab)}")
            tqdm.tqdm.write(f"第{i}个iter：最高频子词对{best_pair}")
            logging.info(f"第{i}个iter：当前词表词数:{len(bpe_vocab)}") #在日志中进行记录
            logging.info(f"第{i}个iter：最高频子词对{best_pair}")
        if(len(bpe_vocab)==size_bpe):#设置最终停止频次，词典大小为10000时停止
            print("finish!")
            break
    return bpe_vocab

num_merges = 100000 #设置循环次数，可以稍微设大一点，在构建成10000个子词后自动停止

######################
#训练后处理
######################

logging.shutdown()
bpe_vocab = build_vocab(train_data, num_merges,size_bpe=10000)
temp=bpe_vocab#记录出真正属于中文词汇的bpe表格，而不是将标点等特殊符号算入
bpe_vocab.add(',')
bpe_vocab.add('。')
bpe_vocab.add('、')
bpe_vocab.add('《')
bpe_vocab.add('》')
bpe_vocab.add('：')
bpe_vocab.add('“')
bpe_vocab.add('”')
bpe_vocab.add('(')
bpe_vocab.add(')')
bpe_vocab.add('[')
bpe_vocab.add(']')
bpe_vocab.add('【')
bpe_vocab.add('】')
bpe_vocab.add('\n')

######################
#验证
######################

def tokenize_word(word, sorted_vocab, unknown_token='<unk>'):
    """
    输入:
        - word: 待编码的单词
        - sorted_vocab: 排序后的子词词典
        - unknown_token: 不能被切分的子词替代符
    """
    # 如果传入的词为空
    if word == "":
        return []
    # 如果词表为空，则将输入的词替换为<UNK>
    if sorted_vocab == []:
        return [unknown_token]

    word_tokens = []
    # 遍历词表拆分单词
    for i in range(len(sorted_vocab)):
        token = sorted_vocab[i] #从词表中选取第i个词
        # 基于该token定义正则，同时将token里面包含句号的变成[.]
        token_reg = re.escape(token.replace('.', '[.]'))
        # 在当前word中进行遍历，找到匹配的token的起始和结束位置
        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, word)]
        # 如果当前token没有匹配到相应串，则跳过
        if  len(matched_positions) == 0:
            continue
        
        # 获取匹配到的子串的起始位置
        end_positions = [matched_position[0] for matched_position in matched_positions]
        start_position = 0

        for end_position in end_positions:
            subword = word[start_position: end_position]
            word_tokens += tokenize_word(subword, sorted_vocab[i+1:], unknown_token)
            word_tokens += [token]
            start_position = end_position + len(token)
        # 匹配剩余的子串
        word_tokens += tokenize_word(word[start_position:], sorted_vocab[i+1:], unknown_token)
        break
    else:
        # 如果word没有被匹配，则映射为<unk>
        word_tokens = [unknown_token] * len(word)
    
    return word_tokens

def tokenize(text, bpe_vocab):

    sorted_vocab = sorted(bpe_vocab, key=lambda subword: len(subword), reverse=True) #根据子词长度进行sort
    # print("待编码语句: ", text)
    tokens = []
    for word in text:
        # word=word.replace(' ','')
        # word = word + "</w>"
        word_tokens = tokenize_word(word, sorted_vocab, unknown_token='<unk>')
        tokens.extend(word_tokens)

    return tokens


######################
#验证初始化及处理
######################
test_data=get_testdata(testpath)#验证集初始化
tokens = tokenize(test_data, bpe_vocab) #验证集推理


######################
#验证后处理
######################

with open('test_result.txt', 'w') as file:
    pass#清空文件
for i in tokens:
    with open('test_result.txt','a+')as f:
        if(i=='\n'):
            f.write('\n')
        else:
            f.write(i)
            f.write(' ')
with open('bpe_vocab.txt','w',encoding='utf-8')as f:
    for i in temp:
        f.write(i)
        f.write('\n')
