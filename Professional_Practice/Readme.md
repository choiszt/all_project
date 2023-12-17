## nndl课设

#### 刘帅 202012267 李楷鸿 2020212233

#### 1.数据集介绍

我们提供了几种供于aquila微调的数据集，分别是基于**socratic**和基于**普通范式**的gsm8k数据

| **dataset(gsm8k)**     | **model** | **evaluation** | **comment**                                |
| ---------------------- | --------- | -------------- | ------------------------------------------ |
| format_train           | aquila-7b | 6%             | 原始模型                                   |
| format_train           | aquila-7b | 12%            | 按照flagai提供数据格式的gsm8k数据          |
| format_train           | lora      | 10.6%          | 按照flagai提供数据格式的gsm8k数据          |
| format_socratic_train  | aquila-7b | 8.3%           | 按照flagai提供数据格式的gsm8k-socratic数据 |
| chain_train_socratic   | aquila-7b | 21%            | 加入**chain-of-thought prompt**的数据      |
| ask_ans_train_socratic | aquila-7b | failed         | 加入问答式多轮human-gpt对话的数据          |

#### 2.代码构成

```python
Aquila-chat-lora.yaml #lora训练的相关配置参数
Aquila-chat.yaml# 全量微调的相关配置参数
aquila_chat.py #利用aquila进行微调的代码
aquila_chat_version2.py#优化了dataset和tokenized的过程（仿照openai格式）
bmtrain_mgpu.sh#bmtrain脚本
gsm8k_test.txt#推理结果
hostfile#单机多卡配置
jsonconvertor.py#数据集处理
test.py#测试程序
train.sh#训练脚本
```

#### 3.运行

利用以下命令运行代码

```bash
bash train.sh
```

- 如果想利用优化后的脚本则需在`bmtrain_mgpu.sh`中将`export SCRIPT_FILE=aquila_chat.py`修改为`export SCRIPT_FILE=aquila_chat_version2.py`
- 如果想启动lora训练版本则需要将train.sh中的`Aquila-chat.yaml`修改为`Aquila-chat-lora.yaml`

