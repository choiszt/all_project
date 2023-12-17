## Readme

### 0-环境配置

#### 项目运行环境

```
nvcc: NVIDIA (R) Cuda compiler driverCopyright (c) 2005-2021 NVIDIA CorporationBuilt on Sun Mar 21 19:15:46 PDT 2021Cuda compilation tools, release 11.3，V11.3.58Build cuda 11.3.r11.3/compiler.29745058 0

paddle version: 2.4.2
paddlenlp version: 2.5.2.post
torch version: 1.10.1
```

由于UIE项目是基于PaddleNLP框架进行构建的，因此我们需要配置paddlepaddle及paddlenlp的框架。

![img](https://wraxk4dmop.feishu.cn/space/api/box/stream/download/asynccode/?code=YzdiZmQxZTE5Nzc0NjIwMjVlNzhmNWNmYTUwMjE3ZjJfUlJFOUxtdmRwdTNzcUZBU2w3NDVXM1R0MENQMzN5c3ZfVG9rZW46QWd6OWJSaGY4b2hocDR4Skw0aWNDd0NHbmtmXzE2ODgyOTY5NzQ6MTY4ODMwMDU3NF9WNA)

查看/usr/local/目录，包含11.x三个cuda版本，首先在paddle官网安装对应版本的paddlepaddle框架

```Bash
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

安装paddlenlp框架

```Bash
pip install --upgrade paddlenlp>=2.0.0rc -i https://pypi.org/simple
```

或采用python程序的安装方法，首先拉取paddleNLP的项目库

```Bash
git@github.com:PaddlePaddle/PaddleNLP.git
cd PaddleNLP
```

之后执行`python setup.py`进行paddleNLP依赖的安装。

1. 首先从Hugging Face Hub 上下载chatglm-6b所需的config等文件

```Bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b
```

1. 从https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/下载模型参数文件
2. 文件组织形式如下

![img](https://wraxk4dmop.feishu.cn/space/api/box/stream/download/asynccode/?code=YjYxMWM3NGNiYWVjOGJlNmMwMTM5NmIzMDI4NmQ1M2Nfa0xKSHdNUmpsUFM2alNJN0tXS01FU1pWeEJmTmJScnNfVG9rZW46RzNENWI3c1dob3BlQk94MFdqSGNDSFZtbmtBXzE2ODgyOTY5ODg6MTY4ODMwMDU4OF9WNA)

1. 调用如下代码检查ChatGLM-6B 模型是否能够成功生成推理。

```Bash
import os
import gradio as gr
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoTokenizer, AutoModel

class Chatbot():
    def __init__(self,path):
        self.path=path
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
 Chatbot()
```

### 1-文件说明

```python
chat_logs #chatbot的一些历史记录和rawdata,可供chatbot进行训练
chatbot   #关于chatbot微调、评估的相关代码
distill   #蒸馏的相关代码文件
results   #UIE预训练模型
app.py    #利用gradio的前端代码，整合了实现的所有功能
utils.py #微调中一些方法和toolbox
csv2json.py #统一训练格式
evaluate.py #测试集评估代码
finetune.py #uie微调代码
give_star.py #用于gradio前端的评分类
gpt.py  #用于gradio前端的gpt初始化类
senta2score.py  #评论生成分数
staring_linear.py#利用机器学习方法进行评分
staring_simple.py#利用简单aspect加权进行评分的实现
```

