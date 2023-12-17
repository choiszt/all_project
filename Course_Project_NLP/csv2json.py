import os
import csv
import numpy as np
import tqdm
import json
import random


candidates={18:"口味",19:"外观",17:"分量",20:"推荐程度",10:"价格水平",11:"性价比",12:"折扣力度",4:"位于商图附近",3:"交通方便",5:"是否容易寻找",6:"排队时间",7:"服务人员态度",8:"停车方便",9:"点菜/上菜速度",13:"装修",14:"嘈杂情况",15:"就餐空间",16:"卫生情况"}


class ASPADataset():
    def __init__(self, data_file):
        super().__init__()
        self.data = []
        self.labels = []
        self.scores = []
        self.results={}

        #  json数据集保存路径
        self.result_path = "/mnt/ve_share/liushuai/PaddleNLP-develop/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/"
        self.data_name = data_file.split("/")[-1].split(".")[0]+".json"
        self.save_json_path=os.path.join(self.result_path, self.data_name)
        self.options=["正向","中性","负向","未提及"]
        self.hash={'1':"正向",'0':"中性",'-1':"负向",'-2':"未提及"}
        self.example=[]

        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            # 获取标题
            self.labels = header
            for row in reader:
                data_item = dict()
                data_item['content'] = row[1]
                for i in range(3,21):
                    data_item[candidates[i]]=row[i]
                # 具体评论内容
                self.results[row[0]]=data_item
                for k in range(3,21):
                    prompt=candidates[k]+"的情感倾向"
                    example=self.generate_cls_example(row[1],self.hash[row[k]],prompt)
                    self.example.append(example)
        random.shuffle(self.example)
        jsonpath="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/Data/json/train.json"
        for exam in tqdm.tqdm(self.example):
            self.writejson(exam,jsonpath)
    def generate_cls_example(self, text, label, prompt_prefix):
        random.shuffle(self.options)
        cls_options = ",".join(self.options)
        prompt = prompt_prefix + "[" + cls_options + "]" #(ls)口味纯正的情感倾向
        result_list = []
        example = {"content": text, "result_list": result_list, "prompt": prompt}
        start = prompt.rfind(label) - len(prompt) - 1
        end = start + len(label)
        result = {"text": label, "start": start, "end": end}
        example["result_list"].append(result)
        return example
    def splitdata(self,prot,data):
        random.shuffle(data)
        trainlen=int(len(data)*prot[0])
        devlen=int(len(data)*prot[1])
        traindata=data[:trainlen]
        devdata=data[trainlen:]
        return traindata,devdata
    def writejson(self,example,jsonpath):
        with open(jsonpath, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    data_dir = "/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/Data/asap/data"
    train_csv_file = os.path.join(data_dir, "train.csv")
    train_data = ASPADataset(train_csv_file)
        