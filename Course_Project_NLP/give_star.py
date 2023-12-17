
import json
import numpy as np
import os
import paddle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


from paddlenlp import Taskflow

class star_evaluator():
    def __init__(self):
        self.candidates={18:"口味",19:"外观",17:"分量",20:"推荐程度",10:"价格水平",11:"性价比",12:"折扣力度",4:"位于商图附近",3:"交通方便",5:"是否容易寻找",6:"排队时间",7:"服务人员态度",8:"停车方便",9:"点菜/上菜速度",13:"装修",14:"嘈杂情况",15:"就餐空间",16:"卫生情况"}
        self.schema =  ["情感倾向[正向,负向,未提及,中性]"]
        self.aspects = ["口味", "外观", "分量","推荐程度","价格水平","性价比","折扣力度","位于商图附近","交通方便","是否容易寻找","排队时间","服务人员态度","停车方便","点菜/上菜速度","装修","嘈杂情况","就餐空间","卫生情况"]
        self.csv_path = "/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/Data/asap/data/train.csv"
        self.df=pd.read_csv(self.csv_path)
        self.df.iloc[:, -19:] = self.df.iloc[:, -19:].replace(-2, 0)
        X = self.df.iloc[:, -18:]
        y = self.df.iloc[:, 2]
        # 创建线性回归模型
        self.model = LinearRegression()
        self.model.fit(X, y)
    def __calscore(self,score):
        finalscore=[]
        for ele in score:
            if(ele!=0):
                finalscore.append(ele)
        if(len(finalscore)==0):
            return 0
        else:
            return sum(finalscore)/len(finalscore)
    def comment2stars(self,text,options):
    # 如果没有指定模型则使用初始模型
        if(options=="uie-senta-base-finetune"):
                model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/base/model_best"
                model="uie-senta-base"
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects,task_path=model_path)
        elif(options=="uie-senta-micro"):
                model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/micro/model_best"
                model="uie-senta-micro"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects,task_path=model_path)
       
        elif(options=="uie-senta-mini-finetune"):               
                model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/mini/model_best"
                model="uie-senta-mini"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects,task_path=model_path)
        elif(options=="uie-senta-medium-finetune"):               
                model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/medium/model_best"
                model="uie-senta-medium"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects,task_path=model_path) 
        
        elif(options=="uie-senta-base"):
                model="uie-senta-base"
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects)
        elif(options=="uie-senta-micro"):
                model="uie-senta-micro"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects)
        elif(options=="uie-senta-mini"):               
                model="uie-senta-mini"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects)
        elif(options=="uie-senta-medium"):               
                model="uie-senta-medium"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects)                 
    # 预测结果
        contents=senta(text)[0]["评价维度"]
        score=[]
        gettext=lambda cont:cont["relations"]["情感倾向[正向,负向,未提及,中性]"][0]['text']
        aspectlist=[]
        for cont in contents:
            try:
                if gettext(cont)=="未提及":
                    score.append(0)
                elif gettext(cont)=="中性":
                    score.append(3)
                elif gettext(cont)=="正向":
                    score.append(5)
                elif gettext(cont)=="负向":
                    score.append(1)
                aspectlist.append(gettext(cont))
            except:
                    aspectlist.append("未提及")
                    pass
        final=self.__calscore(score)
        # 星级限定在1~5之间的0.5整数倍
        if final >= 5.0:
                return 5.0,*aspectlist
        elif final <= 1.0:
                return 1.0,*aspectlist
        else:
                return int(final + 0.5),*aspectlist
    # 读取csv文件
    def comment2stars_linear(self,text,options):
    # 如果没有指定模型则使用初始模型
        if(options=="uie-senta-base-finetune"):
                model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/base/model_best"
                model="uie-senta-base"
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects,task_path=model_path)
        elif(options=="uie-senta-micro-finetune"):
                model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/micro/model_best"
                model="uie-senta-micro"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects,task_path=model_path)
       
        elif(options=="uie-senta-mini-finetune"):               
                model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/mini/model_best"
                model="uie-senta-mini"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects,task_path=model_path)
        elif(options=="uie-senta-medium-finetune"):               
                model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/results/medium/model_best"
                model="uie-senta-medium"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects,task_path=model_path) 
        
        elif(options=="uie-senta-base"):
                model="uie-senta-base"
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects)
        elif(options=="uie-senta-micro"):
                model="uie-senta-micro"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects)
        elif(options=="uie-senta-mini"):               
                model="uie-senta-mini"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects)
        elif(options=="uie-senta-medium"):               
                model="uie-senta-medium"                
                senta = Taskflow("sentiment_analysis", model=model, schema=self.schema,aspects=self.aspects)               
    # 预测结果
        contents=senta(text)[0]["评价维度"]
        gettext=lambda cont:cont["relations"]["情感倾向[正向,负向,未提及,中性]"][0]['text']
        templist=[]
        for i in range(3,21):
                for ele in contents:
                        if ele['text']==self.candidates[i]:
                                try:
                                        if gettext(ele)=="未提及":
                                                templist.append(0)
                                        elif gettext(ele)=="中性":
                                                templist.append(0)
                                        elif gettext(ele)=="正向":
                                                templist.append(1)
                                        elif gettext(ele)=="负向":
                                                templist.append(-1)
                                        break
                                except:
                                        templist.append(0)
                                        
        score=self.model.predict(np.array(templist).reshape(1,-1))
        if score >= 5.0:
                return 5.0
        elif score <= 1.0:
                return 1.0
        else: 
                return int(score + 0.5)
        # return score[0]
