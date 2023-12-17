import json
import jsonlines
import os
import tqdm
class socratic_jsonconvertor():#gsm——socratic format代码
    def __init__(self,mode="train"):
        self.mode=mode
        self.trainpath=f"/mnt/ve_share/liushuai/nndl/FlagAI-master/examples/Aquila/Aquila-chat/grade-school-math/grade_school_math/data/{mode}_socratic.jsonl"
        self.conversations=self.read_file(self.trainpath)
        self.resultspath="/mnt/ve_share/liushuai/nndl/FlagAI-master/examples/Aquila/Aquila-chat/grade-school-math/grade_school_math/data"
        self.resultlist=[]
    def read_file(self,jsonl_file):
        conversations = []
        with jsonlines.open(jsonl_file) as reader:
             for line in reader:
                conversations.append(line)
        return conversations
    def formatsource(self,source):
        results=[]
        results.append(source['question'])
        textlist=source['answer'].split("**")
        for subtext in textlist:
           sublist=subtext.split("\n")
           for textinfo in sublist:
                results.append(textinfo)
        return results
    def convert_json(self):
        for id,text in tqdm.tqdm(enumerate(self.conversations)):
            dict1={"from":"human","value":text['question']}
            dict2={"from":"gpt","value":text['answer']}
            tmplist=[dict1,dict2]
            self.resultlist.append({"id":f"format_socratic_{self.mode}_{id}","conversations":tmplist,"instruction": ""})
    def write_json(self):
        path=os.path.join(self.resultspath,f"format_socratic_{self.mode}.jsonl")
        with jsonlines.open(path, 'w') as writer:
                for result in self.resultlist:
                	writer.write(result)   
class common_jsonconvertor(): #gsm数据format代码
    def __init__(self,mode="train"):
        self.mode=mode
        self.trainpath=f"/mnt/ve_share/liushuai/nndl/FlagAI-master/examples/Aquila/Aquila-chat/grade-school-math/grade_school_math/data/{mode}.jsonl"
        self.conversations=self.read_file(self.trainpath)
        self.resultspath="/mnt/ve_share/liushuai/nndl/FlagAI-master/examples/Aquila/Aquila-chat/grade-school-math/grade_school_math/data"
        self.resultlist=[]
    def read_file(self,jsonl_file):
        conversations = []
        with jsonlines.open(jsonl_file) as reader:
             for line in reader:
                conversations.append(line)
        return conversations
    def formatsource(self,source):
        results=[]
        results.append(source['question'])
        textlist=source['answer']
        for subtext in textlist:
           sublist=subtext.split("\n")
           for textinfo in sublist:
                results.append(textinfo)
        return results
    def convert_json(self):
        for id,text in tqdm.tqdm(enumerate(self.conversations)):
            dict1={"from":"human","value":text['question']}
            dict2={"from":"gpt","value":text['answer']}
            tmplist=[dict1,dict2]
            self.resultlist.append({"id":f"format_{self.mode}_{id}","conversations":tmplist,"instruction": ""})
    def write_json(self):
        path=os.path.join(self.resultspath,f"format_{self.mode}.jsonl")
        with jsonlines.open(path, 'w') as writer:
                for result in self.resultlist:
                	writer.write(result)     
class chain_jsonconvertor(): #chain-of-thought数据format的代码
    def __init__(self,mode="train"):
        self.mode=mode
        self.trainpath=f"/mnt/ve_share/liushuai/nndl/FlagAI-master/examples/Aquila/Aquila-chat/grade-school-math/grade_school_math/data/{mode}_socratic.jsonl"
        self.conversations=self.read_file(self.trainpath)
        self.resultspath="/mnt/ve_share/liushuai/nndl/FlagAI-master/examples/Aquila/Aquila-chat/grade-school-math/grade_school_math/data"
        self.resultlist=[]
    def read_file(self,jsonl_file):
        conversations = []
        with jsonlines.open(jsonl_file) as reader:
             for line in reader:
                conversations.append(line)
        return conversations
    def formatsource(self,source):
        results=[]
        prompt="Answer the following question by reasoning step-by-step."
        results.append(prompt+source['question'])
        textlist=source['answer'].split("**")
        for subtext in textlist:
           sublist=subtext.split("\n")
           for textinfo in sublist:
                results.append(textinfo)
        return results  
    def convert_json(self):
        for id,text in tqdm.tqdm(enumerate(self.conversations)):
            rawtext=self.formatsource(text)
            ans=""
            for i in range(1,len(rawtext)-1):
                 if(i%2==0):
                    ans+=rawtext[i]
            ans+=rawtext[-1]
            dict1={"from":"human","value":rawtext[0]}
            dict2={"from":"gpt","value":ans}
            tmplist=[dict1,dict2]
            self.resultlist.append({"id":f"chain_{self.mode}_{id}","conversations":tmplist,"instruction": ""})
    def write_json(self):
        path=os.path.join(self.resultspath,f"chain_{self.mode}_socratic.jsonl")
        with jsonlines.open(path, 'w') as writer:
                for result in self.resultlist:
                	writer.write(result)       
class ask_ans_socratic_jsonconvertor():#问答式socratic数据集生成
    def __init__(self,mode="train"):
        self.mode=mode
        self.trainpath=f"/mnt/ve_share/liushuai/nndl/FlagAI-master/examples/Aquila/Aquila-chat/grade-school-math/grade_school_math/data/{mode}_socratic.jsonl"
        self.conversations=self.read_file(self.trainpath)
        self.resultspath="/mnt/ve_share/liushuai/nndl/FlagAI-master/examples/Aquila/Aquila-chat/grade-school-math/grade_school_math/data"
        self.resultlist=[]
    def read_file(self,jsonl_file):
        conversations = []
        with jsonlines.open(jsonl_file) as reader:
             for line in reader:
                conversations.append(line)
        return conversations
    def formatsource(self,source):
        results=[]
        results.append(source['question'])
        textlist=source['answer'].split("**")
        for subtext in textlist:
           sublist=subtext.split("\n")
           for textinfo in sublist:
                results.append(textinfo)
        return results
    def convert_json(self):
        for id,text in tqdm.tqdm(enumerate(self.conversations)):
            results=self.formatsource(text)
            tmplist=[]
            for i in range(len(results)):
                tmpdict={}
                if i==0:
                    tmpdict["from"]="human"
                elif i==len(results)-1:
                    tmpdict["from"]="gpt"
                    tmpdict["value"]=results[i].lstrip("#### ")
                    tmplist.append(tmpdict)
                    continue
                elif i%2==0:
                    tmpdict["from"]="gpt"
                elif i%2==1:
                    tmpdict["from"]="human"  
                tmpdict["value"]=results[i]
                tmplist.append(tmpdict)
            self.resultlist.append({"id":f"ask_ans_{self.mode}_socratic_{id}","conversations":tmplist})
    def write_json(self):
        path=os.path.join(self.resultspath,f"ask_ans_{self.mode}_socratic.jsonl")
        with jsonlines.open(path, 'w') as writer:
                for result in self.resultlist:
                	writer.write(result)                                                      
convertor=ask_ans_socratic_jsonconvertor(mode="train")
convertor.convert_json()
convertor.write_json()
convertor2=chain_jsonconvertor(mode="train")
convertor2.convert_json()
convertor2.write_json()
convertor3=common_jsonconvertor(mode="train")
convertor3.convert_json()
convertor3.write_json()
convertor4=socratic_jsonconvertor(mode="train")
convertor4.convert_json()
convertor4.write_json()