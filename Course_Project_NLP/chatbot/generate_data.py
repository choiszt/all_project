import csv
import json
data_file="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/Data/asap/data/train.csv"
requirements="生成一段关于餐厅的评价，要包括口味、外观、分量、推荐程度、价格水平、性价比、折扣力度、位于商图附近、交通方便、是否容易寻找、排队时间、服务人员态度、停车方便、点菜/上菜速度、装修、嘈杂情况、就餐空间、卫生情况其中的一个或多个方面"
jsonlist=[]
with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
                tempdict={"conversation":requirements}
                tempdict.update({"response":row[1]})
                jsonlist.append(tempdict.copy())
                tempdict.clear()
with open("/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/chatbot/train.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(jsonlist, ensure_ascii=False)+"\n")