
import json
import numpy as np
import os

from paddlenlp import Taskflow
from staring_linear import linear_model


schema =  ["情感倾向[正向,负向,未提及,中性]"]
aspects = ["口味", "外观", "分量","推荐程度","价格水平","性价比","折扣力度","位于商图附近","交通方便","是否容易寻找","排队时间","服务人员态度","停车方便","点菜/上菜速度","装修","嘈杂情况","就餐空间","卫生情况"]
modelpath="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/model_6"
output_path = '/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/zx_workplace/output'
text="头回去，不清楚状况，11点半到只有一个地方还空着，被吧台小哥拦下说没地方，想着是等他给个号，结果人家忙着对个服务员狂骂，好几桌人都在看，素质好差…最后我说那里有个空位，小哥很不爽的让我去问临桌…只想说海鲜年糕火锅很差，我们都没选择团购，店里说这款特价78，比团购多一点料，除了海虹和花蛤，基本就是在捞年糕和白菜…图册里的芝士只是图册。还不如火炉火。服务员很差，问几个问题都不知道，来回来去的等他去问，点了活章鱼和面包蟹，都快吃完了一问居然没下过单…如果喜欢吃海鲜，这里相对便宜，但千万别点年糕锅。"

def comment2stars(text, calc_mode="simple", output_path=None, model="uie-senta-base",model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/liushuai_workdir/model_5"):
    """
    :param text: text to be stared
    :param calc_mode: "simple" means calculate with average, "Linear" means use Linear Regression model. default "simple"
    :param output_path: print senta output, default None
    :param model: model uses, default uie-senta-base
    :param model_path: model path, which is a folder
    """
    # 如果没有指定模型则使用初始模型
    if model_path == None:
        senta = Taskflow("sentiment_analysis", model=model, schema=schema,aspects=aspects)
    else:
        senta = Taskflow("sentiment_analysis", model=model, schema=schema,aspects=aspects,task_path=modelpath)
    # 预测结果
    a=senta(text)
    # print(a)
    # 如果指定了输出路径则输出到json文件夹中
    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(a, f, indent=4, ensure_ascii=False)

    senta_result = a[0]["评价维度"]

    sentiments = {"正向":1,"负向":-1,"未提及":-2,"中性":0}
    # 获取情感倾向列表
    res = np.array([sentiments[dim["relations"]["情感倾向[正向,负向,未提及,中性]"][0]["text"]] for dim in a[0]["评价维度"]])
    # 使用情感极性评价星级
    if calc_mode == "simple":
        valid = [np.count_nonzero(res == j-1) for j in list(range(3))]
        # 负面评价代表1分，中性为3分，正面评价为5分。计算平均数
        score = int((1 * valid[0] + 3 * valid[1] + 5 * valid[2]) * 2 / (valid[0] + valid[1] + valid[2]))/2
    else:

        score = 1.0
    return score

if __name__ == "__main__":
    print("Final stars:{}".format(comment2stars(text)))
    comment2stars(text, calc_mode="simple", output_path=None, model="uie-senta-base",model_path="/mnt/ve_share/liushuai/PaddleNLP/applications/