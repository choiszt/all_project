import paddle
import pandas as pd
import numpy as np

# 读取csv文件
csv_path = "/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/Data/asap/data/train.csv"
data = pd.read_csv(csv_path)

# 将最后19维数据中的-2都转为0
# data.iloc[:, -19:] = data.iloc[:, -19:].replace(-2, 0)

X_list = data.iloc[:, -18:].values
y_list = data.iloc[:, -19].values

for i in range(10):
    # valid0 = np.count_nonzero(X_list[i] == 1)
    # valid1 = np.count_nonzero(X_list[i] == -1)
    # valid = valid0 + valid1
    valid = []
    for j in range(3):
        valid.append(np.count_nonzero(X_list[i] == j-1))
    valid = int((1 * valid[0] + 3 * valid[1] + 5 * valid[2]) * 2 / (valid[0] + valid[1] + valid[2]))/2
    print(i, X_list[i], valid ,y_list[i], sep='\t')