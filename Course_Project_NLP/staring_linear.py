import paddle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 读取csv文件
csv_path = "/mnt/ve_share/liushuai/PaddleNLP/applications/sentiment_analysis/unified_sentiment_extraction/Data/asap/data/train.csv"
df = pd.read_csv(csv_path)

# 将最后19维数据中的-2都转为0
# df.iloc[:, -19:] = df.iloc[:, -19:].replace(-2, 0)

# print(data.head())
# 选择 X 和 y
X = df.iloc[:, -18:]
y = df.iloc[:, 2]

# 创建线性回归模型
linear_model = LinearRegression()

# 拟合模型
linear_model.fit(X, y)

# 打印模型参数
print('模型系数:', linear_model.coef_)
print('模型截距:', linear_model.intercept_)

if __name__ == "__main__":
    for i in range(1,10):
        Xi = X.iloc[i].values.reshape(1, -1)
        print(X.iloc[i].values.reshape(1, -1))
        # 后面这部分要加上，需要对y的取值进行修正
        pred_y = int(linear_model.predict(Xi) * 2 + 1)/2
        if pred_y < 1:
            pred_y = 1
        elif pred_y > 5:
            pred_y = 5
        print(i, y.iloc[i], pred_y)

    # with open('./output/linear_regression_model.pkl', 'wb') as f:
    #     pickle.dump(linear_model, f)