import numpy as np
with open("./output1.txt")as f :
    datas=f.readlines()
    originlist=[]
    pridict=[]
    for data in datas:
        data=data.strip().split('\t')
        originlist.append(float(data[-2]))
        pridict.append(float(data[-1]))
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(originlist, bins=50, alpha=0.5, label='origin')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Origin Data')
ax1.legend()

ax2.hist(pridict, bins=50, alpha=0.5, label='pridict')
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Histogram of Predicted Data')
ax2.legend()
cosine_similarity = np.dot(originlist, pridict) / (np.linalg.norm(originlist) * np.linalg.norm(pridict))
plt.suptitle(f'The cosine similarity of Origin and Predicted Data is {cosine_similarity}')
# 计算两个列表的相似度
plt.savefig("./trash.jpg")