
import numpy as np
import csv
import os
import pandas as pd
from pandas.testing import assert_frame_equal
 
 


# File50 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_50000.csv',index_col=0)
# File100 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_100000.csv',index_col=0)
# File150 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_150000.csv',index_col=0)
# File200 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_200000.csv',index_col=0)
# File250 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_250000.csv',index_col=0)
# File300 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_300000.csv',index_col=0)
# File350 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_350000.csv',index_col=0)
# File400 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_400000.csv',index_col=0)
# File450 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_450000.csv',index_col=0)
# File500 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_500000.csv',index_col=0)
# File550 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_550000.csv',index_col=0)
# File600 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_600000.csv',index_col=0)
# File630 = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_632740.csv',index_col=0)
# File = pd.concat([File50,File100,File150,File200,File250,File300,File350,File400,File450,File500,File550,File600,File630],axis=1)
# File.to_csv("train_all.csv")
# print('finished')

df = pd.read_csv('/home/zhouzhenyu/cond_adver/zhou/Vggish/train_all.csv',index_col=0)
print(df.columns)
print(type(df.columns))
print(len(df.columns))