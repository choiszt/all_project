import csv
import pandas as pd
data=pd.read_csv("/SCRAPY_JOB_10/BEIYOU_1.csv")
newtitle=data['title'].str.strip()
newdate=data['date'].str.strip()
data['title']=newtitle
data['date']=newdate

data=pd.read_csv("/SCRAPY_JOB_10/CHENGDIAN_1.csv")
newtitle=data['title'].str.strip()
newdate=data['date'].str.strip()
data['title']=newtitle
data['date']=newdate

data=pd.read_csv("/SCRAPY_JOB_10/XIDIAN_1.csv")
newtitle=data['title'].str.strip()
newdate=data['date'].str.strip()
data['title']=newtitle
data['date']=newdate