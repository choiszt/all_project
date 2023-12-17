from itemadapter import ItemAdapter
import csv
class ScrapyJob10Pipeline:#定义本项目的pipeline处理类
    def process_item(self, item, spider):#对于爬取到的数据进行处理
        try:
            dict_item = dict(item)  #把抓取数据生成列表对象
            if spider.name=='XIDIAN_JIUYE_ALL':
                self.XIDIAN_writer.writerow(dict_item)  #将数据写入到文件中
            elif spider.name=='CHENGDIAN':
                self.BEIDA_writer.writerow(dict_item)  # 将数据写入到文件中
            elif spider.name == 'BUPT':
                self.BUPT_writer.writerow(dict_item)  # 将数据写入到文件中
            return item
        except Exception as err:
            print(err)
    def open_spider(self, spider):
        #爬虫开启时候的动作，写模式打开json文件，并设置为爬虫类的
        self.XIDIAN_file =open('XIDIAN_1.csv', 'w+', newline='', encoding='utf-8')
        self.XIDIAN_writer = csv.DictWriter(self.XIDIAN_file,fieldnames=['title','date','views'])
        self.XIDIAN_writer.writeheader()#写入表头
        self.BEIDA_file = open('CHENGDIAN_1.csv', 'w+', newline='', encoding='utf-8')
        self.BEIDA_writer = csv.DictWriter(self.BEIDA_file, fieldnames=['title', 'date', 'views'])
        self.BEIDA_writer.writeheader()  # 写入表头
        self.BUPT_file = open('BEIYOU_1.csv', 'w+', newline='', encoding='utf-8')
        self.BUPT_writer = csv.DictWriter(self.BUPT_file, fieldnames=['title', 'date', 'views','quantity'])
        self.BUPT_writer.writeheader()  # 写入表头
    def close_spider(self, spider):#爬虫关闭时候的动作，关闭文件
        self.XIDIAN_file.close() #关闭文件
        self.BEIDA_file.close()  # 关闭文件
        # self.BUPT_file.close()


