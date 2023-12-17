import scrapy
from SCRAPY_JOB_10.items import ScrapyJob10Item#导入ScrapyJob10Item类
from SCRAPY_JOB_10.middlewares import ScrapyJob10DownloaderMiddleware#导入下载中间件类
import time
from datetime import datetime

class mySpider(scrapy.spiders.Spider):
    name = "XIDIAN_JIUYE_ALL" #定义爬虫的名字
    allowed_domains = ["job.xidian.edu.cn"]#描述网站域名
    start_urls = ["https://job.xidian.edu.cn/campus/index?domain=xidian&city=&page=1"]

    xidian_next_page=''#定义1个变量，存储下一页的url
    def parse(self, response): #解析爬取的内容
        item = ScrapyJob10Item()  # 生成在items.py中定义的Item对象,用于接收爬取的数据
        #下一页的url，存在于列表页的class="next"的li里面的a标签的href中；注意是相对路径
        next_page_href = response.css('li[class="next"]>a::attr(href)').extract()
        # 末尾页的url，存在于列表页的class="last"的li里面的a标签的href中；注意是相对路径
        last_page_href = response.css('li[class="last"]>a::attr(href)').extract()
        if next_page_href != last_page_href:  # 不相等，并且不空，说明不是最后一页
            self.xidian_next_page = 'https://job.xidian.edu.cn' + next_page_href[0]
        else:
            self.xidian_next_page = ''
        #从当前列表页中，获取所有帖子的详情url；class="infoList"的ul下面的第1个li中的a标签的href中
        c_page_url_list = response.css('ul[class="infoList"]>li:nth-child(1)>a')
        for job in c_page_url_list:  # 循环打开和解析每个详情页
            #此处，调用下载中间件类中的XIDIAN_driver，进行详情页的打开
            driver = ScrapyJob10DownloaderMiddleware.get_XIDIAN_driver()
            driver.get('https://job.xidian.edu.cn' + job.css('a::attr(href)').extract()[0])
            time.sleep(1)#等待几秒钟
            # class ="name text-primary"的a标签的文本，就是招聘主题；
            #注意下面的语法是selenium的find_element函数的语法；通过.text,获取对象的text；另注意要返回列表给item对象
            item['title'] = [driver.find_element('css selector', 'a[class="name text-primary"]').text]
            # class ="share"的div下面的ul下面的第1个li的文本，就是发布时间
            date_text = driver.find_element('css selector', 'div[class="share"]>ul>li:nth-child(1)').text
            date_text=date_text[date_text.find('：') + 1:] #取“：”后面的日期；西电样例：“发布时间：2021-12-03 13:21”
            if datetime.strptime(date_text,'%Y-%m-%d %H:%M')<datetime.strptime('2021-9-1 00:00','%Y-%m-%d %H:%M'):
                #当读取出来的日期已经比设定的还早，说明已经不需要读取了，直接退出循环
                self.xidian_next_page=''#退出前给该处置空，不用再继续执行其他列表页了
                break#测试时候设定时间为'2021-12-03'，正式执行时候，此处时间按要求设置为'2021-09-01'
            item['date'] = [date_text]  #赋值为列表，传递给item
            # class ="share"的div的ul的第2个li中，就是浏览次数：“浏览次数：6”
            views_text = driver.find_element('css selector', 'div[class="share"]>ul>li:nth-child(2)').text
            item['views'] = [views_text[views_text.find('：') + 1:]]  # 先获取文本，再截取“：”后面的次数
            yield item#提交给pipelines
        #处理完列表页的所有二级页面后，继续打开下一页进行抓取
        if self.xidian_next_page!='':
            yield scrapy.Request(self.xidian_next_page, callback=self.parse)
