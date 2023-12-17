import scrapy
from SCRAPY_JOB_10.items import ScrapyJob10Item#导入ScrapyJob10Item类
from SCRAPY_JOB_10.middlewares import ScrapyJob10DownloaderMiddleware#导入下载中间件类
import datetime
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver import ActionChains
from lxml import etree

class mySpider(scrapy.spiders.Spider):
    name = "CHENGDIAN" #定义爬虫的名字
    allowed_domains = ["https://yjsjob.uestc.edu.cn/"]#描述网站域名
    start_urls=['https://yjsjob.uestc.edu.cn/coread/more-eminfo.jsp']

    def start_requests(self):
        driver=webdriver.Chrome()
        driver.get(self.start_urls[0])
        for j in range(110):
            page=driver.page_source
            htmllist=etree.HTML(page)
            urllist=[]
            for i in range(1,16):
                urllist.append(htmllist.xpath(f'/html/body/div/section/section/div/div[2]/div/ul/li[{i}]/div[2]/h6/a/@href'))
            if(j==0):
                button = driver.find_element(By.XPATH,'/html/body/div/section/section/div/div[2]/ul/li[6]/a')
            else:
                button = driver.find_element(By.XPATH,'/html/body/div/section/section/div/div[2]/ul/li[8]/a')
            ActionChains(driver).move_to_element(button).click(button).perform()
            time.sleep(1)
            #print(urllist)
            for per in urllist:
                yield scrapy.Request("https://yjsjob.uestc.edu.cn/"+per[0])
    def parse(self, response):
        theme=response.xpath('/html/body/div/section/section[1]/div/div/h1/text()').extract()[0]
        print(theme)
        views=response.xpath('/html/body/div/section/section[1]/div/div/span[2]/text()').extract()[0].split('：')[1]
        print(views)
        word = response.xpath('/html/body/div/section/section[1]/div/div/span[1]/text()').extract()[0]
        year=word.split("年")[0][-4:]
        month=word.split("年")[1].split('月')[0]
        day=word.split("年")[1].split('月')[1].split('日')[0]
        date=datetime.date(int(year),int(month),int(day)).isoformat()
        print(year,month,day)
        item = ScrapyJob10Item()
        if datetime.datetime.strptime(date, '%Y-%m-%d') > datetime.datetime.strptime('2021-9-1', '%Y-%m-%d'):
            item['title'] = theme
            item["date"] = date
            item['views'] = views
            yield item

    # def parse(self, response):
    #     driver=webdriver.Chrome()
    #     for i in range(2):
    #         for i in range(1,3):
    #             theme=response.xpath(f'/html/body/div/section/section/div/div[2]/div/ul/li[1]/div[2]/h6/a/text()').extract()[0]
    #             views=response.xpath(f'/html/body/div/section/section/div/div[2]/div/ul/li[{i}]/div[2]/ul/li[1]/text()').extract()[0].split(':')[1]
    #             word = response.xpath(f'/html/body/div/section/section/div/div[2]/div/ul/li[{i}]/div[2]/ul/li[2]/text()').extract()[0]
    #             year=word.split("年")[0][-4:]
    #             month=word.split("年")[1].split('月')[0]
    #             day=word.split("年")[1].split('月')[1].split('日')[0]
    #             date=datetime.date(int(year),int(month),int(day)).isoformat()
    #             print(theme,date,views)
    #             item = ScrapyJob10Item()
    #             item['title'] = theme
    #             item["date"] = date
    #             item['views'] = views
    #             button = driver.find_element(By.XPATH, '/html/body/div/section/section/div/div[2]/ul/li[6]/a')
    #             ActionChains(driver).move_to_element(button).click(button).perform()
    #             time.sleep(1)
    #             return item