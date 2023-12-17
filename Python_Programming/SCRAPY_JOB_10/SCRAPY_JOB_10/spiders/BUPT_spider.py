import scrapy
from SCRAPY_JOB_10.items import BUPTItem
from SCRAPY_JOB_10.middlewares import ScrapyJob10DownloaderMiddleware#导入下载中间件类
import datetime
import time
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver import ActionChains
from lxml import etree

class mySpider(scrapy.spiders.Spider):
    name = "BUPT"
    allowed_domains = ["job.bupt.edu.cn"]
    start_urls=['https://job.bupt.edu.cn/frontpage/bupt/html/recruitmentinfoList.html?type=1']
    def start_requests(self):
        driver=webdriver.Chrome()
        driver.get(self.start_urls[0])
        for j in range(85):
            page=driver.page_source
            parse=etree.HTML(page)
            urllist=[]
            for i in range(1,16):
                urllist.append(parse.xpath(f'//*[@id="listPlace"]/div[{i}]/div[2]/a/@href'))
            if(j<=1):
                button = driver.find_element(By.XPATH,'/html/body/div[1]/div[4]/div[2]/ul/li[8]/a')
            elif(j==2):
                button = driver.find_element(By.XPATH,'/html/body/div[1]/div[4]/div[2]/ul/li[9]/a')
            elif(j>2):
                button =driver.find_element(By.XPATH,'/html/body/div[1]/div[4]/div[2]/ul/li[10]/a/em')
            ActionChains(driver).move_to_element(button).click(button).perform()

            time.sleep(1)
            print(urllist)
            for per in urllist:
                yield scrapy.Request(per[0])

    def parse(self, response):
        # name = response.xpath('/html/body/div[1]/div[2]/div[2]/div[1]/div[1]/text()').extract()
        # info = response.xpath('/html/body/div[1]/div[3]/div/text()').extract()
        # print(len(info))
        # date = info[0].split('\xa0', -1)[1][4:14]
        # hot = info[0].split('\xa0', -1)[2][7:-4]
        theme=response.xpath('/html/body/div[1]/div[2]/div[2]/div[1]/div[1]/text()').extract()[0]
        date=response.xpath('/html/body/div[1]/div[3]/div/text()').extract()[0].split('浏'[0])[0].rstrip()[-10:]
        views=response.xpath('/html/body/div[1]/div[3]/div/text()').extract()[0].split('：')[3].rstrip()
        quantity = response.xpath('/html/body/div[1]/div[4]/div/div[1]/div[2]/table/tbody/tr[2]/td[3]/text()').extract()
        item = BUPTItem()
        try:
            if len(quantity)==0:
                item['quantity']=1
            else:
                item['quantity']=quantity[0]
        except IndexError:
            item['quantity'] = 1
        if datetime.datetime.strptime(date, '%Y-%m-%d') > datetime.datetime.strptime('2021-9-1', '%Y-%m-%d'):
            item['title'] = theme
            item["date"] = date
            item['views'] = views
            yield item
