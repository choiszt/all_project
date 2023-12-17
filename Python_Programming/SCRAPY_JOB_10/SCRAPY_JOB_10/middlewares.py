# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter
from selenium import webdriver###导入selenium模块
import time###导入selenium模块
import scrapy###导入scrapy模块

class ScrapyJob10SpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class ScrapyJob10DownloaderMiddleware:
    @classmethod
    def __init__(cls):###修改该类的init初始化函数
        cls.XIDIAN_driver = webdriver.Chrome()###初始化时候，构造driver对象
        cls.BEIDA_driver = webdriver.Chrome()###初始化时候，构造driver对象
        cls.BUPT_driver = webdriver.Chrome()
    @classmethod
    def __del__(cls):  ###修改该类的删除时候的函数
        cls.XIDIAN_driver.close()  ###关闭driver
        cls.BEIDA_driver.close()  ###关闭driver
        cls.BUPT_driver.close()
    @classmethod
    def get_XIDIAN_driver(cls):
        return cls.XIDIAN_driver
    @classmethod
    def get_BEIDA_driver(cls):
        return cls.BEIDA_driver
    @classmethod
    def get_BUPT_driver(cls):
        return cls.BUPT_driver

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s


    Chrome_options = webdriver.ChromeOptions()
    Chrome_options.headless = True
    chrome = webdriver.Chrome(chrome_options=Chrome_options)

    # def process_request(self, request, spider):
    #     # Called for each request that goes through the downloader
    #     # middleware.
    #
    #     # Must either:
    #     # - return None: continue processing this request
    #     # - or return a Response object
    #     # - or return a Request object
    #     # - or raise IgnoreRequest: process_exception() methods of
    #     #   installed downloader middleware will be called
    #     return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.
        self.chrome.get(request.url)
        time.sleep(1)
        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return scrapy.http.HtmlResponse(url=request.url,body=self.chrome.page_source.encode('utf-8'),request=request,status=200)


    def process_request(self, request, spider):
        if spider.name == "XIDIAN_JIUYE_ALL":
            self.XIDIAN_driver.get(request.url)  ###用driver打开该页面
            time.sleep(1)  ###等候几秒钟比较保险
            return scrapy.http.HtmlResponse(url=request.url, body=self.XIDIAN_driver.page_source.encode('utf-8'),
                                            encoding='utf-8', request=request, status=200)
        elif spider.name == "CHENGDIAN":
            self.BEIDA_driver.get(request.url)  ###用driver打开该页面
            time.sleep(1)  ###等候几秒钟比较保险
            return scrapy.http.HtmlResponse(url=request.url, body=self.BEIDA_driver.page_source.encode('utf-8'),
                                            encoding='utf-8', request=request, status=200)
        elif spider.name == "BUPT":
            self.BUPT_driver.get(request.url)  ###用driver打开该页面
            time.sleep(1)  ###等候几秒钟比较保险
            return scrapy.http.HtmlResponse(url=request.url, body=self.BEIDA_driver.page_source.encode('utf-8'),
                                            encoding='utf-8', request=request, status=200)
    # def process_response(self, request, response, spider):
    #     # Called with the response returned from the downloader.
    #
    #     # Must either;
    #     # - return a Response object
    #     # - return a Request object
    #     # - or raise IgnoreRequest
    #     return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)
