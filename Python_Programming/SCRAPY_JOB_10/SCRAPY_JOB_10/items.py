# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
class ScrapyJob10Item(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field()  # 招聘帖子的主题
    date = scrapy.Field()  # 招聘帖子的发帖时间
    views = scrapy.Field()  # 招聘帖子的浏览次数
class BUPTItem(scrapy.Item):
    # define the fields for your item here like:
    title = scrapy.Field()  # 招聘帖子的主题
    date = scrapy.Field()  # 招聘帖子的发帖时间
    views = scrapy.Field()  # 招聘帖子的浏览次数
    quantity=scrapy.Field()  #职位数量
