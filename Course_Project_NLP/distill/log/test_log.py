import logging

# # 创建logger对象
# logger = logging.getLogger('mylogger')
# logger.setLevel(logging.DEBUG)

# # 创建文件处理器
# file_handler = logging.FileHandler('mylog.log')
# file_handler.setLevel(logging.DEBUG)

# # 创建格式化器
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# # 将文件处理器添加到logger
# logger.addHandler(file_handler)

# # 记录日志信息
# logger.debug('debug message')
# logger.info('info message')
# logger.warning('warning message')
# logger.error('error message')
# logger.critical('critical message')

distill_logger = logging.getLogger("Distill_logger")
distill_logger.setLevel(logging.DEBUG)
# 创建文件处理器
file_handler = logging.FileHandler('./distill.log')
file_handler.setLevel(logging.DEBUG)
# 创建格式化器
formatter = logging.Formatter('[%(asctime)s] %(name)s [%(levelname)s]: %(message)s')
file_handler.setFormatter(formatter)
# 将文件处理器添加到logger
distill_logger.addHandler(file_handler)


def record():
    for i in range(10):
        distill_logger.info("Number {}".format(i))

record()