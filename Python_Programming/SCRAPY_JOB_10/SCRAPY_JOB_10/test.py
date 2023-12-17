import time

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
driver=webdriver.Chrome()
driver.get('https://job.bupt.edu.cn/frontpage/bupt/html/recruitmentinfoList.html?type=1')
list=driver.find_element(By.XPATH,'//*[@id="listPlace"]')
for j in range(2):
    for i in range(1,3):
        driver.find_element(By.XPATH,f'/html/body/div[4]/div/div[2]/div[2]/table/tbody/tr[{i}]/td[1]/a').click()
        time.sleep(2)
        print(driver.find_element(By.XPATH,'/html/body/div[4]/div/div[2]/div[1]/h3').text)
        time.sleep(2)
        print(driver.find_element(By.XPATH,'/html/body/div[1]/div[3]/div').text.split('浏'[0])[0].rstrip()[-10:])#日期
        print(driver.find_element(By.XPATH, '/html/body/div[1]/div[3]/div').text.split('：')[3])

        driver.back()
        time.sleep(3)
    button=driver.find_element(By.XPATH,'/html/body/div[1]/div[4]/div[2]/ul/li[8]/a')
    ActionChains(driver).move_to_element(button).click(button).perform()
    time.sleep(2)

