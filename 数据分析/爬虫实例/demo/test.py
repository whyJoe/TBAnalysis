# -*-coding:utf-8-*-
"""
Created on 2019/11/21 16:41
@author: joe
"""
from selenium import webdriver
import time
browser = webdriver.Chrome()
browser.get("https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_idx=1&tn=21002492_6_hao_pg&wd=%E7%BD%91%E6%98%93%E9%82%AE%E7%AE%B1&rsv_pq=839fb36f000452fe&rsv_t=b381vjyIN9eenGpjwrU5sUwXa5P6%2FfGyAZrc1c72l1O9r%2FKfqzm1l%2B%2FWXBzoT9yO6f8Hxu4W0fo&rqlang=cn&rsv_enter=1&rsv_dl=ib&rsv_sug3=4&rsv_sug1=2&rsv_sug7=100&sug=%25E7%25BD%2591%25E6%2598%2593%25E9%2582%25AE%25E7%25AE%25B1&rsv_n=1")
browser.find_element_by_xpath('//*[@id="op_email3_username"]').send_keys('liuzhfor@163.com')
browser.find_element_by_xpath('//*[@id="1"]/div[1]/div/form/table/tbody/tr[2]/td[2]/span/input').send_keys('wyy88888888')
browser.find_element_by_xpath('//*[@id="1"]/div[1]/div/form/table/tbody/tr[3]/td[2]/a[1]').click()
