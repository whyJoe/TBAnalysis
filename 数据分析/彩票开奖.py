# -*-coding:utf-8-*-
"""
Created on 2019/12/9 16:33
@author: joe
"""
import random
import time

red = []
blue = []
buy_red = []
buy_blue = []


def isRepeat(array, num):
    for i in array:
        if i == num:
            return False
    return True


# 用于开奖
def LottoRun():
    count = 1
    while (count <= 6):
        num = random.randint(1, 33)
        if isRepeat(red, num):
            red.append(num)
            count = count + 1
    blue.append(random.randint(1, 16))
    print('开奖结果为 : ' + str(red) + str(blue))


# 电脑随机买
def buyLotto():
    count = 1
    while (count <= 6):
        num = random.randint(1, 33)
        if isRepeat(buy_red, num):
            buy_red.append(num)
            count = count + 1
    buy_blue.append(random.randint(1, 16))
    print('投注结果为 : ' + str(buy_red) + str(buy_blue))
    return buy_red, buy_blue


def win(red, blue, buy_red, buy_blue, num):
    list.sort(red)
    list.sort(buy_red)
    for i in range(0, 6):
        if red[i] != buy_red[i]:
            return False
    if blue[0] != buy_blue[0]:
        return False
    return True


# num = 1
# while (True):
#     print('-' * 5 + '第' + str(num) + '期' + '-' * 5)
#     LottoRun()
#     for i in range(0, 5):
#         buyLotto()
#         if win(red, blue, buy_red, buy_blue, num):
#             print('第' + str(num) + '期中了')
#             break
#         buy_red, buy_blue = [], []
#     red, blue = [], []
#     num = num + 1

initMoney = 10000
money = 100
a = 1
while (initMoney > 0):
    count = 1
    while (count <= 3):
        num = random.randint(0, 9)
        if isRepeat(red, num):
            red.append(num)
            count = count + 1
    result = red[0] + red[1] + red[2]
    go = ['单', '双']
    touzhu = go[random.randint(0, 1)]
    print('买了'+str(money)+'元' + touzhu)
    if result % 2 == 0:
        sj = '双'
    else:
        sj = '单'
    print('第' + str(a) + '期 : ' + str(red[0]) + ' + ' + str(red[1]) \
          + ' + ' + str(red[2]) + ' = ' + str(result) + ' ' + sj)

    if touzhu == sj:
        initMoney = initMoney + money
        money = 100
    else:
        initMoney = initMoney - money
        money = money * 2
    print('*' * 5 + '还剩' + str(initMoney) + '元' + '*' * 5)
    red = []
    sj = ''
    a = a + 1
    # time.sleep(2)
