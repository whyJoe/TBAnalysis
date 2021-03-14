import itchat
import time
import re
import jieba
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
from wordcloud import WordCloud, ImageColorGenerator

itchat.auto_login(hotReload=True)
friends = itchat.get_friends(True)
fs_list = []
obj = {}
for friend in friends:
    obj['UserName'] = friend['UserName']
    obj['NickName'] = friend['NickName']
    obj['AttrStatus'] = friend['AttrStatus']
    print(friend)

# SINCERE_WISH = u'祝%s新年快乐！'
#
# friendList = itchat.get_friends(update=True)[1:]
# for friend in friendList:
#     print(friend['Sex'])
# time.sleep(.5)

#
# uu = itchat.search_friends('谭志豪')
# print(uu)
#
# for i in open('a.txt','r',encoding='utf-8'):
#     if i == "\n":
#         continue
#     else:
#         time.sleep(1)
#         itchat.send(msg=i, toUserName=aa)
# itchat.run(True)
# print(itchat.get_friends())
# 22aa19596d644a3fa2b15d95ac5b198c





