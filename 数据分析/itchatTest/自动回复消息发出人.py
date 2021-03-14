# -*-coding:utf-8-*-
import itchat
import re
import time

re.compile('')


def reply_msg():
    users = itchat.search_friends("江兆尧")
    userName = users[0]['UserName']
    print(userName)
    for i in range(10):
        itchat.send('第'+str(i+1)+'次说唉', toUserName=userName)
        time.sleep(1)


if __name__ == '__main__':
    itchat.auto_login(True)
    reply_msg()
    itchat.run()