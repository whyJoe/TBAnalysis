# -*-coding:utf-8-*-
import itchat
from apscheduler.schedulers.blocking import BlockingScheduler
import time


def send_msg():
    user_info = itchat.search_friends(name=u'史泽豪')
    print(user_info)
    if len(user_info) > 0:
        user_name = user_info[0]['UserName']
        for i in range(10):
            time.sleep(1)
            itchat.send('11111', toUserName=user_name)


def after_login():
    scheduler.add_job(send_msg, 'date', run_date='2019-7-23 10:14:00')
    scheduler.start()


def after_logout():
    scheduler.shutdown()


if __name__ == '__main__':
    scheduler = BlockingScheduler()
    itchat.auto_login(hotReload=True,loginCallback=after_login, exitCallback=after_logout)
    itchat.run()
