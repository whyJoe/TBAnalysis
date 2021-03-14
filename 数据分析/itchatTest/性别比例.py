# -*-coding:utf-8-*-
import itchat
itchat.auto_login()

friends = itchat.get_friends(update=True)[0:]
print(friends)
# print(friends)
# 初始化性别的变量
male = female = others = 0
# 循环得到的全部好友
# 在好友的信息中有Sex标签,发现规律是当其值为1是表示男生,2表示女生,0表示没有填写的
for i in friends[1:]:
    sex = i['Sex']
    if (sex == 1):
        male += 1
    elif (sex == 2):
        female += 1
    else:
        others += 1

total = len(friends[2:])
print(total)
print("男生好友比例 : %.2f%%" % (float(male) / total * 100) + "\n"
      "女生好友比例 : %.2f%%" % (float(female) / total * 100) + "\n"
      "其他性别好友比例 : %.2f%%" % (float(others) / total * 100) + "\n")
