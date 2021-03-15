# -*-coding:utf-8-*-
"""
Created on 2021/3/15 18:39
@author: joe
"""

from tkinter import filedialog  # 路径选择
from tkinter import *
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

root = Tk()  # 创建tkinter的主窗口
root.title("客户消费行为分析系统")
root.geometry('1000x500')

path = StringVar()


def getpath():
    path_ = tk.filedialog.askopenfilename()
    path.set(path_)


# 标签
Label(root, text='目录选择:', width=10).grid(row=0)

tk.Entry(root, textvariable=path, width=50).grid(row=0, column=1, columnspan=3)

btn1 = Button(root, text='选择文件', command=getpath).grid(row=0, column=4)

btn2 = Button(root, text='读取数据', command=getpath).grid(row=1, column=0, padx=5)
Text(root, width=70, height=30).grid(row=2, column=0, columnspan=5, rowspan=12, padx=5,pady=5)

# 用户访问量分析
Label(root, text='用户访问量分析', width=30).grid(row=0, column=6, columnspan=3, padx=10)
Button(root, text='日pv，人均pv，uv对比图', command=getpath, width=30).grid(row=1, column=6, padx=10)
Button(root, text='日访问量分析', command=getpath, width=30).grid(row=1, column=7, padx=10)
Button(root, text='时访问量分析', command=getpath, width=30).grid(row=2, column=6, padx=10)
Button(root, text='“日新增uv”和“日新增uv的pv”的分析', command=getpath, width=30).grid(row=2, column=7, padx=10)

# 不同行为类型用户pv分析
Label(root, text='不同行为类型用户pv分析', width=20).grid(row=3, column=6, columnspan=3)
Button(root, text='四种行为的日行为分析', command=getpath, width=30).grid(row=4, column=6, padx=10)
Button(root, text='四种行为的时行为分析', command=getpath, width=30).grid(row=4, column=7, padx=10)

# 用户购买情况分析
Label(root, text='用户购买情况分析', width=20).grid(row=5, column=6, columnspan=3, padx=10)
Button(root, text='时购买行为分析', command=getpath, width=30).grid(row=6, column=6, padx=10)
Button(root, text='日购买行为分析', command=getpath, width=30).grid(row=6, column=7, padx=10)

# 用户复购行为分析
Label(root, text='用户复购行为分析', width=20).grid(row=7, column=6, columnspan=3, padx=10)
Button(root, text='用户每一周的复购率与回购率', command=getpath, width=30).grid(row=8, column=6, padx=10)
#
# 用户访问量分析
Label(root, text='用户行为与商品种类关系分析', width=30).grid(row=9, column=6, columnspan=3, padx=10)
Button(root, text='不同用户行为类型的转化率', command=getpath, width=30).grid(row=10, column=6, padx=10)
Button(root, text='不同用户行为类型对各类商品的行为分析', command=getpath, width=30).grid(row=10, column=7, padx=10)

# 用户访问量分析
Label(root, text='漏斗模型与RFM模型分析', width=20).grid(row=11, column=6, columnspan=3, padx=10)
Button(root, text='漏斗模型分析', command=getpath, width=30).grid(row=12, column=6, padx=10)
Button(root, text='RFM模型分析', command=getpath, width=30).grid(row=12, column=7, padx=10)

# 主循环
root.mainloop()
