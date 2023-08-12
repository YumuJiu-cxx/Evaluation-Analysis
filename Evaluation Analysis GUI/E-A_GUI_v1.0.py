# -*- coding:utf-8 -*-
import tkinter as tk
import pickle
from lib import *
from tensorflow.keras import *

w2v_model_file = ''
log_line_num = 0


class Home():
    def __init__(self, master):
        self.root = master
        self.root.config()
        self.root.title('评价识别.net')
        self.root.geometry('800x500')  # 窗口大小
        self.root.iconbitmap('favicon.ico')  # 网页图标
        self.root['background'] = '#000000'

        window_one(self.root)


class window_one():
    def __init__(self, master):
        self.master = master
        self.master.config(bg='#000000')
        self.frame1 = tk.Frame(self.master, width=800, height=500, bg='#000000')
        self.frame1.pack()

        upload_label = tk.Label(self.frame1, text='输入评价', font=('楷体', 15), fg='#FFFFFF', bg="#000000")
        upload_label.place(relx=0.5, rely=0.2, anchor='center')
        self.comment_text = tk.Text(self.frame1, width=50, height=10, font='楷体', fg='#222222', bg="#999999",
                                    relief='sunken')
        self.comment_text.place(relx=0.5, rely=0.45, anchor='center')

        # 开始按钮
        btn_update = tk.Button(self.frame1, text='开始分析', font=('楷体', 12), fg='#FFFFFF', bg="#222222",
                               command=self.start_analysis)
        btn_update.place(relx=0.5, rely=0.7, anchor='center')

        # 分类框
        self.classify_text = tk.StringVar()
        classify_label = tk.Label(self.frame1, textvariable=self.classify_text, font=('楷体', 14), fg='#FFFFFF',
                                  bg="#000000")
        classify_label.place(relx=0.45, rely=0.8, anchor='center')

        # 评级框
        self.scoring_text = tk.StringVar()
        scoring_label = tk.Label(self.frame1, textvariable=self.scoring_text, font=('楷体', 14), fg='#FFFFFF',
                                 bg="#000000")
        scoring_label.place(relx=0.55, rely=0.8, anchor='center')

    def start_analysis(self):
        """开始分析"""

        classify_model = models.load_model('Model/comment-classify_model.h5')
        scoring_model = models.load_model('Model/comment-scoring_model.h5')

        # 加载tokenizer
        with open('Tokenizer/tokenizer_classify.pickle', 'rb') as handle:
            classify_tokenizer = pickle.load(handle)
        with open('Tokenizer/tokenizer_scoring.pickle', 'rb') as handle:
            scoring_tokenizer = pickle.load(handle)

        # 取得评价
        comment = self.comment_text.get("1.0", "end")[:-1]

        # 分类
        classify_text = predict_classify(comment, classify_tokenizer, classify_model)

        # 评分
        comment_scoring = predict_scoring(comment, scoring_tokenizer, scoring_model)
        if comment_scoring <= 0.2:
            self.scoring_text.set('一星')
        if comment_scoring > 0.2 and comment_scoring <= 0.4:
            self.scoring_text.set('二星')
        if comment_scoring > 0.4 and comment_scoring <= 0.6:
            self.scoring_text.set('三星')
        if comment_scoring > 0.6 and comment_scoring <= 0.8:
            self.scoring_text.set('四星')
        if comment_scoring > 0.8:
            self.scoring_text.set('五星')

        self.classify_text.set(classify_text)


if __name__ == '__main__':
    root = tk.Tk()
    Home(root)
    root.mainloop()
