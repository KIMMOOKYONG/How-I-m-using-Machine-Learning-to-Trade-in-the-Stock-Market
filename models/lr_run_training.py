"""
LR training 
author - Kaneel Senevirathne
date - 1/13/2022
"""

# from td.client import TDClient
import requests, time, re, os
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
import datetime
plt.style.use('grayscale')

# Linear algebra functions(선형대수함수)
from scipy import linalg
import math
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime
import os
import sys
import pickle

# append path
# import 문을 통해 다른 파이썬 파일을 불러올 때, 
# 파이썬은 내부적으로 파일을 찾기 위해 sys.path와 PYTHONPATH에 있는 경로를 탐색합니다.
# 현재 경로를 sys.path에 추가한다.
current_dir = os.getcwd()
sys.path.append(current_dir)

from stock_utils import stock_utils
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

class LR_training:

    def __init__(self, model_version, threshold = 0.98, start_date = None, end_date = None):

        self.model_version = model_version
        self.threshold = threshold
        
        # 훈련에 사용할 데이터의 기간을 설정한다.
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        # get stock ticker symbols
        # 학습을 진행할 종목코드를 tickers.csv파일에서 불러온다.
        # 여러개 종목코드를 파라미터로 전달할 경우, 개별 종목에 대한 훈련이 아니고, 여러 종목의 데이터를 가져와서
        # 하나의 학습 데이터를 생성하는 구조인것으로 보인다.
        dow = ['001440']
        tickers = pd.read_csv('tickers.csv', dtype=str)
        tickers = list(tickers['ticker'])
        stocks = dow + tickers[:20]
        self.stocks = list(np.unique(stocks))
        print(f'학습할 종목코드: {self.stocks}')

        # main dataframe
        self.main_df = pd.DataFrame(columns = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target'])

        # init models
        self.scaler = MinMaxScaler()
        self.lr = LogisticRegression()

        # run logistic regresion
        # 학습 진행 순서 정의
        self.fetch_data()
#         self.create_train_test()
#         self.fit_model()
#         self.confusion_matrix()
#         self.save_model()

    def fetch_data(self):
        """
        get train and test data
        """ 
        for stock in self.stocks:
            try: 
                df = stock_utils.create_train_data(stock, n = 10)
                self.main_df = self.main_df.append(df)
            except:
                print('fetch_data(self) 함수 오류 발생')
                pass
        print(f'{len(self.main_df)} samples were fetched from the database..')
        
        # 디버깅을 위해서 데이터를 덤프함.
        self.main_df.to_csv('main_df.csv')

    def create_train_test(self):
        """
        create train and test data
        """
        self.main_df = self.main_df.sample(frac = 1, random_state = 3). reset_index(drop = True)
        self.main_df['target'] = self.main_df['target'].astype('category')
        
        y = self.main_df.pop('target').to_numpy()
        y = y.reshape(y.shape[0], 1)
        x = self.scaler.fit_transform(self.main_df)

        #test train split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, \
            test_size = 0.05, random_state = 50, shuffle = True)

        print('Created test and train data...')

    def fit_model(self):

        print('Training model...')
        self.lr.fit(self.train_x, self.train_y)
        
        #predict the test data
        self.predictions = self.lr.predict(self.test_x)
        self.score = self.lr.score(self.test_x, self.test_y)
        print(f'Logistic regression model score: {self.score}')

        #preds with threshold
        self.predictions_proba = self.lr._predict_proba_lr(self.test_x)
        self.predictions_proba_thresholded = self._threshold(self.predictions_proba, self.threshold)
      
    def confusion_matrix(self):
        cm = confusion_matrix(self.test_y, self.predictions)
        self.cmd = ConfusionMatrixDisplay(cm)
        
        cm_thresholded = confusion_matrix(self.test_y, self.predictions_proba_thresholded)
        self.cmd_thresholded = ConfusionMatrixDisplay(cm_thresholded)

        
    def _threshold(self, predictions, threshold):

        prob_thresholded = [0 if x > threshold else 1 for x in predictions[:, 0]]

        return np.array(prob_thresholded)

    def save_model(self):

        #save models
        saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
        model_file = f'lr_{self.model_version}.sav'
        model_dir = os.path.join(saved_models_dir, model_file)
        pickle.dump(self.lr, open(model_dir, 'wb'))

        scaler_file = f'scaler_{self.model_version}.sav'
        scaler_dir = os.path.join(saved_models_dir, scaler_file)
        pickle.dump(self.scaler, open(scaler_dir, 'wb'))

        print(f'Saved the model and scaler in {saved_models_dir}')
        cm_path = os.path.join(os.getcwd(), 'results/Confusion Matrices')
        
        #save cms
        plt.figure()
        self.cmd.plot()
        plt.savefig(f'{cm_path}/cm_{self.model_version}.jpg')

        plt.figure()
        self.cmd_thresholded.plot()
        plt.savefig(f'{cm_path}/cm_thresholded_{self.model_version}.jpg')
        print(f'Figures saved in {cm_path}')

import argparse

if __name__ == "__main__":
    run_lr = LR_training('v3')
