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

import logging
import log_utils.logger_init
logger = logging.getLogger("__LR_training__")

class LR_training:

    def __init__(self, model_version, tickers = "tickers.csv", threshold = 0.98, start_date = None, end_date = None):

        """
        @model_version: 버전
        @tickers: 학습 데이터 종목 코드 파일
        @threshold: 임계값
        @start_date: 시작일
        @end_date: 종료일
        
        # ------------------------------------        
        tickers.csv 파일 구조
        # ------------------------------------
        tickers
        종목코드들
        """
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
        
        # stocks 변수에 학습에 사용할 종목 코드 목록을 저장한다.
        # dow: 학습 할 임의 종목을 설정한다.
        # tickers: 학습에 사용할 종목 정보를 저장한 파일
        dow = []
        tickers = pd.read_csv('tickers.csv', dtype=str)
        tickers = list(tickers['ticker'])
        stocks = dow + tickers[:20] # 상위 20개 종목
        self.stocks = list(np.unique(stocks))
        logger.info(f"학습할 종목코드: {self.stocks}")

        # main dataframe
        # 학습 데이터 저장 데이터프레임
        self.main_df = pd.DataFrame(columns = ["volume", "normalized_value", "3_reg", "5_reg", "10_reg", "20_reg", "target"])

        # init models
        self.scaler = MinMaxScaler() # 정규화
        self.lr = LogisticRegression() # 로지스틱 회귀 모델

        # run logistic regresion
        # 학습 진행 순서 정의
#         self.fetch_data()
#         self.create_train_test()
#         self.fit_model()
#         self.confusion_matrix()
#         self.save_model()

        current_dir = os.getcwd()
        logger.info(f"current_dir: {current_dir}")
        print("aaaa")

    """
    주가정보를 수집해서, 모델 학습용 데이터를 생성하는 함수.
    OLHC 가격 정규화 및 라벨링 작업 수행
    라베링 작업은 green dot (category 0) or a red dot (category 1)인지 여부를 target 칼럼에 라벨링
    """
    def fetch_data(self):
        """
        get train and test data
        """ 
        for stock in self.stocks:
            try: 
                # 함수호출경로: create_train_data() - > get_data() - > normalized_values()                
                # 종목별 특성 정보를 제거하고 학습용 데이터만 생성한다.
                # 종목의 갯수는 의미가 없다.
                df = stock_utils.create_train_data(stock, n = 10)
                self.main_df = self.main_df.append(df)
            except:
                logger.warning("fetch_data(self) 함수 오류 발생")
                pass
            
        logger.info(f"{len(self.main_df)} samples were fetched from the database..")
        
        # 디버깅을 위해서 생성 데이터 덤프함.
        self.main_df.to_csv("dump_main_df.csv")

    """
    학습 데이터를 훈련용 데이터와 테스트 데이터로 분활
    """
    def create_train_test(self):
        
        """
        create train and test data
        학습 데이터에서 무작위 샘플링을 통해서 데이터 추출
        """        
        self.main_df = self.main_df.sample(frac = 1, random_state = 3). reset_index(drop = True)
        """
        데이터프레임으로 부터 무작위(확률, 임의) 표본 추출하는 방법
        특정 개수의 표본 무작위 추출(number)
        특정 비율의 표본 무작위로 추출(fraction)
        복원 무작위 표본 추출(random sampling with replacement)
        가중치를 부여하여 표본 추출(weights)
        칼럼에 대해 무작위 표본 추출(axis=1, axis='column')
        특정 칼럼에 대해 무작위 표본 추출한 결과를 numpy array로 할당
        """        
        
        self.main_df['target'] = self.main_df['target'].astype('category')        
        """
        카테고리형(Categorical) 데이터는 데이터프레임의 칼럼에서 특정한 형태의 데이터가 반복되는 경우 사용
        예를 들어 성별(남성, 여성), 나이(10대, 20대, ...)와 같이 특정 구간의 데이터가 반복되는 경우
        카테고리형 데이터를 이용하면 반복된 데이터를 코드화하여 데이터의 사이즈를 줄여서 메모리 사용량이 줄어 들고 데이터 처리 속도가 빨라집니다.        
        """        
        # pop 메서드를 사용하면 해당 칼럼 데이터를 추출하고, 소스 데이터프레임에서 해당 칼럼 삭제
        # x, y 2차원 데이터로 변환
        y = self.main_df.pop('target').to_numpy()
        y = y.reshape(y.shape[0], 1)
        # 스케일링 적용
        x = self.scaler.fit_transform(self.main_df)

        # test train split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, \
            test_size = 0.05, random_state = 50, shuffle = True)

        print('Created test and train data...')

    def fit_model(self):

        print('Training model...')
        # self.lr.fit(self.train_x, self.train_y)
        # ravel() 함수는 다차원 1배열을 1차원 배열로 변환해주는 함수.
        self.lr.fit(self.train_x, self.train_y.ravel())
        
        # predict the test data
        self.predictions = self.lr.predict(self.test_x)
        self.score = self.lr.score(self.test_x, self.test_y)
        print(f'Logistic regression model score: {self.score}')

        # preds with threshold
        # category 0, 1의 분류에 대한 각 분류에 속할 확률값
        self.predictions_proba = self.lr._predict_proba_lr(self.test_x)
        # 아래 코드는 뭐지 ???
        # 가중치 조정을 통한 분류 결과 조정??
        self.predictions_proba_thresholded = self._threshold(self.predictions_proba, self.threshold) # 0.98
    
    # 학습 결과 cm으로 시각화 데이터 생성
    def confusion_matrix(self):
        cm = confusion_matrix(self.test_y, self.predictions)
        self.cmd = ConfusionMatrixDisplay(cm)
        # self.cmd.plot()
        
        cm_thresholded = confusion_matrix(self.test_y, self.predictions_proba_thresholded)
        self.cmd_thresholded = ConfusionMatrixDisplay(cm_thresholded)
        # self.cmd_thresholded.plot()

    # 분류 결과에 대한 가중치 조정        
    def _threshold(self, predictions, threshold):

        prob_thresholded = [0 if x > threshold else 1 for x in predictions[:, 0]]
        return np.array(prob_thresholded)

    # 모델 저장
    def save_model(self):

        # save models
        # 모델 객체를 pickled binary file 형태로 저장한다.
        saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
        model_file = f'lr_{self.model_version}.sav'
        model_dir = os.path.join(saved_models_dir, model_file)
        pickle.dump(self.lr, open(model_dir, 'wb'))

        # Scaler객체 pickled binary file 형태로 저장한다.
        scaler_file = f'scaler_{self.model_version}.sav'
        scaler_dir = os.path.join(saved_models_dir, scaler_file)
        pickle.dump(self.scaler, open(scaler_dir, 'wb'))

        print(f'Saved the model and scaler in {saved_models_dir}')
        cm_path = os.path.join(os.getcwd(), 'results/Confusion Matrices')
        
        # save cms
        # confusion matricx 저장
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
    
