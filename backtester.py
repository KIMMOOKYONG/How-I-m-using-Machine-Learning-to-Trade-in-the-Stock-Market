"""
stock backtester to test the model given a dataset. 
author - Kaneel Senevirathne
date - 1/13/2022
"""

import numpy as np
from stock_utils.simulator import simulator
from stock_utils.stock_utils import get_stock_price
from models import lr_inference
from datetime import datetime
from datetime import timedelta
# from td.client import TDClient
import pandas as pd
from models.lr_inference import LR_v1_predict, LR_v1_sell
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
import pickle
from tqdm import tqdm

import logging
import log_utils.logger_init
logger = logging.getLogger("__bt_logger__")

class backtester(simulator):
    #model: 학습 모델을 실행할 함수명을 파라미터로 전달
    #lr_inference 파일에 저장된 함수를 호출
    def __init__(self, stocks_list, model, model_version, capital, start_date, end_date, threshold = 0.99, sell_perc = 0.04, hold_till = 5,\
         stop_perc = 0.005):
        
        super().__init__(capital) #initialize simulator

        self.stocks = stocks_list #종목 목록
        self.model = model #모델명
        self.model_version = model_version #모델 버전
        self.start_date = start_date #시작일
        self.day = start_date #시작일, 매매일
        self.end_date = end_date #종료일
        self.status = 'buy' #the status says if the backtester is in buy mode or sell mode
        self.threshold = threshold
        self.sell_perc = sell_perc #상승 비율
        self.hold_till = hold_till #보유 기간
        self.stop_perc = stop_perc #손절 비율

        logger.debug(f"금번 실행 파라미터:")
        logger.debug(f"종목목록:{self.stocks}")
        logger.debug(f"모델명: {self.model}")
        logger.debug(f"모델버전: {self.model_version}")
        logger.debug(f"시작일: {self.start_date}")
        logger.debug(f"종료일: {self.end_date}")
        logger.debug(f"실행모드: {self.status}")
        logger.debug(f"임계값: {self.threshold}")
        logger.debug(f"상승비율: {self.sell_perc}")
        logger.debug(f"보유기간: {self.hold_till}")
        logger.debug(f"손절비율: {self.stop_perc}")

        #백테스팅 결과 저장 폴더 설정 및 폴더 생성
        #current directory
        current_dir = os.getcwd()
        results_dir = os.path.join(current_dir, "results")
        folder_name = f"{str(self.model.__name__)}_{self.threshold}_{self.hold_till}"
        
        self.folder_dir = os.path.join(results_dir, folder_name)
        logger.debug(f"백테스팅 모델 경로: {self.folder_dir}")        
        if not os.path.exists(self.folder_dir):
            #create a new folder
            os.makedirs(self.folder_dir)

    def backtest(self):
        """
        start backtesting
        """
        #1일 날짜값
        delta = timedelta(days = 1)
        
        #progress bar to track progress
        #시작일과 종료일의 기간 일수
        total_days = (self.end_date - self.start_date).days
        d = 0

        #진행바 생성
        pbar = tqdm(desc = "Progress", total = total_days)

        #시작일에서 종료일까지 루프 실행
        while self.day <= self.end_date:
            
            #daily scanner dict
            self.daily_scanner = {}  
            if self.status == "buy":
                #scan stocks for the day
                self.scanner()
                if list(self.daily_scanner.keys()) != []:
                    recommended_stock = list(self.daily_scanner.keys())[0] #매수 종목
                    recommended_price = list(self.daily_scanner.values())[0][2] #매수 가격
                    self.buy(recommended_stock, recommended_price, self.day) #buy stock
                    logger.info(f"Bought {recommended_stock} for {recommended_price} on the {self.day}")
                    self.status = "sell" #change the status to sell, 매수했으면 매도 모드로 변경한다.
                else:
                    logger.debug('No recommendations')
                    pass
            else: #if the status is sell
                #get stock price on the day
                stocks = [key for key in self.buy_orders.keys()]
                for s in stocks:
                    recommended_action, current_price = LR_v1_sell(s, self.buy_orders[s][3], self.buy_orders[s][0], self.day, \
                        self.sell_perc, self.hold_till, self.stop_perc)
                    if recommended_action == "SELL":
                        logger.info(f'Sold {s} for {current_price} on {self.day}')
                        self.sell(s, current_price, self.buy_orders[s][1], self.day)
                        self.status = 'buy'              
            #go to next day
            self.day += delta
            d += 1
            pbar.update(1)
        pbar.close()
        #sell the final stock and print final capital also print stock history 
        self.print_bag()
        self.print_summary() 
        self.save_results()      
        return

    """
    특정일의 예축값, 임계값 확류 및 종가 반환하는 함수
    """
    def get_stock_data(self, stock, back_to = 40):
        """
        this function queries to td database and get data of a particular stock on a given day back to certain amount of days
        (default is 30). 
        back_to에 40일 설정하는 이유는 regression 연산을 위해서 과거 데이터 필요
        regression 기간을 변경하면 back_to의 값을 재조정해야한다.
        """
        #get start and end dates        
        #(datetime.datetime(2020, 11, 22, 0, 0), datetime.datetime(2021, 1, 1, 0, 0))
        end = self.day #매매일
        start = self.day - timedelta(days = back_to) #과거 40일전 날짜 추출
        
        """
        prediction, prediction_thresholded, close_price = LR_v1_predict(stock, start, end, threshold = 0.5)        
        prediction, prediction_thresholded, close_price = LR_v1_predict(stock, '2020-11-22', '2021-01-01', threshold = 0.5)        
        @prediction: 예측 확률값
        @prediction_thresholded: 임계값 확률값
        @close_price: 당일 종가                 
        """
        prediction, prediction_thresholded, close_price = self.model(stock, start, end, self.model_version, self.threshold)
        
        # 예측값, 가중치 적용 예측값, 종가
        return prediction[0], prediction_thresholded, close_price

    def scanner(self):
        """
        scan the stocks to find good stocks
        """
        for stock in self.stocks:
            #to ignore the stock if no data is available. for staturdays or sundays etc            
            #토,일요일 등 주식시장이 개장 안하는 날
            try:
                prediction, prediction_thresholded, close_price = self.get_stock_data(stock)
                #if prediction greater than
                #매수 여부를 결정한다.
                #임계치 확률값이 1보다 작으면
                if prediction_thresholded < 1: # if prediction is zero
                    # 일별 종목별 예측값, 가중치 적용 예측값, 종가 정보를 저장한다.
                    self.daily_scanner[stock] = (prediction, prediction_thresholded, close_price)
                    logger.info(f"{stock} 코드: {self.daily_scanner[stock]}")
            except Exception as e:
                logger.debug(f"{stock} 코드 scanner() 함수 오류 발생")
                logger.exception(e)

        def take_first(elem):
            return elem[1]

        # OrderedDict 순서를 보장하는 딕셔너리
        self.daily_scanner = OrderedDict(sorted(self.daily_scanner.items(), key = take_first, reverse = True))

    def save_results(self):
        """
        save history dataframe create figures and save
        """
        #save csv file
        results_df_path = os.path.join(self.folder_dir, 'history_df.csv')
        self.history_df.to_csv(results_df_path, index = False)
        
        #save params and results summary
        results_summary_path = os.path.join(self.folder_dir, 'results_summary')
        results_summary = [self.initial_capital, self.total_gain]
        params_path = os.path.join(self.folder_dir, 'params')
        params = [self.threshold, self.hold_till, self.sell_perc, self.stop_perc, self.start_date, self.end_date]
        
        with open(results_summary_path, 'wb') as fp:
            pickle.dump(results_summary, fp)
        with open(params_path, 'wb') as fp:
            pickle.dump(params, fp)

if __name__ == "__main__":
    #stocks list
    dow = ['001440']
    other = []

    #모델버전
    model_version = "v2"
    stocks = list(np.unique(dow + other))
    back = backtester(dow, LR_v1_predict, model_version, 3000, datetime(2021, 12, 1), datetime(2022, 4, 14), threshold = 0.98, sell_perc = 0.03, hold_till = 10,\
        stop_perc = 0.03)
    back.backtest()

    """
    백테스팅을 실행하면
    results/LR_v1_predict_임계값_보유기간 형식의 폴더가 생성되고
    폴더 내부에 아래의 파일이 생성된다.
    history_df.csv(stock,buy_price,n_shares,sell_price,net_gain,buy_date,sell_date)
    params
    results_summary
    """    
