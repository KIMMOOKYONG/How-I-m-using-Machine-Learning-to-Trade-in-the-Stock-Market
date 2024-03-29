"""
stock backtester to test the model given a dataset. 
author - Kaneel Senevirathne
date - 1/13/2022
"""

# doctest는 기본적으로 unittest, pytest처럼 테스를 위한 모듈이다.
from doctest import OutputChecker
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from stock_utils.stock_utils import timestamp, create_train_data, get_data, create_test_data_lr, get_stock_price
from datetime import timedelta
import time

import logging
import log_utils.logger_init
logger = logging.getLogger("__LR_training__")

def load_LR(model_version):

    file =  f"./saved_models/lr_{model_version}.sav"
    logger.debug(f"로딩 모델 경로: {file}")
    
    loaded_model = pickle.load(open(file, 'rb'))

    return loaded_model

def load_scaler(model_version):

    file = f"./saved_models/scaler_{model_version}.sav"
    logger.debug(f"로딩 스케일러 경로: {file}")
    
    loaded_model = pickle.load(open(file, 'rb'))

    return loaded_model

def _threshold(probs, threshold):
    """
    Inputs the probability and returns 1 or 0 based on the threshold
    """
    # probs[:, 0]: 분류가 0일 확률 값
    # 0일 확률값이 threshold 값 보다 크면, 0(매수), 아니면 1(매도)
    prob_thresholded = [0 if x > threshold else 1 for x in probs[:, 0]]

    logger.debug("_threshold() 호출")
    logger.debug(f"파라미터: probs({probs}, threshold({threshold}))")
    logger.debug(f"prob_thresholded 값: {prob_thresholded}")
    
    return np.array(prob_thresholded)


"""
매수, 매도를 예측하고자는 날짜(종료일)를 기준으로 과거 데이터(-40일, 시작일) 파라미터로 전달하면
기준 날짜의 매수, 매도를 예측하는 함수이다.
"""
def LR_v1_predict(stock, start_date, end_date, model = 'v2', threshold = 0.98):
    """
    this function predict given the data
    주가 데이터를 기반으로 고가, 저가 예측
    시작일과 종료일 기간에 해당하는 학습용 데이터를 생성
    """
    # create model and scaler instances 
    # 모델, 스케일러 로딩
    # @model: 모델 버전
    scaler = load_scaler(model)
    lr = load_LR(model)
    
    # create input
    # 백테스팅 데이터 생성
    # 라벨링 없음    
    data = create_test_data_lr(stock, start_date, end_date)
    
    # get close price of final date
    # 테스트 데이터의 마지막 날 종가
    close_price = data['close'].values[-1]
    logger.debug(f"close_price: {close_price}")
    
    # get input data to model
    # 모델에 입력할 데이터 생성
    input_data = data[['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']]
    # 1D 데이터를 2D 데이터로 변환(모델 입력값이 2차원이므로)
    # 마지막 데이터 추출 및 1D를 2D로 변환
    input_data = input_data.to_numpy()[-1].reshape(1, -1)
    
    # scale input data
    # 마지막 날 데이터 스케일링, 정규화
    input_data_scaled = scaler.transform(input_data)
    # 예측값
    prediction = lr._predict_proba_lr(input_data_scaled)
    # 가중치 적용 예측값
    prediction_thresholded = _threshold(prediction, threshold)
   
    # 예측값, 가중치 적용 예측값, 종가
    return prediction[:, 0], prediction_thresholded[0], close_price

"""
매도 
"""
def LR_v1_sell(stock, buy_date, buy_price, todays_date, sell_perc = 0.1, hold_till = 3, stop_perc = 0.05):
    """
    gets stock price. recommnd to sell if the stock price is higher sell_perc * buy_price + buy_price
    stock - stock ticker symbol
    buy_date - the date the stock was bought
    todays_date - date today
    sell_perc - sell percentage 
    hold_till - how many days to hold from today
    """
    current_price = get_stock_price(stock, todays_date) # current stock value, 현재가
    sell_price = buy_price + buy_price * sell_perc # 매도 가격
    stop_price = buy_price - buy_price * stop_perc # 손절 가격
    sell_date = buy_date + timedelta(days = hold_till) # the day to sell 
    time.sleep(1) #to make sure the requested transactions per seconds is not exeeded.
    #some times it returns current price as none
    if (current_price is not None) and ((current_price < stop_price) or (current_price >= sell_price) or (todays_date >= sell_date)):
        return "SELL", current_price # if criteria is met recommend to sell
    else:
        return "HOLD", current_price # if crieteria is not met hold the stock





