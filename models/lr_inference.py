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

def load_LR(model_version):

    file =  f'./saved_models//lr_{model_version}.sav'
    loaded_model = pickle.load(open(file, 'rb'))

    return loaded_model

def load_scaler(model_version):

    file = f'./saved_models//scaler_{model_version}.sav'
    loaded_model = pickle.load(open(file, 'rb'))

    return loaded_model

def _threshold(probs, threshold):
    """
    Inputs the probability and returns 1 or 0 based on the threshold
    """
    prob_thresholded = [0 if x > threshold else 1 for x in probs[:, 0]]

    return np.array(prob_thresholded)


"""

"""
def LR_v1_predict(stock, start_date, end_date, threshold = 0.98):
    """
    this function predict given the data
    주가 데이터를 기반으로 고가, 저가 예측
    """
    # create model and scaler instances 
    # 모델, 스케일러 로딩
    scaler = load_scaler('v3')
    lr = load_LR('v3')
    
    # create input
    # 로지스틱 회귀 테스트 데이터 생성
    # 라벨이 붙어있지 않는 데이터
    data = create_test_data_lr(stock, start_date, end_date)
    
    # get close price of final date
    # 테스트 데이터의 마지막 날 종가
    close_price = data['close'].values[-1]
    
    # get input data to model
    # 모델에 입력할 데이터, 종가 칼럼을 제거
    input_data = data[['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']]
    # 1D 데이터를 2D 데이터로 변환
    # 마지막 날 데이터
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





