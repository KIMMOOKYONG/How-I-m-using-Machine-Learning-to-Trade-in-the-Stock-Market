import requests
import json
import time
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
import datetime as dt
import math

from tqdm import tqdm

#----------------------------
# 네이버 금융에서 주가 데이터 다운로드 받기
# ticker: 종목코드
# start: 시작일
# end: 종료일
#----------------------------
def s_download(ticker, start, end):
    time.sleep(0.2)
    url = f"https://fchart.stock.naver.com/siseJson.nhn?symbol={ticker}&requestType=1&startTime={start}&endTime={end}&timeframe=day"
    result = requests.post(url)

    data1 = result.text.replace("'",  '"').strip()
    data1 = json.loads(data1)

    data2 = DataFrame(data1[1:], columns=data1[0])
    data2 = data2.reset_index()
    data2["날짜"] = pd.to_datetime(data2["날짜"])

    df = data2[["날짜","시가", "고가", "저가", "종가", "거래량"]].copy()
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df = df.set_index("date")
    df = df.dropna()
    df.loc[:,["open", "high", "low", "close", "volume"]] = df.loc[:,["open", "high", "low", "close", "volume"]].astype(int)
    df = df.loc[:,["open", "high", "low", "close", "volume"]]

    return df


if __name__ == "__main__":

    load_data_path = os.getcwd()
    load_data_path = os.path.join(load_data_path, "stock_utils/KOSPI50.csv")

    save_data_path = os.getcwd()
    save_data_path = os.path.join(save_data_path, "datasets")

    data = pd.read_csv(load_data_path, encoding="euc-kr", dtype="str")

    start = 0
    #-1을 하는 이유는 배열이 0부터 시작하기 때문
    end = data.shape[0] - 1 
    pbar = tqdm(desc = "Progress", total = end)
    while start <= end:
        df = s_download(data["종목코드"][start], "20100101", "20211231")
        filename = str(data["종목코드"][start])
        target_name = os.path.join(save_data_path, filename)
        df.to_csv(target_name)
        start = start + 1
        pbar.update(1)
