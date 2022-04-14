from models.lr_inference import *
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings(action="ignore")

end_date = datetime(2022,4,14)
start_date = end_date - timedelta(days=40)
start_date, end_date

tickers = {
    "삼성전자":"005930",
    "삼성전기":"009150",
    "대한전선":"001440",
    "LS전선아시아":"229640",
    "한국가스공사":"036460",
    "대아티아이":"045390",
    "진원생명과학":"011000",
    "SK케미칼":"285130"
}

for ticker in tickers:
    action = LR_v1_predict(tickers[ticker], start_date, end_date, threshold=0.98)
    print("#" * 50)
    print(f"{ticker}({tickers[ticker]}) 액션: {action[1]}, 종가: {action[2]}")
    print("#" * 50)