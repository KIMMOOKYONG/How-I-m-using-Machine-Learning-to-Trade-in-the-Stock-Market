# Trading-strategy-
Here I test a simple ML trading strategy

국내 환경에 받도록 일부 소스코드를 변경할 예정이다.

# 실행 순서(colab에서)
!git clone https://github.com/KIMMOOKYONG/How-I-m-using-Machine-Learning-to-Trade-in-the-Stock-Market.git  
cd How-I-m-using-Machine-Learning-to-Trade-in-the-Stock-Market  
!python ./models/lr_run_training.py  
!python backtester.py  


# 소스코드 삭제 방법(colab에서)
cd ..  
rm -rf ./How-I-m-using-Machine-Learning-to-Trade-in-the-Stock-Market  


# 매매전략
- ML모델 활용 매수 관점인지 예측
- 단기 저점이면 매수
- 일정 비율 상승하면 매도
- 일정 비율 하락하면 손절
- 단일 종목 매수
- 매도 관련 두개의 하이퍼 파라미터 존재(사용자 설정해야함)
- binary classification problem with two targeted outputs.
- local minimums(category 0, green dots)
- local maximums(category 1, red dots)
- determine the inputs of the model.
- 입력 값으로 주가와 거래량 사용(주가의 방향성 예측하기에는 너무 작은 정보)
- 4개의 추가 입력 파라미터 생성
- 입력01: normalized stock price(OHLC를 단일 변수로 변환, 0 ~ 1)

![normalized value calculation](https://miro.medium.com/max/356/1*EeIdzaCOAYph6d1QA4orkQ.png)


## Please note
- I have commented out the paths in the code. If you use code you'd have to correct them accordingly. 
- You will need to create a TDA developer account (https://developer.tdameritrade.com/apis) and get the API key in order to download the data. I have left this blank in the code where I used to download data. 

The blog article with results --> (https://medium.com/analytics-vidhya/how-im-using-machine-learning-to-trade-in-the-stock-market-3ba981a2ffc2)

Some results - 

Predicted buying opportunities

![1_2rdETRDZHCg4NqnIaXDWDA](https://user-images.githubusercontent.com/85404022/156928802-094d90c1-8ae6-491c-aea3-0672c3b032de.png)

Model comparison with the S&P 500 gains in 2021

![1_ZjoDR8c7fDGqMKXyRIbfpw](https://user-images.githubusercontent.com/85404022/156928807-9e94e6b8-0ce4-4ea1-8aad-bf1771ca1b3b.png)
