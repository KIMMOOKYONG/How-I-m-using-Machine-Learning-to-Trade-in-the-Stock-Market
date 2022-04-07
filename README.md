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


# The ML model
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
- 입력01이 1에 가까우면 고가 가깝다.(0에 가까우면 저가)
- 입력01 하나의 변수에 단일 가격의 변동 정보를 모두 담을 수 있다.
- 입력02: 거래량
- 입력03: 03 day regression coefficient(과거 3일간의 종가 데이터를 가지고 선형회귀를 통해서 계산, 3일간의 주가 방향을 나타냄)
- 입력04: 05 day regression coefficient
- 입력05: 10 day regression coefficient
- 입력06: 20 day regression coefficient

![normalized value calculation](https://miro.medium.com/max/356/1*EeIdzaCOAYph6d1QA4orkQ.png)

![](https://miro.medium.com/max/350/1*OMsY8j4udl-g7r1YpjwYcw.png)

# Training and validating the model
- 주가 데이터 수집은 네이버 이용
- 학습 및 검증 데이터는 다우 30 종목 + S&P 500 종목 주가 데이터 사용(2007 ~ 2020)
- 테스트 데이터는 2021년 데이터 사용
- 알고리즘을 통해서 local mins(0) and max(1) points 찾아서 데이터 라벨링
- 다음은 입력 데이터 예시

![](https://miro.medium.com/max/700/1*fqafvwhvO_7owgIiU-GQtw.png)

- Volume, normalized value, 3_reg, 5_reg, 10_reg, 20_reg are the input parameters and the target is the output. If target is 0, the row represents data from a buying point (local minimum) and if the row represents a 1 it is a selling point (local maximum).
- split the data into train & validation sets
- train the LR model
- The LR model used the input parameters and predicted the target value.
- 

# Validation results and analysis
- 모델 성능 검증을 위해서 하나의 종목을 선택한다.(Gs, 골드만삭스)
- 학습된 LR model을 통해서 주가의 방향성을 예측
- Testing results for stock ticker GS. Green dots represent buying points and red points represent selling points predicted by our model.

![](https://miro.medium.com/max/700/1*Enhp6QJ6nVm5XQjcuL7tSQ.png)

- 지역 고가, 저가 데이터를 가지고 모델을 학습시켰기 때문에 일반 가격에 대한 방향성 예측에는 약점이 있다.
- 2021년 테스트 데이터를 입력하고 LR 모델을 통해서 매수 포인트를 대다수 정확하게 예측은 하지만 일부 데이터는 잘못 예측하고 있다.
- Confusion matrix of results from the validation dataset

![](https://miro.medium.com/max/416/1*SPRHtOiSwfOZgyvKsAmk6Q.jpeg)

- Confusion matrix를 활용해서 모델을 보정
- 

## Please note
- I have commented out the paths in the code. If you use code you'd have to correct them accordingly. 
- You will need to create a TDA developer account (https://developer.tdameritrade.com/apis) and get the API key in order to download the data. I have left this blank in the code where I used to download data. 

The blog article with results --> (https://medium.com/analytics-vidhya/how-im-using-machine-learning-to-trade-in-the-stock-market-3ba981a2ffc2)

Some results - 

Predicted buying opportunities

![1_2rdETRDZHCg4NqnIaXDWDA](https://user-images.githubusercontent.com/85404022/156928802-094d90c1-8ae6-491c-aea3-0672c3b032de.png)

Model comparison with the S&P 500 gains in 2021

![1_ZjoDR8c7fDGqMKXyRIbfpw](https://user-images.githubusercontent.com/85404022/156928807-9e94e6b8-0ce4-4ea1-8aad-bf1771ca1b3b.png)
