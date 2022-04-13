import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import seaborn as sns

#모델 백테스트 결과를 저장한다.
class create_stats:

    def __init__(self, model_name, threshold, hold_till):
        """
        @model_name: 모델명
        @threshold: 임계치
        @hold_till: 보유 일수
        """        
        self.model = model_name
        self.threshold = threshold
        self.hold_till = hold_till
        
        #분석결과 저장 폴더 설정
        results_dir = os.getcwd()
        results_dir = os.path.join(results_dir, 'results')
        #results_dir = #add results directory
        self.folder_name = f'{str(self.model)}_{self.threshold}_{self.hold_till}'
        self.folder_dir = os.path.join(results_dir, self.folder_name)

        history_df_path = os.path.join(self.folder_dir, 'history_df.csv')
        self.history_df = pd.read_csv(history_df_path)
        self.history_df['buy_date'] = pd.to_datetime(self.history_df['buy_date'])
        self.history_df['sell_date'] = pd.to_datetime(self.history_df['sell_date'])
        
        params_path = os.path.join(self.folder_dir, 'params')
        with open(params_path, 'rb') as fp:
            self.params = pickle.load(fp)
        
        results_summary_path = os.path.join(self.folder_dir, 'results_summary')
        with open(results_summary_path, 'rb') as fp:
            self.results_summary = pickle.load(fp)
        
        #get params from stored files
        self.initial_capital = self.results_summary[0]
        self.total_gain = self.results_summary[1]
        self.start_date = self.params[4]
        self.end_date = self.params[5]

        self.calculate_stats()
        self.save_stats()
    
    def calculate_stats(self):

        #calculate total percentage win
        self.total_percentage = np.round(self.total_gain/self.initial_capital * 100, 2)
        #total gains
        self.total_gains = np.round(self.history_df[self.history_df['net_gain'] > 0]['net_gain'].sum(), 2)
        self.maximum_gain = np.round(self.history_df[self.history_df['net_gain'] > 0]['net_gain'].max(), 2)
        #total losses
        self.total_losses = np.round(self.history_df[self.history_df['net_gain'] < 0]['net_gain'].sum(), 2)
        self.maximum_loss = np.round(self.history_df[self.history_df['net_gain'] < 0]['net_gain'].min())
    
    def save_stats(self):

        #분석결과 저장 폴더 설정
        results_dir = os.getcwd()
        results_dir = os.path.join(results_dir, 'results')

        df = pd.read_csv(results_dir + "/model_result_summary.csv")

        results_dict = {'Model': f'{self.model}_{self.threshold}_{self.hold_till}',\
            'Gains': self.total_gains,
            'Losses': self.total_losses,
            'Profit': np.round(self.total_gain, 2),
            'Profit Percentage': self.total_percentage,
            'Maximum Gain': self.maximum_gain,
            'Maximum Loss': self.maximum_loss}
        df = df.append(results_dict, ignore_index = True)
        df.to_csv(results_dir + "/model_result_summary.csv")        

if __name__ == "__main__":
    # 모델명: LR_v1_predict
    # 임계값: 1
    # 보유일수: 1
    cs = create_stats('LR_v1_predict', 1, 1)