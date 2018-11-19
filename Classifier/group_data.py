import numpy as np
import pandas as pd


def data():
	df = pd.read_csv('D:\\MS Data Science Files\\Thesis\\classified_data.csv')
	df_summed = df.groupby(['ticker', 'date']).agg({'NEG': 'sum', 'POS':'sum', 'UNC':'sum'}).reset_index()
	df_mean = df.groupby(['ticker', 'date']).agg({'NEG': 'mean', 'POS':'mean', 'UNC':'mean'}).reset_index()
	df_mean = df_mean.rename(columns= {'NEG' :'mean_neg', 'POS': 'mean_pos', 'UNC':'mean_unc'})
	df_summed = df_summed.rename(columns={'NEG' : 'sum_neg', 'POS': 'sum_pos', 'UNC':'sum_unc'})
	df_merged = pd.merge(df_summed, df_mean, how= 'left', left_on=['ticker','date'], right_on=['ticker','date'])
	return df_merged



def group_main(df1, df2):
	df_report_date = pd.merge(df1, df2, how='left', left_on=['date', 'ticker'],right_on=['report_day','ticker'])
	df_next_date = pd.merge(df1, df2, how= 'left', left_on= ['date','ticker'], right_on=['next_day','ticker'])
	df_joined = df_report_date.append(df_next_date)
	df_clean = df_joined.dropna()
	return df_clean

# df1 = data()

# print(len(data))
earn_df = pd.read_csv('D:\\MS Data Science Files\\Thesis\\earnings_data.csv')