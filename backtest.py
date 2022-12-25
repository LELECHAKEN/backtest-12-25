import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bottleneck
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from datetime import datetime
import gc
import os,sys,pdb,glob,math
import warnings
import statsmodels.api as sm

warnings.filterwarnings('ignore')
v = 'factor'
ret = 'return'
start_date = '2014-01-01'
end_date = '2021-01-01'
ifrank = False


data = pd.read_csv('data.csv')
data['return'] = data.groupby('ticker')['last'].apply(lambda x:(x/x.shift(1)-1).shift(-1))
# use volume/sum of volume in the past 20 days as factor 
data['factor'] = data.groupby('ticker')['volume'].apply(lambda x:x/x.rolling(20).sum()).shift(1,axis = 0) 

# remove extreme values
def extreme_process_MAD(x): 
    median = x.median()
    MAD = abs(x - median).median()
    x[x>(median+3*1.4826*MAD)] = median+3*1.4826*MAD
    x[x<(median-3*1.4826*MAD)] = median-3*1.4826*MAD
    return x    

data['factor'] = data.groupby('date')['factor'].apply(extreme_process_MAD)
data[ret] = np.where(abs(data[ret])>0.5, np.nan, data[ret])

# standardize 
train_data = data[(data['date']<start_date)]
test_data = data[(data['date']>=start_date) &  (data['date']<=end_date)]
factor_mean = train_data['factor'].mean()
factor_std = train_data['factor'].std()
test_data.loc[:,['factor']] = (test_data['factor']-factor_mean)/factor_std

df = test_data[['date', 'ticker', ret,v]].copy()
df['long'] = df[ret][df['factor'] > 0] * df['factor'][df['factor'] > 0] 
df['short'] = df[ret][df['factor'] < 0] * df['factor'][df['factor'] < 0] 

df_tmp = df[df['factor'] > 0]
long_sum = df_tmp.groupby(['date'])['factor'].sum() # sum of long factor
y_mean = df.groupby(['date'])[ret].mean() # mean of return of all stocks for each day and mintime
aa = df.groupby(['date'])['long'].sum() # get long return
longret = aa / long_sum - y_mean # return minus avg of market

df_tmp = df[df['factor'] < 0]
short_sum = df_tmp.groupby(['date'])['factor'].sum()
y_mean = df.groupby(['date'])[ret].mean()
aa = df.groupby(['date'])['short'].sum()
shortret = - aa / short_sum + y_mean

overall_ret = longret.add(shortret,fill_value = 0)/2 # all return (market basic included)

tmp_long = longret
tmp_all = overall_ret
#tmp_all.to_csv(v + "_pnl.csv")

pnl_file = pd.DataFrame(tmp_all)
pnl_file = pnl_file.stack()
pnl_file = pnl_file.reset_index()
pnl_file.columns= ['date', 'tme','pnl1']
pnl_file = pnl_file.loc[:,['date','pnl1']]

longNum = df[df['factor'] > 0].groupby(['date'])['factor'].count()
shortNum = df[df['factor'] < 0].groupby(['date'])['factor'].count()
longsum = df[df['factor'] > 0].groupby(['date'])['factor'].sum()
shortsum = df[df['factor'] < 0].groupby(['date'])['factor'].sum()

df_longshort = pd.DataFrame([longsum, shortsum, longNum, shortNum]).T.reset_index()

df_longshort.columns = ['date', 'long', 'short', 'longNum', 'shortNum']

pnl_file = pd.merge(pnl_file, df_longshort, on = ['date'], how = 'left')
pnl_file['pnl'] = pnl_file['pnl1']

pnl_file = pnl_file[['date', 'long', 'short', 'longNum', 'shortNum', 'pnl']]
pnl_file = pnl_file.set_index('date')

pnl_file.to_csv(v + "_pnl1.txt")

tmp_all = pd.DataFrame(tmp_all,columns = ['tmp_all'])
tmp_all['average']=tmp_all.mean(axis=1)

tmp_all['year'] = [int(x[:4]) for x in tmp_all.index.tolist()]
tmp_all['date'] = tmp_all.index.tolist()

def get_drawdown(x):
    drawdown = 0
    down = 0
    st = x['date'].tolist()[0]
    start = x['date'].tolist()[0]
    end = x['date'].tolist()[0]
    ret_list = x['average'].tolist()
    for i in range (0, len(x)):
        if ret_list[i] < 0:
            down = down + ret_list[i]
        else:
            if down < drawdown:
                drawdown = down
                end = x['date'].tolist()[i]
                start = st
                st = x['date'].tolist()[i]
                down = 0
            else:
                down = 0
                st = x['date'].tolist()[i]
    return (drawdown, start, end)

output = pd.DataFrame(index = list(set(tmp_all['year'].tolist())), columns = ['from', 'to', 'return', 'pnl_per_day', 'win_rate', 'sharpe', 'drawdown'])
output['return'] = tmp_all.groupby('year')['average'].sum()
output['pnl_per_day'] = tmp_all.groupby('year')['average'].mean()
output['from'] = tmp_all.groupby('year')['date'].min()
output['to'] = tmp_all.groupby('year')['date'].max()
output['win_rate'] = tmp_all[tmp_all['average'] > 0].groupby('year')['average'].count()/tmp_all.groupby('year')['average'].count()
output['drawdown'] = tmp_all.groupby('year')['average', 'date'].apply(get_drawdown)
output[['drawdown', 'dd_start', 'dd_end']] = output['drawdown'].apply(pd.Series)
#output['dd_start'] = tmp_all.groupby('year')['average', 'date'].apply(get_drawdown)
#output['dd_end'] = tmp_all.groupby('year')['average', 'date'].apply(get_drawdown)
output['sharpe'] = tmp_all.groupby('year')['average'].mean()/tmp_all.groupby('year')['average'].std()*np.sqrt(242)
output['win/loss'] = -tmp_all[tmp_all['average'] > 0].groupby('year')['average'].mean()/tmp_all[tmp_all['average'] <0].groupby('year')['average'].mean()

sum_ret = tmp_all['average'].mean()*242
sum_pnl_per_day = tmp_all['average'].mean()
sum_winrate = tmp_all[tmp_all['average'] > 0]['average'].count()/tmp_all['average'].count()
sum_drawdown = get_drawdown(tmp_all)
sumdd_start = sum_drawdown[1]
sumdd_end = sum_drawdown[2]
sum_dd = sum_drawdown[0]
sum_sharpe = tmp_all['average'].mean()/tmp_all['average'].std()*np.sqrt(250)
sum_win_ret = -tmp_all[tmp_all['average'] > 0]['average'].mean()/tmp_all[tmp_all['average'] <0]['average'].mean()
output.loc['summary'] = [min(tmp_all['date'].tolist()), max(tmp_all['date'].tolist()), sum_ret, sum_pnl_per_day, sum_winrate, sum_sharpe, sum_dd,sumdd_start,sumdd_end,sum_win_ret]

output.sort_values(by = ['to'])
