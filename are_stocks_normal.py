import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

# TITLE
st.title('Do Stock Returns Follow a Normal Distribution?')

# USER INPUT
col1, col2 = st.columns(2)
ui_ticker = col1.selectbox('Pick a stock ticker', ['^GSPC', 'AAPL', 'GOOG', 'TSLA', 'BTC-USD', 'ETH-USD'])

intervals = ['1d', '1wk', '1mo', '3mo']
ui_interval = col2.selectbox('Pick a time interval', intervals)

col1, col2, col3, col4 = st.columns(4)
curr_year = date.today().year
curr_month = date.today().month
months = list(range(1, 13))
years = list(range(1980, curr_year+1))
ui_start_month = col1.selectbox('Start Month of study', months, index=0)
ui_start_year = col2.selectbox('Start Year of study', years, index=0)
ui_end_month = col3.selectbox('End Month of study', months, index=curr_month-1)
ui_end_year = col4.selectbox('End Year of study', years, index=len(years)-1)
ui_start = date(ui_start_year, ui_start_month, 1)
ui_end = date(ui_end_year, ui_end_month, 1)

# PULL DATA
data = yf.download(ui_ticker, interval=ui_interval, start=ui_start, end=ui_end, auto_adjust=True)

if len(data) > 30:
    valid = True
else:
    valid = False
    st.write("There are less than 30 data points. This is insufficient to complete the analysis.")
    st.write("Please adjust the start date, end date and/or interval and try again.")

if valid:
    # CLEAN DATA
    data = data[['Close']].copy()
    data.columns = data.columns.droplevel(1)
    data['return'] = np.log(data['Close'] / data['Close'].shift(1))
    data.drop(data.index[0], inplace=True)

    # PERFORM ANALYSIS
    mu = data['return'].mean()
    n = len(data)
    var = np.power(data['return'] - mu, 2).sum()/(n-1)
    std = np.sqrt(var)
    skew = np.power((data['return'] - mu)/std, 3).sum()*n/((n-1)*(n-2))
    ex_kurt = np.power((data['return'] - mu)/std, 4).sum()*n*(n+1)/((n-1)*(n-2)*(n-3)) - 3*(n-1)**2 / ((n-2)*(n-3))

    # DISPLAY RESULTS
    st.subheader('Do the returns look like a bell curve?')
    st.write('Note that a bell curve should be symetric (skew = 0) without too much weight in the tails (kurtosis=0)')
    st.write(f'Skew = {skew:.2f}')
    st.write(f'Excess Kurtosis = {ex_kurt:.2f}')
    fig, ax = plt.subplots()
    ax.hist(data['return'])
    ax.set_title(ui_ticker + " " + ui_interval + " returns")
    ax.set_xlabel(ui_interval + " stock returns")
    ax.set_ylabel("frequency")
    st.pyplot(fig)


    st.write('Data Summary')
    data_summary = pd.DataFrame({
        'description': ['number of data points', 
                        'first data point', 
                        'last data point', 
                        'interval of returns',
                        'average return',
                        'standard deviation of return'],
        'value': [str(n), 
                  str(data.index[0].date()),
                  str(data.index[-1].date()),
                  ui_interval,
                  f'{mu:.2%}',
                  f'{std:.2%}'
                  ]
        })
    data_summary.set_index('description', inplace=True)
    st.table(data_summary)
