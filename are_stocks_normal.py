import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from datetime import date

# TITLE
st.title('Do Stock Returns Follow a Normal Distribution (AKA Bell Curve🔔)?')
st.markdown('#')

# USER INPUT
col1, col2 = st.columns(2)
ui_ticker = col1.selectbox('Pick a stock ticker', ['^GSPC', 'AAPL', 'GOOG', 'TSLA', 'BTC-USD', 'ETH-USD'])

intervals = ['1d', '1wk', '1mo', '3mo']
ui_interval = col2.selectbox('Pick a time interval', intervals, index=1)

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
    if skew < -0.5:
        skew_msg = ('the distribution has a long left tail because people were panic selling!📉🫨 '
        'This indicates the returns do not follow a normal distribution')
    elif skew < 0.5:
        skew_msg = ('the distribution is relatively symetric '
        'This indicates the returns follow a normal distribution')
    else:
        skew_msg = ('the distribution has a long right tail because people are euphoric buying!🚀 '
        'This indicates the returns do not follow a normal distribution')

    ex_kurt = np.power((data['return'] - mu)/std, 4).sum()*n*(n+1)/((n-1)*(n-2)*(n-3)) - 3*(n-1)**2 / ((n-2)*(n-3))
    if ex_kurt < -0.5:
        kurt_msg = ('the distribution has light 🐥 tails because this stock is boring and doesnt move much. '
        'This indicates the returns do not follow a normal distribution')
    elif ex_kurt < 0.5:
        kurt_msg = ('the distribution has "normal" weight in the tails '
        'This indicates the returns follow a normal distribution')
    else:
        kurt_msg = ('the distribution has heavy 🦍 tails because the stock has frequent extreme price moves. '
        'This is likely due to earnings reports and investors reacting to breaking news! '
        'This indicates the returns do not follow a normal distribution')

    # DISPLAY RESULTS
    st.markdown('#')
    st.subheader('Do the returns look like a bell curve?')
    st.text('Note that a bell curve should be symetric meaning it should be a mirror image of itself.\n'
            'We can measure symetry with skewness which is 0 for a symetric distribution.\n\n'
            f'The skew for the observed data is {skew:.2f} which means {skew_msg}')
    st.text('')
    st.text('Note that a bell curve should have a medium amount of weight in its tails.\n'
            'This means that stocks have extreme moves ocassionally but not all the time\n'
            'We can measure tail weight with excess kurtosis which is 0 for a normal distribution.\n\n'
            f'The excess kurtosis for the observed data is {ex_kurt:.2f} which means {kurt_msg}')

    st.markdown('#')
    fig, ax = plt.subplots()
    ax.hist(data['return'])
    ax.set_title(ui_ticker + " " + ui_interval + " returns")
    ax.set_xlabel(ui_interval + " stock returns")
    ax.set_ylabel("number of returns in bucket")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    st.pyplot(fig)

    st.markdown('#')
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

    fig = plt.figure()
    sm.qqplot(data['return'], line='45', ax=fig.add_subplot(111))
    st.pyplot(fig) 
