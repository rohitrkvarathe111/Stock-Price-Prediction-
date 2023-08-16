#!/usr/bin/env python
# coding: utf-8

#  <h1 style="font-family: Sans-Serif; padding: 26px; font-size: 50px; color: #641811; text-align: center; line-height: 1.5;"><b>STOCK PRICE PREDICTION </b>
# <hr>

# # <b>1. <span style='color:#80055D'>|</span> INTRODUCTION 🔎</b>

# Stock Price Prediction is a machine learning project that aims to predict the
# future value of a stock based on its past performance and other market conditions.
# The goal is to analyze financial data, including historical stock prices, news articles,
# economic indicators, and other relevant information to build a model that can accurately
# predict the direction of stock prices. The resulting predictions can be used by investors
# to make informed decisions about buying or selling stocks.

# ## <b>2.<span style='color:#80055D'>|</span> IMPORT NECESSARY LIBRARIES </b>

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import chart_studio.plotly as py


# In[4]:


import plotly.graph_objs as go
from plotly.offline import plot


# In[5]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


# ## <b>3.<span style='color:#80055D'>|</span> Imported the given data set as a CSV </b>

# 

# In[6]:


SP=pd.read_csv("SP data.csv")
SP


# ## <b>4.<span style='color:#80055D'>|</span> Data Head Displayed </b>

# In[7]:


SP.head(10)


# In[8]:


print(SP[SP.isnull().any(axis=1)])


# In[9]:


import pandas as pd

# Assuming the dataset is stored in a pandas DataFrame called 'df'
desired_value = 249.000000

# Filter the DataFrame based on the desired value in the 'Low' column
filtered_data = SP[SP['Low'] == desired_value]

# Print the filtered data
print(filtered_data)


# ## <b>5. <span style='color:#80055D'>|</span> Get basic information</b>

# In[10]:


SP.info()


# ## <b>6. <span style='color:#80055D'>|</span> Convert date in data(SP) to data end time </b>

# In[11]:


SP['Date'] = pd.to_datetime(SP['Date'])


# In[12]:


print(f'Dataframe contains SP between {SP.Date.min()} {SP.Date.max()}')


# In[13]:


print(f'Total dates = {(SP.Date.max()  - SP.Date.min()).days}')


# ## <b>7. <span style='color:#80055D'>|</span> Descriptive Statistics of Numeric Variables </b>

# In[14]:


SP.describe()


# ## <b>8. <span style='color:#80055D'>|</span> Make a plot for each column </b>

# In[15]:


SP[['Open','High','Low','Close','Adj Close']].plot(kind='box')


# # <b>9. <span style='color:#80055D'>|</span> Stock Price Plot </b>

# # Date wise closing price of the stock

# In[16]:


layout = go.Layout(
    title = 'Stock price plot')


# In[17]:


SP_data = [{'x':SP['Date'], 'y':SP['Close']}]
plot = go.Figure(data=SP_data)


# In[18]:


plot


# # <b>10. <span style='color:#80055D'>|</span> Import Sklearn  </b>

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# In[21]:


from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[22]:


X = np.array(SP.index).reshape(-1,1)
Y = SP['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# In[23]:


scaler = StandardScaler().fit(X_train)
scaler


# # <b>11. <span style='color:#80055D'>|</span> Stock price forecasting with the help of linear regression model and fitting  </b>

# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[26]:


trace0 = go.Scatter(
      x = X_train.T[0],
      y = Y_train,
   mode = 'markers',
    name = 'Actual'
)
trace1 = go.Scatter(
     x = X_train.T[0],
    y = lm.predict(X_train).T,
    mode = 'lines',
    name = 'Predicted'
)





SP_data = [trace0,trace1]
layout.xaxis.title.text = 'Day'


# In[27]:


plot2 = go.Figure(data=SP_data, layout=layout)
plot2


# This code defines a string scores that displays metrics for a linear regression model (lm) on both the training set (X_train and Y_train) and the test set (X_test and Y_test). The metrics displayed are the R^2 score and the mean squared error (MSE). The string uses f-string syntax to embed the results of the r2_score and mse functions applied to the train and test sets. The ljust and center methods are used to align the text in the string. When the scores string is printed, it will display a table showing the metric names, their values on the training set, and their values on the test set.

# In[28]:


scores = f'''
{'Metric'.ljust(10)}{'Train'.center(20)}{'Test'.center(20)}
{'r2_score'.ljust(10)}{r2_score(Y_train, lm.predict(X_train))}\t{r2_score(Y_test, lm.predict(X_test))}
{'MSE'.ljust(10)}{mse(Y_train, lm.predict(X_train))}\t{mse(Y_test, lm.predict(X_test))}
'''
print(scores)


# # THANKYOU 😊

#  <h1 style="font-family: Sans-Serif; padding: 26px; font-size: 50px; color: #641811; text-align: center; line-height: 1.5;"><b> Submitted by Rohit Varathe </b>
# <hr>

# In[ ]:




