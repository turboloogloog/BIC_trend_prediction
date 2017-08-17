#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:48:10 2017

@author: von
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
bitcoin =pd.read_csv('data.csv',parse_dates=[0], dayfirst = True)
bitcoin = bitcoin.iloc[::-1]
date1 = [ x.to_pydatetime() for x in bitcoin['Date'].tolist() ]
plt.figure()
y_Close = list(bitcoin['Price'])

plt.figure()
plt.plot(date1, y_Close)
plt.title('Bitcoin Price from 2012 to now')
plt.xlabel('Date')
plt.ylabel('Price/$')

observed_b42017 = bitcoin[bitcoin['Date']<'2016-08-11']
dates = [ x.to_pydatetime() for x in observed_b42017['Date'].tolist() ]
observed_values =observed_b42017['Price'].tolist()
first_observation =observed_values[0]
last_observation = observed_values[-1]
T=len(observed_values)
predicted_values = observed_values
for i in range(0, 12):
    h = i + 1
    dates.append(dates[-1] + relativedelta(months=+1))
    value=last_observation+h*((last_observation-first_observation)/(T-1))
    predicted_values.append(value)
plt.figure()
plt.plot(dates,predicted_values)
plt.title('Predicted')
plt.xlabel('Date')
plt.ylabel('Predicted_values')





