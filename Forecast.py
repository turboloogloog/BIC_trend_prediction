#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:00:21 2017
@author: von
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from holtwinters import linear
bitcoin =pd.read_csv('data.csv',parse_dates=[0], dayfirst = True)
bitcoin = bitcoin.iloc[::-1]
price=list(bitcoin['Price'])
date1 = [ x.to_pydatetime() for x in bitcoin['Date'].tolist() ]
plt.figure()
plt.plot(date1,price)
plt.title("Price")
price=bitcoin["Price"]
N=price.size
def sse(x, y):
    return np.sum(np.power(x - y,2))
model1=price.ewm(alpha =0.05, adjust = False)
model2=price.ewm(alpha =0.1 , adjust = False)
model3=price.ewm(alpha =0.3 , adjust = False)
model4=price.ewm(alpha =0.9 , adjust = False)
smoothed_1 = model1.mean()
smoothed_2 = model2.mean()
smoothed_3 = model3.mean()
smoothed_4 = model4.mean()
plt.figure()

line1, = plt.plot(smoothed_1, label = "Alpha 0.05, SSE {:.2f}".format(sse(bitcoin['Price'].values[1:], smoothed_1[:-1]) ))
line2, = plt.plot(smoothed_2, label = "Alpha 0.1,  SSE {:.2f}".format(sse(bitcoin['Price'].values[1:], smoothed_2[:-1]) ))
line3, = plt.plot(smoothed_3, label = "Alpha 0.3,  SSE {:.2f}".format(sse(bitcoin['Price'].values[1:], smoothed_3[:-1]) ))
line4, = plt.plot(smoothed_4, label = "Alpha 0.9,  SSE {:.2f}".format(sse(bitcoin['Price'].values[1:], smoothed_4[:-1]) ))
line_original,=plt.plot(price, label = 'Original')
plt.title("Various values of Alpha")
plt.xlim((N,1))
plt.legend(handles=[line_original,line1, line2, line3,line4])

sse_all_one = []
alphas = np.arange(0.01,1.00,0.01)
for an_alpha in alphas:
    smoothed = price.ewm(alpha =an_alpha, adjust = False).mean()
    sse_all_one.append( sse(smoothed[:-1], price.values[1:]) )
plt.figure()
plt.plot(sse_all_one)
plt.title("SSE for one step smoothing")
plt.ylabel("SSE")
plt.xlabel("Alpha")
plt.xticks(np.linspace(0, 100, 10), ["{0:1.1f}".format(x)
for x in np.linspace(0,1,10) ])
optimal_alpha_one = alphas[ np.argmin(sse_all_one) ]
print("Optimal Alpha for 1-step forecast{0}".format(optimal_alpha_one))

sse_all_two = []
for i in alphas:
    smoothed = price.ewm(alpha = i, adjust=False).mean()
    sse_all_two.append( sse(smoothed[:-2], price.values[2:]) )
plt.figure()
plt.plot(sse_all_two)
plt.title("SSE for two step smoothing")
plt.ylabel("SSE")
plt.xlabel("Alpha")
plt.xticks(np.linspace(0, 100, 10), ["{0:1.1f}".format(x)
for x in np.linspace(0,1,10) ])
optimal_alpha_two = alphas[np.argmin(sse_all_two)]
print("Optimal Alpha for 2-step forecast{0}".format(optimal_alpha_two))

sse_all_three = []
for i in alphas:
    smoothed = price.ewm(alpha = i, adjust=False).mean()
    sse_all_three.append( sse(smoothed[:-3], price.values[3:]) )
plt.figure()
plt.plot(sse_all_three)
plt.title("SSE for three step smoothing")
plt.ylabel("SSE")
plt.xlabel("Alpha")
plt.xticks(np.linspace(0, 100, 10), ["{0:1.1f}".format(x)
for x in np.linspace(0,1,10) ])
optimal_alpha_three = alphas[np.argmin(sse_all_three)]
print("Optimal Alpha for 3-step forecast{0}".format(optimal_alpha_three))
last_observed=(price.tolist())[-1]
smoothed_best=price.ewm(alpha = 0.95, adjust=False).mean().tolist()
for i in range(30):
    passed_value=0.95*last_observed+(1-0.95)*smoothed_best[-1]
    smoothed_best.append(passed_value)
plt.figure()
plt.xlim(1950,2100) 
plt.title("Forecast for next 30 days")
a=list(bitcoin['Price'])
plt.plot(a,label="Original")
plt.plot(smoothed_best,label='Forecast for next 30 days')
plt.legend(loc="lower right")

"""
Holt's Linear method 
"""
ts=price.tolist()
x_smoothed, Y, alpha, beta ,rmse= linear(ts,30)
plt.figure()
plt.xlim(1950,2100) 
plt.plot(a,label="Original")
plt.plot(x_smoothed,label='Holtwinters Linear forecast')
plt.plot(smoothed_best,label='Forecast for next 30 days where alpha=0.95')
plt.title('''Holt's Linear method to compute 30 days forecasts''')
plt.legend(loc="lower right")
