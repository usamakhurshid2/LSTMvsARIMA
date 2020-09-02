import warnings
from pandas import read_csv
from numpy import save
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from datetime import datetime
import numpy as np



def parse(x):
    return datetime.strptime(x, '%Y-%m-%d')


def profit_calculator(real_values, predicted_values, investment):
    delta_real = list()
    delta_predicted = list()
    for i in range(len(real_values) - 1):
        change_in_prediction = (predicted_values[i + 1] - predicted_values[i]) / predicted_values[i]
        change_in_real = (real_values[i + 1] - real_values[i]) / real_values[i]
        delta_real.append(change_in_real)
        delta_predicted.append(change_in_prediction)
    change_r = np.array(delta_real)
    change_p = np.array(delta_predicted)
    for t in range(len(change_r)):
        if change_p[t] > 0:
            investment = investment * (1 + change_r[t])
    return investment

# ====================================================================#
#                        Loading dataset                             #
# ====================================================================#


series = read_csv('data.csv',
                  header=0,
                  index_col='Date',
                  squeeze=True,
                  parse_dates=True,
                  date_parser=parse)

X = series
size = 4763
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
error = list()
original = list()
# ====================================================================#

# ====================================================================#
#                          ARIMA model                               #
# ====================================================================#
p = 2  # AR term, defined from plotting Auto-correlation
d = 1  # Differencing
q = 0  # Moving Average term, defined from plotting Partial ACF
# ====================================================================#

for t in range(len(test)):
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    original.append(obs)
    error_term = obs - yhat
    error.append(error_term)
    warnings.filterwarnings("ignore")
    print('predicted=%3f, '
          'expected=%3f, '
          'error=%3f'
          % (yhat, obs, error_term))

print(model_fit.summary())
# ====================================================================#

# ====================================================================#
#                       Calculating Error                            #
# ====================================================================#

rmse = sqrt(mean_squared_error(original, predictions))
mae = mean_absolute_error(original, predictions)
investment = profit_calculator(original, predictions, 1000)

print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % mae)
print('Profit', investment)

# ====================================================================#
#                            Plotting                                 #
# ====================================================================#

pyplot.plot(error)
pyplot.show()

pyplot.plot(original, label="Original", color='orange')
pyplot.plot(predictions, '--', color='blue', label="Predicted", )
pyplot.xlabel("Days")
pyplot.ylabel("Market Index (KSE - 100)")
pyplot.legend(loc="upper left")
pyplot.show()

print(original)
# ====================================================================#
