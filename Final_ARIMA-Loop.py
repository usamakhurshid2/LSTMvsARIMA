import warnings
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

#====================================================================#
#           Function for evaluation of one ARIMA model               #
#====================================================================#

def evaluate_arima_model(X, arima_order):

    # prepare training dataset
    train_size = 4763   # Choosing 90% as training data
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]

    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    # calculating error
    error = sqrt(mean_squared_error(test, predictions))
    return error

#====================================================================#
#               Function for finding the best order                  #
#====================================================================#

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    # creating loops for all three
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

#====================================================================#
#        Calling above made functions with set parameters            #
#====================================================================#

series = read_csv('data.csv', index_col='Date', parse_dates=True)
# evaluate parameters
p_values = range(0, 3)
d_values = 1
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)

#====================================================================#