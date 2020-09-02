# LSTMvsARIMA
A comparative paper on the financial forecasting ability of ARIMA and LSTM.
Data file contains the data of KSE-100 from Pakistan Stock Exchange. You can find three code files in there. 

First is the ARIMA_Loop file, this will find the best configuration from the data for which the rmse values are the lowest.
This combination of p,d and q can be used in the next code file i.e. ARIMA. 

From this file you can run a proper prediction using the given data. And also get the comparative graphs along with the profits calculated.
After this you can run the LSTM code file, which will run an LSTM model on the given data and do the same as ARIMA model code. Predicting, graphing and calculating profits based on those predicitons while using a LSMT model. 
