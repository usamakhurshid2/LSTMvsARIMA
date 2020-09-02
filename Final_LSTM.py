import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tqdm.keras import TqdmCallback
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from numpy import save

warnings.simplefilter(action='ignore', category=FutureWarning)

# ====================================================================#
#                           Input Variables                          #
# ====================================================================#

file_name = 'data.csv'
number_of_steps_in = 200  # Number of steps taken as input
number_of_steps_out = 1  # Number of steps taken as output
n_features = 1  # Total number of features
slicing_data_point = 4563  # Data dividing point


# ====================================================================#


def spliting_sequence(sequence, number_of_steps_in,
                      number_of_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # finding the end of this range
        end_ix = i + number_of_steps_in
        out_end_ix = end_ix + number_of_steps_out
        # checking if we are still in the sequence
        if out_end_ix > len(sequence):
            break
        # gathering input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


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
        # print(t)
        if change_p[t] > 0:
            # print(change_r[t])
            investment = investment * (1 + change_r[t])
    # print(investment)
    return investment


# ====================================================================#
#                  Data Reading & Pre-Processing                     #
# ====================================================================#

data = pd.read_csv(file_name)
predata = np.array(data.iloc[:, 1:2])
dataprep = np.array(data.iloc[:slicing_data_point, 1:2])
scaler = MinMaxScaler(feature_range=(0, 1))
price = scaler.fit_transform(dataprep)

X, y = spliting_sequence(price, number_of_steps_in, number_of_steps_out)
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))

# ====================================================================#


# ====================================================================#
#                           LSTM Model                               #
# ====================================================================#
nodes_one = 200  # Number of nodes in the first layer
nodes_two = 150  # Number of nodes in the second layer
activ = 'tanh'  # Activator name
optim = 'adam'  # Optimizer used
loss_function = 'mse'  # Loss function used
size_of_one_batch = 100  # Size of one batch
number_of_epochs = 5  # Total number of epochs
# ====================================================================#

model = Sequential()
model.add(LSTM(nodes_one,
               activation=activ,
               input_shape=(number_of_steps_in, n_features)))
model.add(RepeatVector(number_of_steps_out))
model.add(LSTM(nodes_two,
               activation=activ,
               return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer=optim,
              loss=loss_function)
model.fit(X, y,
          batch_size=size_of_one_batch,
          epochs=number_of_epochs,
          verbose=0,
          callbacks=[TqdmCallback(verbose=1)])
print(model.summary())

# ====================================================================#

# ====================================================================#
#                        Prediction Stage                            #
# ====================================================================#

index = len(predata) - slicing_data_point
Predicted_values = np.ones((1, 1))
cutting_point = slicing_data_point

for i in range(index):
    test_sample = np.array(data.iloc[cutting_point:cutting_point + number_of_steps_in, 1:2])
    test_scaled = scaler.fit_transform(test_sample)
    test_reshape = test_scaled.reshape((1, number_of_steps_in, n_features))
    prediction = model.predict(test_reshape, verbose=0)
    prediction_reshape = prediction.reshape(number_of_steps_out, -1)
    prediction_transformed = scaler.inverse_transform(prediction_reshape)
    cutting_point = cutting_point + number_of_steps_out
    # Breaking loop
    if cutting_point + number_of_steps_in > len(predata):
        break
    Predicted_values = np.concatenate((Predicted_values, prediction_transformed))
# ====================================================================#

# ====================================================================#
#                    Calculating error (RMSE)                        #
# ====================================================================#

previous_data = np.array(data.iloc[:slicing_data_point + number_of_steps_out + number_of_steps_in, 1:2])
Interm = np.array(Predicted_values[1:])
Final_graph = np.concatenate((previous_data, Interm))
to_train = np.array(data.iloc[len(predata) - len(Interm):, 1:2])

# print("Prediction: ", len(Final_graph), type(Final_graph))
# print("Original: ", len(predata), type(predata))


rmse = sqrt(mean_squared_error(to_train, Interm))
mae = mean_absolute_error(to_train, Interm)
profit = profit_calculator(to_train, Interm, 1000)
print("Root Mean Squared for our predicted vs actual values: ", rmse)
print("Mean Absolute Error for our predicted vs actual values: ", mae)
print("Total profit: ", profit)
# ====================================================================#


# ====================================================================#
#                            Plotting                                #
# ====================================================================#

Predicted = np.array(Final_graph[-530:, :])
Real = np.array(predata[-530:, :])
error = Real - Predicted
plt.plot(error)
plt.show()

save('LSTM.npy', Predicted)

plt.plot(Real, label="Original", color='orange')
plt.plot(Predicted, '--', color='blue', label="Predicted")
plt.xlabel("Days")
plt.ylabel("Market Index (KSE - 100)")
plt.legend(loc="upper left")
plt.show()
# ====================================================================#


rmse = sqrt(mean_squared_error(Real, Predicted))
mae = mean_absolute_error(Real, Predicted)
profit = profit_calculator(Real, Predicted, 1000)
print("Root Mean Squared for our predicted vs actual values: ", rmse)
print("Mean Absolute Error for our predicted vs actual values: ", mae)
print("Total profit: ", profit)