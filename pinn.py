import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt


def Supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n_out)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


class PINN:

    def __init__(self, mu, training):
        self.t_diff = 1  # Daily data
        self.gradient_t = (training.diff() / self.t_diff).iloc[1:]  # dx/dt
        self.gradient_tt = (self.gradient_t.diff() /
                            self.t_diff).iloc[1:]  # d2x/dt2
        self.gradient_t = self.gradient_t.reset_index(drop=True)
        self.gradient_tt = self.gradient_tt.reset_index(drop=True)
        self.mu = mu

    def load_data(self, filename, columnNo):
        self.data = pd.read_csv(filename)
        self.training_set = self.data.iloc[:, columnNo]
        self.test = self.training_set.tail(10)
        self.training_set = self.training_set.iloc[:-10]
        self.training_set = self.training_set.reset_index(drop=True)
        self.df = pd.concat((self.training_set, self.gradient_t), axis=1)
        self.gradient_tt.columns = ["grad_tt"]
        self.df = pd.concat((self.df, self.gradient_tt), axis=1)
        self.df.columns = ["y_t", "grad_t", "grad_tt"]
        self.df = self.df.dropna()

    def convert(self, offset=30, in_val=35, out_val=10):
        self.offset = offset
        self.in_val = in_val
        self.out_val = out_val

        self.data = Supervised(self.df.values, in_val, out_val)
        cols_to_drop = [
            f"var{i}(t-{j})" for j in range(in_val, 1, -1)
            for i in range(2, 4)
        ]
        self.data.drop(cols_to_drop, axis=1, inplace=True)

        self.train = np.array(self.data[0:len(self.data) - 1])
        self.forecast = np.array(self.data.tail(1))

        self.trainy = self.train[:, -offset:]
        self.trainX = self.train[:, :-offset]

        self.forecasty = self.forecast[:, -offset:]
        self.forecastX = self.forecast[:, :-offset]

        self.trainX = self.trainX.reshape(
            (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.forecastX = self.forecastX.reshape(
            (self.forecastX.shape[0], 1, self.forecastX.shape[1]))

    def vpinn_loss_fn(self, y_true, y_pred):
        squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])
        #squared_difference2 = tf.square(y_true[:, 2]-y_pred[:, 2])
        #squared_difference1 = tf.square(y_true[:, 1]-y_pred[:, 1])
        squared_difference3 = tf.square(y_pred[:, 2] - self.mu_var *
                                        (y_pred[:, 1] -
                                         (y_pred[:, 0]**2 * y_pred[:, 1]) -
                                         (1 / self.mu_var) * y_pred[:, 0]))
        return tf.reduce_mean(
            squared_difference,
            axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)

    def shm_loss_fn(self, y_true, y_pred, spring_constant=4.0, mass=1.0):
        omega = np.sqrt(spring_constant / mass)

        squared_difference = tf.square(y_true[:, 0] - y_pred[:, 0])
        #squared_difference2 = tf.square(y_true[:, 2]-y_pred[:, 2])
        #squared_difference1 = tf.square(y_true[:, 1]-y_pred[:, 1])
        squared_difference3 = tf.square(y_pred[:, 2] +
                                        (omega**2) * y_pred[:, 0])
        return tf.reduce_mean(
            squared_difference,
            axis=-1) + 0.2 * tf.reduce_mean(squared_difference3, axis=-1)

    def lorenz_loss_fn(self, y_true, y_pred):
        mu = tf.Variable(4, name="mu", trainable=True, dtype=tf.float32)
        splitr = 0.8
        sigma = 10
        rho = 28
        beta = 8 / 3
        squared_difference_x = tf.square(y_true[:, 0] - y_pred[:, 0])
        squared_difference_z = tf.square(y_pred[:, 2] - mu *
                                         (y_pred[:, 1] -
                                          (y_pred[:, 0]**2) * y_pred[:, 1]))

        return tf.reduce_mean(squared_difference_x, axis=-1) + \
                0.2 * tf.reduce_mean(squared_difference_z, axis=-1)

    def train_model(self, type):
        print(type)
        self.mu_var = tf.Variable(4,
                                  name="mu",
                                  trainable=True,
                                  dtype=tf.float32)
        splitr = 0.8

        self.model = Sequential()
        self.model.add(
            LSTM(50, input_shape=(self.trainX.shape[1], self.trainX.shape[2])))
        self.model.add(Dense(30))
        if type == "VPINN":
            self.model.compile(loss=self.vpinn_loss_fn, optimizer='adam')
        elif type == "SHM":
            self.model.compile(loss=self.shm_loss_fn, optimizer='adam')
        elif type == "LORENZ":
            self.model.compile(loss=self.lorenz_loss_fn, optimizer='adam')
        self.history = self.model.fit(
            self.trainX[:int(splitr * self.trainX.shape[0])],
            self.trainy[:int(splitr * self.trainX.shape[0])],
            epochs=100,
            batch_size=64,
            validation_data=(
                self.trainX[int(splitr *
                                self.trainX.shape[0]):self.trainX.shape[0]],
                self.trainy[int(splitr *
                                self.trainX.shape[0]):self.trainX.shape[0]]),
            shuffle=False)

    def evaluate_model(self):
        self.forecast_without_mc = self.forecastX
        yhat_without_mc = self.model.predict(
            self.forecast_without_mc)  # Step Ahead Prediction
        self.forecast_without_mc = self.forecast_without_mc.reshape(
            (self.forecast_without_mc.shape[0],
             self.forecast_without_mc.shape[2]))  # Historical Input

        self.final_forecast = yhat_without_mc[:, 0:self.offset - 1:3]
        self.final_forecast[self.final_forecast < 0] = 0
        self.true_forecast = self.forecasty[:, 0:self.offset - 1:3]
        return mean_absolute_error(self.final_forecast, self.true_forecast)

    def plot_forecasts(self, final_forecast, true_forecast, name=0):
        # Get the length of the prediction
        prediction_length = final_forecast.shape[1]

        # Create timestamps for x-axis
        timestamps = range(prediction_length)

        # Plot the forecasts
        plt.plot(timestamps,
                 final_forecast.flatten(),
                 label='Final Forecast',
                 color='blue')
        plt.plot(timestamps,
                 true_forecast.flatten(),
                 label='True Forecast',
                 color='red')

        # Add labels and legend
        plt.xlabel('Timestamp')
        plt.ylabel('Wind Speed')
        plt.legend()

        if name == 0:
            plt.savefig(name + ".png")
        else:
            print("Plot will not be printed")

        # Show the plot
        plt.show()
