import matplotlib.pyplot as plt

def plot_forecasts(final_forecast, true_forecast):
    # Get the length of the prediction
    prediction_length = final_forecast.shape[1]

    # Create timestamps for x-axis
    timestamps = range(prediction_length)

    # Plot the forecasts
    plt.plot(timestamps, final_forecast.flatten(), label='Final Forecast', color='blue')
    plt.plot(timestamps, true_forecast.flatten(), label='True Forecast', color='red')

    # Add labels and legend
    plt.xlabel('Timestamp')
    plt.ylabel('Wind Speed')
    plt.legend()

    # Show the plot
    plt.show()
