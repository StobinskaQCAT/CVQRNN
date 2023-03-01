import matplotlib.pyplot as plt


def plot_prediction(true_data, pred, train_num, epoch, directory):
    """
    Plot the function (true data points), with the output of the network

    :param true_data: np.array(), data to plot, not split
    :param pred: np.array(), prediction of the network
    :param train_num: int, number of training data-points
    :param epoch: int, number of the epoch at which network was tested
    :param directory: string, directory name

    :return: -
    """

    data_len = len(true_data)

    # plot true data
    plt.plot(true_data, "b.", label="true data", alpha=0.5)
    plt.plot(true_data, "b--", alpha=0.1)

    # plot output of the network
    plt.plot(pred, ".", color="orange", label="prediction", alpha=0.5)
    plt.plot(pred, "-", color="orange", alpha=0.1)

    # plot the line between true and output
    plt.plot(
        (range(data_len), range(data_len)),
        (true_data, pred[:data_len]),
        c='red', alpha=0.2)

    # line indicating in which point the training stopped
    plt.axvline(train_num, color="red", linestyle="--")  #

    plt.title(f"Epoch {epoch}")
    plt.legend()
    plt.savefig("{}/prediction_epoch_{:02}.pdf".format(directory, epoch))
    plt.clf()


def plot_forecast(true_data, forecast, train_num, epoch, directory):
    """
    Plot the function, with separation for training and test data and
    plot the forecast

    :param true_data: np.array(), data to plot, not split
    :param forecast: np.array(), forecast of the network
    :param train_num: int, number of training data-points
    :param epoch: int, number of the epoch at which network was tested
    :param directory: string, directory name

    :return: -
    """

    # plot the whole function
    plt.plot(true_data, "y--")
    # plot the training data
    plt.plot(true_data[:train_num], "g-")

    # plot predicted points as 'x' markers
    plt.plot(
        range(train_num, train_num + len(forecast)),
        forecast,
        "b.")
    # plot predicted line
    plt.plot(range(train_num, train_num + len(forecast)), forecast, "b-",
             alpha=0.1)

    # give a title of the number of epoch
    plt.title(f"Epoch {epoch}")
    # indicate in which point the training stopped
    plt.axvline(train_num, color="red", linestyle="--")
    plt.savefig("{}/forecast_epoch_{:02}.pdf".format(directory, epoch))
    plt.clf()
