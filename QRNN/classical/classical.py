import argparse
import datetime
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import QRNN.utils.plot as plot
from QRNN.utils.LSTMCell import FeedBack
import QRNN.utils.dataLoader as dataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for the run')

    parser.add_argument('--in_len', type=int, help='Length of the input')
    parser.add_argument('--out_len',
                        type=int,
                        default=1,
                        help='Length of the output')
    parser.add_argument('--data_len',
                        type=int,
                        help='Limit of the training and validating dataset')
    parser.add_argument('--resolution',
                        type=int,
                        help='Number of points in the dataset')
    parser.add_argument('--noise',
                        type=float,
                        default=0.01,
                        help='Noise of data')
    parser.add_argument('--units',
                        type=int,
                        default=3,
                        help='Number of units of LSTM')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=10,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--run_num',
                        type=int,
                        default=1,
                        help='Number of runs')
    parser.add_argument('--data_name', default="sine_wave", help='Data set')
    parser.add_argument('--save_dir', help='Directory name to save files')
    args = parser.parse_args()
    return args


def main():
    # for keeping track of prediction loss, normal loss for train and val
    rep_pred_mse, rep_loss, rep_loss_val = [], [], []
    history_params = []
    func_save, pred_save, fore_save, epoch_save = [], [], [], []

    inputs = parse_args()

    # parameters of the run
    max_epochs = inputs.epochs
    batch_size = inputs.batch_size
    run_num = inputs.run_num
    save_dir = inputs.save_dir

    # parameters of the data
    in_len = inputs.in_len
    out_len = inputs.out_len
    data_len = inputs.data_len
    resolution = inputs.resolution
    noise = inputs.noise
    data_name = inputs.data_name

    # parameters of the network
    units = inputs.units
    lr = inputs.lr

    fig_dir = f"./figures/classical_runs/{save_dir}"
    res_dir = f"./results/classical/{save_dir}.pkl"

    #########################
    # Data preparation
    ########################

    # load train, validation data as well as full function (for plotting)
    if data_name in [
        "sine_wave", "sine_wave_2", "bessel", "triangle_wave", "line",
        "legandre", "legandre_4", "cos_wave_damped"
    ]:
        (x_train, y_train), (x_val, y_val), function = \
            eval(f"dataLoader.{data_name}")(
                data_len, resolution, in_len, out_len, noise, batch_size=1)

        # remember the length of training points and validation points
        train_len = x_train.shape[0]
        val_len = x_val.shape[0]

        # modify the labels (from full sequence to just last point)
        y_train = y_train[:, -out_len:]
        y_val = y_val[:, -out_len:]

    else:
        print("No matching data set")
        return 0

    x_train = np.expand_dims(x_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    x_val = np.expand_dims(x_val, axis=2)
    y_val = np.expand_dims(y_val, axis=2)

    for repeat in range(run_num):

        ##########################################
        # Tensorboard preparation
        ##########################################

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = f"./logs/{data_name}/units_{units}_lr_{lr}_inlen_{in_len}" \
                 f"_outlen_{out_len}_datalen_{resolution}__{current_time}"
        summary_writer = tf.summary.create_file_writer(logdir)

        ##########################################
        # Model preparation
        ##########################################

        model = FeedBack(units=units, out_steps=out_len)

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(learning_rate=lr),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        ##########################################
        # Training
        ##########################################

        # for keeping track of losses in one run
        prediction_loss_mse, loss, val_loss = [], [], []
        params_history = np.array([])

        for k in range(1, max_epochs + 1):
            # train for one epoch
            history = model.fit(x_train,
                                y_train,
                                epochs=1,
                                validation_data=(x_val, y_val),
                                batch_size=batch_size)

            # print summary once - to check number of parameters
            # if k == 1:
            #     print(model.summary())

            #######################################
            # plot the history of parameters change
            #######################################

            w = np.array([])
            for i in model.variables:
                w = np.append(w, i)

            # if we want to get the number of all parameters
            # print(f"Number of all parameters: {np.prod(w.shape)}")
            # return 0 

            # plot params history
            if k == 1:
                params_history = np.reshape(w, (-1, 1))
            else:
                params_history = np.append(params_history,
                                           np.reshape(w, (-1, 1)),
                                           axis=1)

            for x in params_history:
                plt.plot(x, color="b", alpha=0.5)

            plt.savefig(f"{fig_dir}/params_history_repeat_{repeat}.pdf")
            plt.clf()

            ###########################
            # Forecasting
            ###########################

            forecast = []
            # take last 'in_len' elements to feed to model
            past = function[train_len - in_len:train_len]

            # repeat few times and check how well is the prediction
            for i in range((len(function) - train_len) // out_len):
                # prediction of next 'out_len' number of points
                a = model(tf.reshape(past, (1, -1, 1))).numpy().flatten()[-1]

                forecast.append(a)  # remember predicted points

                # change past to newly predicted points of length 'in_len'
                past = np.append(past, a)[-in_len:]

            forecast = np.array(forecast)

            # plot the forecast every epoch to check how it is improving
            if k % 5 == 0 or k == 1:
                plot.plot_forecast(function, forecast, train_len, k, fig_dir)

            ##############################
            # Prediction
            ##############################

            prediction = np.array([])

            for i in range(0, len(function) - in_len, out_len):
                # prediction of next 'out_len' number of  points
                a = model(tf.reshape(function[i:i + in_len],
                                     (1, -1, 1))).numpy().flatten()[-1]

                prediction = np.append(prediction, a)

                print("\r{:.1f}% of samples".format(100 * i /
                                                    (len(function) - in_len)),
                      end="")

            if k % 5 == 0 or k == 1:
                plot.plot_prediction(function[in_len:], prediction,
                                     train_len - in_len, k, fig_dir)

            ############################
            # Saving parameters
            ############################

            # save loss for training
            loss.append(history.history['loss'])
            # ... and validation
            val_loss.append(history.history['val_loss'])

            # squared mean error between prediction and ground truth
            forecast_true = function[train_len:train_len + len(forecast)]
            future_loss_mse = np.mean((forecast - forecast_true) ** 2)

            # calculate the prediction loss
            prediction_loss_mse.append(future_loss_mse)

            # save only during the first run
            if repeat == 0:
                print("\nSaving forecast and prediction")
                # save the forecast points to pickle them
                fore_save.append(forecast)
                # save the prediction points to pickle them
                pred_save.append(prediction)

            # write to tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('training loss',
                                  history.history['loss'][0],
                                  step=k)
                tf.summary.scalar('validation loss',
                                  history.history['val_loss'][0],
                                  step=k)
                tf.summary.scalar('prediction loss (MSE)',
                                  future_loss_mse,
                                  step=k)

        # keep track of all losses from every run
        rep_pred_mse.append(prediction_loss_mse)
        rep_loss.append(loss)
        rep_loss_val.append(val_loss)
        history_params.append(params_history)

        # pickle losses every run so even if it breaks in the middle we
        # still have some data
        with open(res_dir, "wb") as output:
            pickle.dump((
                rep_loss,  # 0 - train loss
                rep_loss_val,  # 1 - test loss
                rep_pred_mse,  # 2 - prediction loss
                history_params,  # 3 - history parameters
                function,  # 4 - function points
                pred_save,  # 5 - prediction points
                fore_save),  # 6 - forecast points
                output)


if __name__ == '__main__':
    main()
