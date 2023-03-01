import argparse
import datetime
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import QRNN.utils.QRNNCell as QRNNCell
import QRNN.utils.plot as plot
import QRNN.utils.dataLoader as dataLoader


def cost_functions(qrnn, xs, ys, gamma):
    """
    Calculate the cost of the qrnn network

    :param qrnn: class(QRNN), qrnn cell with its trained weights
    :param xs: np.array(), input data
    :param ys: np.array(), output data
    :param gamma: float, penalty for losing trace

    :return: (float, float), (1) loss (MSE) of all inputs and outputs and
        (2) trace of the final state
    """

    loss = 0
    state = None

    # calculating the immediate loss does not make a good comparison with
    # classical approach where only prediction goes to the cost function
    immediate_loss = False

    # for input data
    for x, y in zip(xs, ys):
        mean_x, state = qrnn(x, state)
        if immediate_loss:
            # calculate loss of immediate response
            loss += tf.reduce_mean((mean_x - y) ** 2)

    # predict first output
    # calculating the loss only at the output works much better and simple
    # function is almost perfect
    if not immediate_loss:
        loss += tf.reduce_mean((mean_x - ys[-1]) ** 2)

    # try to predict rest of the output based on previous outputs --
    # forecasting; it's called autoregressive model
    for y in ys[len(xs):]:
        mean_x, state = qrnn(mean_x, state)
        loss += tf.reduce_mean((mean_x - y) ** 2)

    # normalize the loss function
    if immediate_loss:
        loss /= len(ys)
    else:
        norm = 1 if len(ys) == len(xs) else len(ys) - len(xs)
        loss /= norm

    # calculate the trace and loss connected to it
    trace = tf.abs(tf.reduce_mean(state.trace()))
    loss += gamma * (1 - trace) ** 2

    return loss, trace


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for the run')

    parser.add_argument('--in_len', type=int, default=4,
                        help='Length of the input')
    parser.add_argument('--out_len', type=int, default=1,
                        help='Length of the output')
    parser.add_argument('--in_modes', type=int, default=1,
                        help='Input modes')
    parser.add_argument('--hid_modes', type=int, default=1,
                        help='Hidden modes')
    parser.add_argument('--cutoff', type=int, default=4,
                        help='Cutoff dimension')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for the training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--gamma', type=float, default=10,
                        help='Trace penalty')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--layers', type=int, default=1,
                        help='Number of layers in one cell')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions in one cell')
    parser.add_argument('--nonlinear', type=int, default=0,
                        help='Number of layers in one cell')
    parser.add_argument('--energy_transmissivity', type=float, default=0,
                        help='Parameter of the lossy channel')
    parser.add_argument('--run_num', type=int, default=1,
                        help='Number of runs')
    parser.add_argument('--data_name', default="sine_wave",
                        help='Data set')
    parser.add_argument('--data_len', type=int, default=20,
                        help='Limit of the training and validating dataset')
    parser.add_argument('--resolution', type=int, default=100,
                        help='Number of points in the dataset')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise of data')
    parser.add_argument('--save_dir',
                        help='Directory name to save files')
    args = parser.parse_args()
    return args


def main():
    """
    Training network for the prediction task. It consists of few parts:
    1) Data loader -- specify your data, which will be split in the
        train/validation and batched
    2) Training -- optimize the circuit for the specific task
    3) Predict -- plot some useful plot (prediction, loss etc.)
    """

    # for keeping track of prediction loss, normal loss for train and val
    # and traces of the output density matrix
    rep_pred_mse = []
    rep_loss, rep_loss_val = [], []
    rep_trace, rep_trace_val = [], []
    history_params = []
    pred_save, fore_save = [], []

    #############################
    # Parser
    #############################

    inputs = parse_args()

    # characterize the dataset

    # length of the input data
    in_len = inputs.in_len
    # length of the output data (usually =1)
    out_len = inputs.out_len
    # N; max val for which function is evaluated
    data_len = inputs.data_len
    # number of points between (0,f(N))
    resolution = inputs.resolution
    # strength of the noise in the data values f(x) + \eps
    noise = inputs.noise

    # define the number of used wires (modes) as well as cutoff
    # dimension for engine and batch size

    # modes for data input; those are measured
    input_modes = inputs.in_modes
    # hidden modes; those are not measured
    hidden_modes = inputs.hid_modes
    # if it's too big it will eat whole ram, but better precision
    # can be achieved
    cutoff_dim = inputs.cutoff
    batch_size = inputs.batch_size

    # characteristics of the network

    # number of different layers
    layers = inputs.layers
    # lossy channel parameters (0 - lossless, 1 - total loss)
    energy_transmissivity = inputs.energy_transmissivity
    # number of repetitions of all the layers (usually layers=1)
    repetitions = inputs.repetitions
    # use kerr nonlinearity (1) or not (0)
    nonlinear = inputs.nonlinear

    # specify hyperparameters of training

    # number of maximal epochs
    epochs = inputs.epochs
    # penalty for trace; it keeps the trace close to 1
    gamma = inputs.gamma
    # learning rate for Adam
    lr = inputs.lr
    # number of runs (to gather statistics)
    run_num = inputs.run_num
    # name of the experiment to save figs and data
    data_name = inputs.data_name
    # name of the directory to which save the experiment
    save_dir = inputs.save_dir

    fig_dir = f"./figures/quantum_runs/{save_dir}"
    res_dir = f"./results/quantum/{save_dir}.pkl"

    #########################
    # Data preparation
    #########################

    # load train, validation data as well as full function (for plotting)
    if data_name in ["sine_wave", "sine_wave_2", "bessel", "triangle_wave",
                     "line", "legandre", "legandre_4", "cos_wave_damped"]:
        (x_train, y_train), (x_val, y_val), function = \
            eval(f"dataLoader.{data_name}")(data_len, resolution, in_len,
                                            out_len, noise, batch_size)

        # remember the length of training points and validation points
        train_len = x_train.shape[0] * batch_size
        val_len = x_val.shape[0]

    elif data_name in ["weather", "chlorine", "sunspots", "santafe",
                       "cyclones"]:
        (x_train, y_train), (x_val, y_val), function = \
            eval(f"dataLoader.{data_name}")(in_len, out_len, batch_size)

        # for playing with data we need to keep only length of the 
        # one particular run (in our case hurricane)
        train_len = int(function.shape[0] * 0.8)
        val_len = int(function.shape[0] * 0.2)

    else:
        print("No matching data set")
        return 0

    # set the random seed
    # tf.random.set_seed(137)
    # np.random.seed(137)

    ############################
    #  Loss of simple prediction
    ############################

    # calculate the loss of not predicting anything at all -- just
    # take the last value
    loss_train_no = 0
    loss_val_no = 0

    for x, y in zip(x_train[:, -1], y_train):
        loss_train_no += np.average((x - y) ** 2)

    for x, y in zip(x_val[:, -1], y_val):
        loss_val_no += np.average((x - y) ** 2)

    loss_no = (loss_val_no + loss_train_no) / (len(y_train) + len(y_val))

    print(f"\nLoss with simplest prediction is {loss_no}\n")

    #############################
    # Experiment (repeat)
    #############################

    # repeat several times to get good (*some) statistic
    for repeat in range(run_num):

        ###########################
        # Tensorboard preparations
        ###########################

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = f"./logs/{data_name}/modes_{input_modes + hidden_modes}" \
                 f"_cut_{cutoff_dim}_lr_{lr}_inlen_{in_len}_" \
                 f"outlen_{out_len}_datalen_{resolution}_gamma_{gamma}" \
                 f"_layers_{layers}_nonlinear_{nonlinear}" \
                 f"_channel_loss_{energy_transmissivity}___" \
                 f"{current_time}"
        summary_writer = tf.summary.create_file_writer(logdir)

        ###############################
        # Initialization
        ###############################

        # initialize the qrnn
        qrnn = QRNNCell.QRNN(input_modes, hidden_modes, cutoff_dim,
                             nonlinear=nonlinear, layers=layers,
                             energy_transmissivity=energy_transmissivity)

        # set up the optimizer
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # for keeping track of losses and traces in one run
        # through all epochs
        losses, losses_val = [], []
        traces, traces_val = [], []
        prediction_losses_mse = []

        # calculate the number of all batches
        all_batches = train_len // batch_size

        #################################
        # Main loop
        #################################

        start_all = time.perf_counter()  # time all epochs

        # perform the optimization
        for k in range(1, epochs + 1):
            # keep track of the loss in one epoch
            loss, loss_val = 0, 0
            trace, trace_val = 0, 0

            #########################
            # Plot params history
            #######################

            if k == 1:
                params_history = np.reshape(qrnn.weights.numpy(), (-1, 1))
                scale_history = qrnn.scale.numpy()
            else:
                params_history = np.append(params_history,
                                           np.reshape(qrnn.weights.numpy(),
                                                      (-1, 1)), axis=1)
                scale_history = np.append(scale_history, qrnn.scale.numpy())

            # Color parameters as passive and active
            for j, x in enumerate(params_history):
                if qrnn.color_mask[j] == 0:
                    plt.plot(x, color="r", alpha=0.5, label="passive")
                if qrnn.color_mask[j] == 1:
                    plt.plot(x, color="b", alpha=0.5, label="active")

            plt.plot(scale_history, color="g", label="scale")

            # make the labels do not repeat
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.savefig(f"{fig_dir}/params_history_repeat_{repeat}.pdf")
            plt.clf()

            ##########################
            # Training
            ##########################

            start = time.perf_counter()  # time one epoch

            print("\n------------------------\n"
                  "Now training\n"
                  "------------------------")

            for j, (x, y) in enumerate(zip(x_train, y_train)):
                print(
                    "\r{:.1f}% of samples".format(100 * j / x_train.shape[0]),
                    end="")

                with tf.GradientTape() as tape:
                    loss_, trace_ = cost_functions(qrnn, x, y, gamma)

                    # one repetition of the optimization
                    grad_w, grad_s = tape.gradient(loss_,
                                                   [qrnn.weights, qrnn.scale])
                    opt.apply_gradients(
                        zip([grad_w, grad_s], [qrnn.weights, qrnn.scale]))

                # classical network does not use regularization,
                # so it is fair to remove regularization term from quantum
                loss += loss_ - gamma * (1 - trace_) ** 2
                trace += trace_

            ##########################
            # Validating
            ##########################

            print("\n------------------------\n"
                  "Now validating\n"
                  "------------------------")

            for j, (x, y) in enumerate(zip(x_val, y_val)):
                print("\r{:.1f}% of samples".format(100 * j / x_val.shape[0]),
                      end="")

                loss_, trace_ = cost_functions(qrnn, x, y, gamma)

                # removing regularization term
                loss_val += loss_ - gamma * (1 - trace_) ** 2
                trace_val += trace_

            # Forecasting and predicting is time-consuming so do it only
            # every *ith* epoch
            if (k % 5 == 0 or k == 1):
                ##########################
                # Forecasting
                ##########################

                print("\n------------------------\n"
                      "Now forecasting\n"
                      "------------------------")

                # points from which network forecast
                point_in = function[train_len - in_len:train_len]

                # get the networks forecast
                forecast = QRNNCell.get_fore(qrnn, point_in,
                                             len(function) - train_len,
                                             out_len)
                # plot the forecast of the network
                plot.plot_forecast(function, forecast, train_len, k, fig_dir)

                ##########################
                # Train and val plot
                ##########################

                print("\n------------------------\n"
                      "Now plotting train and val\n"
                      "------------------------")

                # get the networks predictions
                prediction = QRNNCell.get_pred(qrnn, function, in_len, out_len)

                # plot the prediction of the network
                plot.plot_prediction(function[in_len:], prediction, train_len,
                                     k, fig_dir)

                # save only during the first run
                if repeat == 0:
                    print("\nSaving forecast and prediction")
                    # save the forecast points to pickle them
                    fore_save.append(forecast)
                    # save the prediction points to pickle them
                    pred_save.append(prediction)

                # squared mean error between prediction and ground truth
                forecast_true = function[train_len:train_len + len(forecast)]
                future_loss_mse = np.mean((forecast - forecast_true) ** 2)

            ##########################
            # Save & tensorboard
            ##########################

            # save all the losses and traces after one epoch
            losses.append(loss / all_batches)
            traces.append(trace / all_batches)
            losses_val.append(loss_val / val_len)
            traces_val.append(trace_val / val_len)
            prediction_losses_mse.append(future_loss_mse)

            # write to tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('training loss', losses[-1], step=k)
                tf.summary.scalar('validation loss', losses_val[-1], step=k)
                tf.summary.scalar('training trace', traces[-1], step=k)
                tf.summary.scalar('validation trace', traces_val[-1], step=k)
                tf.summary.scalar('prediction loss (MSE)', future_loss_mse,
                                  step=k)

            # time one epoch
            end = time.perf_counter()

            # prints progress at every rep
            if k % 1 == 0:
                print("\n")
                print(
                    "Rep: {} Cost: {:.4f} Trace: {:.4f} Time: {:.2f}s".format(
                        k, losses[-1], traces[-1],
                        end - start))

                print(
                    "Rep: {} Cost: {:.4f} Trace: {:.4f} Time: {:.2f}s".format(
                        k, losses_val[-1], traces_val[-1],
                        end - start))

                print()

        # time all epochs
        end_all = time.perf_counter()

        print("It took in total {:.2f} minutes".format(
            (end_all - start_all) / 60))  # tell time

        # save all the losses and traces from all repetitions (for statistics)
        rep_loss.append(losses)
        rep_loss_val.append(losses_val)
        rep_trace.append(traces)
        rep_trace_val.append(traces_val)
        rep_pred_mse.append(prediction_losses_mse)
        history_params.append(params_history)

        #  pickle the data just in case every repetition to not lose anything
        with open(res_dir, "wb") as output:
            pickle.dump(
                (
                    rep_loss,  # 0 - train loss
                    rep_loss_val,  # 1 - test loss
                    rep_pred_mse,  # 2 - prediction loss
                    history_params,  # 3 - history parameters
                    function,  # 4 - function points
                    pred_save,  # 5 - prediction points
                    fore_save,  # 6 - forecast points
                    rep_trace,  # 7 - train trace
                    rep_trace_val,  # 8 - test trace
                    loss_no),  # 9 - naive prediction loss
                output)


if __name__ == '__main__':
    main()
