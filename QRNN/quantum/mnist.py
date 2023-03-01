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


def sigmoid(x):
    """
    Calculate the sigmoid

    :param x: np.array(float)
    :return: np.array(float)
    """
    # ALWAYS USE THE TF FUNCTIONS, DO NOT MIX TF WITH NP!!!!
    # THEN GRADIENT WILL BE OK!
    return 1 / (1 + tf.math.exp(-x))


def NLL(true, pred):
    """
    Caluclate the negative log-loss
    :param true: np.array(), labels
    :param pred: np.array(), predictions
    :return: float, loss
    """
    return - tf.reduce_mean((true * tf.math.log(pred) +
                             (1 - true) * tf.math.log(1 - pred)))


def cost_functions_bin(qrnn, xs, ys, gamma):
    """
    Calculate the cost of the qrnn network for MNIST binary
    classification

    :param qrnn: class(QRNN), qrnn cell with its trained weights
    :param xs: np.array(), input data
    :param ys: np.array(), output data
    :param gamma: float, penalty for losing trace

    :return: (float, float), (1) loss (MSE) of all inputs and outputs
    and (2) trace of the final state
    """

    loss = 0

    # first run without previous state
    mean_x, state = qrnn(xs[0], None)

    # for input data
    for x in xs[1:-1]:
        # calculate loss of immediate response
        # loss += tf.reduce_mean((mean_x - x) ** 2)
        mean_x, state = qrnn(x, state)

    mean_x, state = qrnn(xs[-1], state, final=True)

    # make sure that the shapes are the same
    # ys = np.reshape(ys, (1, -1))
    # mean_x = np.reshape(mean_x, (1, -1))
    pred = sigmoid(mean_x)

    # print(pred, ys)

    loss += NLL(ys, pred)

    # calculate the loss and trace
    trace = tf.abs(tf.reduce_mean(state.trace()))
    loss += gamma * (1 - trace) ** 2

    # calculate accuracy
    acc = np.mean((pred > 0.5) == ys)

    return loss, trace, acc


def parse_args():
    parser = argparse.ArgumentParser(description='Arguments for the run')

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
    parser.add_argument('--data_len', type=int, default=100,
                        help='Limit of the training and validating dataset')
    parser.add_argument('--resolution', type=int, default=3,
                        help='Number of points in the dataset')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise of data')
    parser.add_argument('--save_dir', default="mnist",
                        help='Directory name to save files')
    args = parser.parse_args()
    return args


def main():
    # for keeping track of prediction loss, normal loss for train and
    # val and traces of the output density matrix
    rep_loss, rep_loss_val = [], []
    rep_trace, rep_trace_val = [], []
    rep_acc, rep_acc_val = [], []
    history_params = []

    #############################
    # Parser
    #############################

    inputs = parse_args()

    # characterize the dataset

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
    # too big will eat whole ram, but better precision can be achieved
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
    # name of the directory to which save the experiment
    save_dir = inputs.save_dir

    fig_dir = f"./figures/quantum_runs/{save_dir}"
    res_dir = f"./results/quantum/{save_dir}.pkl"

    #########################
    # Data preparation
    #########################

    # load train, validation data 
    (x_train, y_train), (x_val, y_val) = \
        dataLoader.mnist_data(resolution, data_len, batch_size)

    # plot some data
    fig_side = 4
    fig, axarr = plt.subplots(1, fig_side)

    if batch_size == 1:
        for i in range(0, fig_side):
            axarr[i].imshow(np.reshape(x_train[i + 1], (resolution, resolution)), cmap='viridis',
                            interpolation='nearest')
            axarr[i].set_title(f"Label {y_train[i + 1]}")
            axarr[i].axis('off')
    else:
        for i in range(0, fig_side):
            axarr[i].imshow(np.reshape(x_train[0, :, i + 1], (resolution, resolution)), cmap='viridis',
                            interpolation='nearest')
            axarr[i].title(f"Label {y_train[0, :, i + 1]}")
            axarr[i].axis('off')

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/demo_reso_{resolution}.pdf", bbox_inches='tight')
    # plt.show()

    # remember the length of training points and validation points
    train_len = x_train.shape[0] * batch_size
    val_len = x_val.shape[0]

    # set the random seed
    # tf.random.set_seed(137)
    # np.random.seed(137)

    # repeat several times to get good statistic i.e. to plot average and std
    for repeat in range(run_num):

        ###########################
        # Tensorboard preparations
        ###########################

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = f"./logs/MNIST/modes_{input_modes + hidden_modes}" \
                 f"_cut_{cutoff_dim}_lr_{lr}_datalen_{data_len}" \
                 f"_resolution{resolution}_gamma_{gamma}_layers_{layers}" \
                 f"_nonlinear_{nonlinear}___{current_time}"
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

        # keep track of losses and traces in one run (through all epochs)
        losses, losses_val = [], []
        traces, traces_val = [], []
        accuracies, accuracies_val = [], []

        # calculate the number of all batches
        all_batches = train_len // batch_size

        #################################
        # Training
        #################################

        START = time.perf_counter()  # time all epochs

        # perform the optimization
        for k in range(epochs):
            # keep track of the loss in one epoch
            loss, loss_val = 0, 0
            trace, trace_val = 0, 0
            accuracy, accuracy_val = 0, 0

            #########################
            # plot params history
            #######################

            if k == 0:
                params_history = np.reshape(qrnn.weights.numpy(), (-1, 1))
                scale_history = qrnn.scale.numpy()
            else:
                params_history = np.append(params_history,
                                           np.reshape(qrnn.weights.numpy(),
                                                      (-1, 1)), axis=1)
                scale_history = np.append(scale_history, qrnn.scale.numpy())

            # Colors were used before when there was only one layer
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
                    loss_, trace_, acc_ = cost_functions_bin(qrnn, x, y, gamma)

                    # one repetition of the optimization
                    grad_w, grad_s = \
                        tape.gradient(loss_, [qrnn.weights, qrnn.scale])
                    opt.apply_gradients(
                        zip([grad_w, grad_s], [qrnn.weights, qrnn.scale]))

                loss += loss_ - gamma * (1 - trace_) ** 2
                trace += trace_
                accuracy += acc_

            ##########################
            # Validating
            ##########################

            print("\n------------------------\n"
                  "Now validating\n"
                  "------------------------")

            for j, (x, y) in enumerate(zip(x_val, y_val)):
                print("\r{:.1f}% of samples".format(100 * j / x_val.shape[0]),
                      end="")

                loss_, trace_, acc_ = cost_functions_bin(qrnn, x, y, gamma)

                loss_val += loss_ - gamma * (1 - trace_) ** 2
                trace_val += trace_
                accuracy_val += acc_

            ##########################
            # Save & tensorboard
            ##########################

            # save all the losses and traces after one epoch
            losses.append(loss / all_batches)
            traces.append(trace / all_batches)
            accuracies.append(accuracy / all_batches)
            losses_val.append(loss_val / val_len)
            traces_val.append(trace_val / val_len)
            accuracies_val.append(accuracy_val / val_len)

            # write to tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('training loss', losses[-1], step=k)
                tf.summary.scalar('validation loss', losses_val[-1], step=k)
                tf.summary.scalar('training trace', traces[-1], step=k)
                tf.summary.scalar('validation trace', traces_val[-1], step=k)
                tf.summary.scalar('training accuracy', accuracies[-1], step=k)
                tf.summary.scalar('validation accuracy', accuracies_val[-1],
                                  step=k)

            end = time.perf_counter()  # time one epoch

            # prints progress at every rep
            if k % 1 == 0:
                print(
                    "\n\n"
                    "Rep: {} Cost: {:.4f} Trace: {:.4f} Acc: {:.4f} "
                    "Time: {:.2f}s".format(k, losses[-1], traces[-1],
                                           accuracies[-1], end - start))

                print(
                    "Rep: {} Cost: {:.4f} Trace: {:.4f} Acc: {:.4f} "
                    "Time: {:.2f}s".format(k, losses_val[-1],
                                           traces_val[-1],
                                           accuracies_val[-1], end - start))
                print()

        END = time.perf_counter()  # time all epochs

        print("It took in total {:.2f} minutes".format(
            (END - START) / 60))  # tell time

        # save all the losses and traces from all repetitions (for statistics)
        rep_loss.append(losses)
        rep_loss_val.append(losses_val)
        rep_trace.append(traces)
        rep_trace_val.append(traces_val)
        rep_acc.append(accuracies)
        rep_acc_val.append(accuracies_val)
        history_params.append(params_history)

        #  pickle the data just in case every repetition to not lose anything
        with open(res_dir, "wb") as output:
            pickle.dump((rep_loss,
                         rep_loss_val,
                         rep_acc,
                         rep_acc_val,
                         history_params,
                         rep_trace,
                         rep_trace_val),
                        output)


if __name__ == '__main__':
    main()
