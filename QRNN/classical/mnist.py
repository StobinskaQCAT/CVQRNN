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
    rep_loss, rep_loss_val = [], []
    history_params = []
    rep_acc_train, rep_acc_test = [], []

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
    (x_train, y_train), (x_val, y_val) = \
        dataLoader.mnist_data(resolution, data_len, 1)
    x_train = np.expand_dims(x_train, axis=2)
    y_train = np.expand_dims(y_train, axis=(1, 2))
    x_val = np.expand_dims(x_val, axis=2)
    y_val = np.expand_dims(y_val, axis=(1, 2))

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

        model.compile(loss=tf.losses.BinaryCrossentropy(),
                      optimizer=tf.optimizers.Adam(learning_rate=lr),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        ##########################################
        # Training
        ##########################################

        # for keeping track of losses in one run
        prediction_loss_mse, loss, val_loss = [], [], []
        acc_train, acc_test = [], []
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

            ##############################
            # Accuracy
            ##############################

            accuracy_train = np.mean((tf.math.sigmoid(model(x_train)) > 0.5) == y_train)
            accuracy_test = np.mean((tf.math.sigmoid(model(x_val)) > 0.5) == y_val)

            print(f"accuracy_train = {accuracy_train}")
            print(f"accuracy_test = {accuracy_test}")

            ############################
            # Saving parameters
            ############################

            # save loss for training
            loss.append(history.history['loss'])
            # ... and validation
            val_loss.append(history.history['val_loss'])
            acc_train.append(accuracy_train)
            acc_test.append(accuracy_test)

            # write to tensorboard
            with summary_writer.as_default():
                tf.summary.scalar('training loss',
                                  history.history['loss'][0],
                                  step=k)
                tf.summary.scalar('validation loss',
                                  history.history['val_loss'][0],
                                  step=k)

        # keep track of all losses from every run
        rep_loss.append(loss)
        rep_loss_val.append(val_loss)
        rep_acc_train.append(acc_train)
        rep_acc_test.append(acc_test)
        history_params.append(params_history)

        # pickle losses every run so even if it breaks in the middle we
        # still have some data
        with open(res_dir, "wb") as output:
            pickle.dump((
                rep_loss,  # 0 - train loss
                rep_loss_val,  # 1 - test loss
                rep_acc_train,  # 2 - accuracy train
                rep_acc_test,  # 3 - accuracy on test
                history_params),  # 4 - history parameters
                output)


if __name__ == '__main__':
    main()
