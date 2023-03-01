import pickle
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import os
import sys
import io

sys.path.append('../../QRNN/utils')

style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 13})

#####

cur_path = os.path.dirname(__file__)
res_path_c = os.path.abspath('../../results/classical')
res_path_q = os.path.abspath('../../results/quantum')

losses_train_q_all, losses_val_q_all = [], []
accuracy_train_q_all, accuracy_val_q_all = [], []

losses_train_c_all, losses_val_c_all = [], []
accuracy_train_c_all, accuracy_val_c_all = [], []

name = f"{res_path_c}/MNIST_units_4_lr_0.001_inlen_4_outlen_1_datalen_7_noise_0.pkl"
with open(name, "rb") as data:
    run = pickle.load(data)

    losses_train_c, losses_val_c = run[0], run[1]
    accuracy_train_c, accuracy_val_c = run[2], run[3]

    losses_train_c_all.append(losses_train_c)
    losses_val_c_all.append(losses_val_c)
    accuracy_train_c_all.append(accuracy_train_c)
    accuracy_val_c_all.append(accuracy_val_c)

for i in [1, 2, 3]:
    name = f"{res_path_q}/MNIST_modes_3_cut_6_lr_0.005" \
           f"_datalen_7_gamma_10_layers_1" \
           f"_nonlinear_0_noise_0_channel_loss_1_{i}.pkl"

    with open(name, "rb") as data:
        run = pickle.load(data)

        losses_train_q, losses_val_q = run[0], run[1]
        accuracy_train_q, accuracy_val_q = run[2], run[3]

        losses_train_q_all.append(losses_train_q)
        losses_val_q_all.append(losses_val_q)
        accuracy_train_q_all.append(accuracy_train_q)
        accuracy_val_q_all.append(accuracy_val_q)

accuracy_train_q_all = np.transpose(np.array(accuracy_train_q_all)[:, 0, :])
accuracy_val_q_all = np.transpose(np.array(accuracy_val_q_all)[:, 0, :])

print(accuracy_train_q_all)

losses_train_c_all = np.transpose(losses_train_c_all[0])[0]
losses_train_c_all = losses_train_c_all[:, np.any(losses_train_c_all < 4, axis=0)]

losses_val_c_all = np.transpose(losses_val_c_all[0])[0]
losses_val_c_all = losses_val_c_all[:, np.any(losses_val_c_all < 4, axis=0)]

accuracy_train_c_all = np.transpose(accuracy_train_c_all[0])
accuracy_train_c_all = accuracy_train_c_all[:, np.all(accuracy_train_c_all != 0.5075, axis=0)]

accuracy_val_c_all = np.transpose(accuracy_val_c_all[0])
accuracy_val_c_all = accuracy_val_c_all[:, np.all(accuracy_val_c_all != 0.5075, axis=0)]

avg_loss_val_c = \
    np.reshape(np.average(np.array(losses_val_c_all), axis=1), (-1))
std_loss_val_c = \
    np.reshape(np.std(np.array(losses_val_c_all), axis=1), (-1))

avg_acc_val_c = \
    np.reshape(np.average(np.array(accuracy_val_c_all), axis=1), (-1))
std_acc_val_c = \
    np.reshape(np.std(np.array(accuracy_val_c_all), axis=1), (-1))

avg_loss_train_c = \
    np.reshape(np.average(np.array(losses_train_c_all), axis=1), (-1))
std_loss_train_c = \
    np.reshape(np.std(np.array(losses_train_c_all), axis=1), (-1))

avg_acc_train_c = \
    np.reshape(np.average(np.array(accuracy_train_c_all), axis=1), (-1))
std_acc_train_c = \
    np.reshape(np.std(np.array(accuracy_train_c_all), axis=1), (-1))

################

avg_loss_val_q = \
    np.reshape(np.average(np.array(losses_val_q_all), axis=1), (-1))
std_loss_val_q = \
    np.reshape(np.std(np.array(losses_val_q_all), axis=1), (-1))

avg_acc_val_q = \
    np.reshape(np.average(np.array(accuracy_val_q_all), axis=1), (-1))
std_acc_val_q = \
    np.reshape(np.std(np.array(accuracy_val_q_all), axis=1), (-1))

avg_loss_train_q = \
    np.reshape(np.average(np.array(losses_train_q_all), axis=1), (-1))
std_loss_train_q = \
    np.reshape(np.std(np.array(losses_train_q_all), axis=1), (-1))

avg_acc_train_q = \
    np.reshape(np.average(np.array(accuracy_train_q_all), axis=1), (-1))
std_acc_train_q = \
    np.reshape(np.std(np.array(accuracy_train_q_all), axis=1), (-1))

# if std deviation goes below zero try this:
# l = 0.01
# avg_bot = avg - std
# avg_bot[std >= l * avg] = avg[std >= l * avg] - l * std[std >= l * avg]

# create the main axis
fig, main_ax = plt.subplots()

main_ax.plot(avg_loss_train_q, label="Quantum training")
main_ax.fill_between(range(len(avg_loss_train_q)),
                     avg_loss_train_q - std_loss_train_q,
                     avg_loss_train_q + std_loss_train_q,
                     alpha=0.2)

main_ax.plot(avg_loss_train_c[:25], label="Classical training")
main_ax.fill_between(range(len(avg_loss_train_c[:25])),
                     avg_loss_train_c[:25] - std_loss_train_c[:25],
                     avg_loss_train_c[:25] + std_loss_train_c[:25],
                     alpha=0.2, )

main_ax.plot(avg_loss_val_q, label="Quantum testing")
main_ax.fill_between(range(len(avg_loss_val_q)),
                     avg_loss_val_q - std_loss_val_q,
                     avg_loss_val_q + std_loss_val_q,
                     alpha=0.2)

main_ax.plot(avg_loss_val_c[:25], label="Classical testing")
main_ax.fill_between(range(len(avg_loss_val_c[:25])),
                     avg_loss_val_c[:25] - std_loss_val_c[:25],
                     avg_loss_val_c[:25] + std_loss_val_c[:25],
                     alpha=0.2)

main_ax.set_xlabel("Epoch")
main_ax.set_ylabel("Loss")

plt.legend()
plt.tight_layout()
plt.savefig(f"../comparison/MNIST_loss.pdf")
# plt.show()

# create the main axis
fig, main_ax = plt.subplots()

main_ax.plot(100 * avg_acc_train_q, label="Quantum training")
main_ax.fill_between(range(len(avg_acc_train_q)),
                     100 * (avg_acc_train_q - 0.5 * std_acc_train_q),
                     100 * (avg_acc_train_q + 0.5 * std_acc_train_q),
                     alpha=0.2)

main_ax.plot(100 * avg_acc_train_c[:25], label="Classical training")
main_ax.fill_between(range(len(avg_acc_train_c[:25])),
                     100 * (avg_acc_train_c[:25] - 0.5 * std_acc_train_c[:25]),
                     100 * (avg_acc_train_c[:25] + 0.5 * std_acc_train_c[:25]),
                     alpha=0.2)

main_ax.plot(100 * avg_acc_val_q, label="Quantum testing")
main_ax.fill_between(range(len(avg_acc_val_q)),
                     100 * (avg_acc_val_q - 0.5 * std_acc_val_q),
                     100 * (avg_acc_val_q + 0.5 * std_acc_val_q),
                     alpha=0.2)

main_ax.plot(100 * avg_acc_val_c[:25], label="Classical testing")
main_ax.fill_between(range(len(avg_acc_val_c[:25])),
                     100 * (avg_acc_val_c[:25] - 0.5 * std_acc_val_c[:25]),
                     100 * (avg_acc_val_c[:25] + 0.5 * std_acc_val_c[:25]),
                     alpha=0.2)

# main_ax.plot(accuracy_train_all)
main_ax.set_xlabel("Epoch")
main_ax.set_ylabel("Accuracy [%]")

plt.legend()
plt.tight_layout()
plt.savefig(f"../comparison/MNIST_accuracy.pdf")
# plt.show()
