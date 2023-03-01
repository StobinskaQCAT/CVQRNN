# Script to recreate Fig. 4 from the paper -- comparison of the losses between
# quantum and classical network with additional line which corresponds to the
# naive prediction

import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from plot_avg_std import plot_avg_std

plt.rcParams.update({'font.size': 13})

# name of the experiment to analyze
experiment = "cos_wave_damped"

# the name of the figure
name = f"{experiment}_loss"

# read quantum runs
with open(f"{Path(__file__).parent.parent.parent}/results/quantum/{experiment}"
          f"_modes_3_cut_6_lr_0.01_inlen_4_outlen_1"
          f"_datalen_200_gamma_10_layers_1_"
          f"nonlinear_0_noise_0_channel_loss_1.pkl",
          "rb") as data:
    run = pickle.load(data)

    losses_train_q, losses_val_q = run[0], run[1]
    naive = run[9]

# read classical runs
with open(f"{Path(__file__).parent.parent.parent}/results/classical/"
          f"{experiment}_units_4_lr_0.01_inlen_4_outlen_1"
          f"_datalen_200_noise_0.pkl", "rb") as data:
    run = pickle.load(data)

    losses_train_c, losses_val_c = run[0], run[1]

################################################################################

# read another quantum runs
# with open(f"{Path(__file__).parent.parent.parent}/results/quantum/{experiment}"
#           f"_modes_3_cut_6_lr_0.01_inlen_4_outlen_1"
#           f"_datalen_200_gamma_10_layers_1_"
#           f"nonlinear_1_noise_0_channel_loss_1.pkl",
#           "rb") as data:
#     run = pickle.load(data)

#     losses_train_q_1, losses_val_q_1 = run[0], run[1]
#
# with open(f"{Path(__file__).parent.parent.parent}/results/quantum/{experiment}"
#           f"_modes_3_cut_6_lr_0.01_inlen_4_outlen_1"
#           f"_datalen_200_gamma_10_layers_1_"
#           f"nonlinear_2_noise_0_channel_loss_1.pkl",
#           "rb") as data:
#     run = pickle.load(data)
#
#     losses_train_q_2, losses_val_q_2 = run[0], run[1]
#
# with open(f"{Path(__file__).parent.parent.parent}/results/classical/"
#           f"{experiment}_units_6_lr_0.01_inlen_4_outlen_1"
#           f"_datalen_200_noise_0.pkl", "rb") as data:
#     run = pickle.load(data)
#
#     losses_train_c_1, losses_val_c_1 = run[0], run[1]

# for comparison with other classical models
# with open(f"../../results/classical/GRU_{experiment}_units"
#             f"_3_lr_0.01_inlen_4_outlen_1"
#             f"_datalen_100_noise_0.pkl", "rb") as data:
#     run = pickle.load(data)
#     losses_train_c_GRU, losses_val_c_GRU = run[0], run[1]

# with open(f"../../results/classical/RNN_{experiment}_units"
#             f"_3_lr_0.01_inlen_4_outlen_1"
#             f"_datalen_100_noise_0.pkl", "rb") as data:
#     run = pickle.load(data)
#     losses_train_c_RNN, losses_val_c_RNN = run[0], run[1]

########################
# plot losses
#######################

# training
plot_avg_std(losses_train_q, "Quantum training")
plot_avg_std(losses_train_c, "Classical training")
# plot_avg_std(losses_val_q_1, "Quantum train, non 0")
# plot_avg_std(losses_val_q_1, "Quantum train, non 2")
# plot_avg_std(losses_val_c_GRU, "Classical GRU")
# plot_avg_std(losses_val_c_RNN, "Classical RNN")

# testing
plot_avg_std(losses_val_q, "Quantum testing")
# plot_avg_std(losses_val_q_1, "Quantum 1")
# plot_avg_std(losses_val_q_2, "Quantum 2")
plot_avg_std(losses_val_c, "Classical testing")
# plot_avg_std(losses_val_c_1, "Classical 5")

# draw a line of a naive prediction
plt.axhline(naive, color="black", ls="--", alpha=0.5, label="Baseline")

plt.yscale('log')

plt.xlabel("Epoch")
plt.ylabel("$C~~~~$", rotation=0)
# plt.legend()
plt.tight_layout()
plt.savefig(f'{Path(__file__).parent.parent}/comparison/{name}.pdf')
plt.clf()
