# Script to recreate the Fig.5 -- dependance of final loss on 
# the input data length

import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from pathlib import Path
import matplotlib.style as style
from matplotlib.ticker import ScalarFormatter

style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 13})

avgs, stds, med = [], [], []
mi, ma = [], []

# experiment to analize
experiment = "bessel"
c_or_q = "q"


def name(i):
    if c_or_q == "q":
        return f"{Path(__file__).parent.parent.parent}/results" \
               f"/quantum/{experiment}" \
               f"_modes_3_cut_6_lr_0.01_inlen_{i}_outlen_1" \
               f"_datalen_200_gamma_10_layers_1_" \
               f"nonlinear_0_noise_0_channel_loss_1.pkl"

    if c_or_q == "c":
        return f"{Path(__file__).parent.parent.parent}/results" \
               f"/classical/{experiment}" \
               f"_units_4_lr_0.01_inlen_{i}_outlen_1" \
               f"_datalen_200_noise_0.pkl"


in_len = [2, 4, 6, 8, 12, 16, 20]

for i in in_len:
    with open(name(i), "rb") as data:
        run = pickle.load(data)

        losses_train_q, losses_val_q = run[0], run[1]

        avgs.append(np.mean(losses_train_q, axis=0)[-1])
        mi.append(np.min(losses_train_q, axis=0)[-1])
        ma.append(np.max(losses_train_q, axis=0)[-1])

avgs = np.array(avgs)
mi = np.array(mi)
ma = np.array(ma)

# create the main axis
fig, main_ax = plt.subplots()

# plot mean and std
main_ax.plot(in_len, avgs, "-", label="Mean")
main_ax.fill_between(in_len, mi, ma, alpha=0.2)

main_ax.set_xlabel("$T$")
main_ax.set_ylabel("$C~~~~$", rotation=0)

main_ax.set_yscale('log')

main_ax.set_yticks([1e-5, 3e-5, 10e-5, 30e-5])
main_ax.get_yaxis().set_major_formatter(ScalarFormatter())
main_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

main_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
if c_or_q == "c":
    plt.savefig(f"../comparison/{experiment}_in_len_dependence_classical.pdf")
if c_or_q == "q":
    plt.savefig(f"../comparison/{experiment}_in_len_dependence_quantum.pdf")
# plt.show()
