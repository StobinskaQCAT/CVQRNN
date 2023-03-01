import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.style as style
from matplotlib.ticker import ScalarFormatter

style.use('tableau-colorblind10')
plt.rcParams.update({'font.size': 13})

res_path = f"{Path(__file__).parent.parent.parent}/results/quantum"

avgs, stds = [], []
pred_all, fore_all = [], []

experiment = "sine_wave"
beta = [0.001, 0.01, 0.03, 0.1]

for i in beta:
    with open(f"{res_path}/{experiment}_modes_3_cut_6_lr_0.01"
              f"_inlen_4_outlen_1_datalen_200_gamma_10_layers_1"
              f"_nonlinear_0_noise_{i}_channel_loss_1.pkl",
              "rb") as data:
        run = pickle.load(data)

        losses_train_q, losses_val_q = run[0], run[1]
        func = run[4]
        pred, fore = run[5], run[6]

        avgs.append(np.mean(losses_train_q, axis=0)[-1])
        stds.append(np.std(losses_train_q, axis=0)[-1])

        pred_all.append(pred)
        fore_all.append(fore)

avgs = np.array(avgs)
stds = np.array(stds)

# if std deviation goes below zero try this:
l = 0.01
avg_bot = avgs - stds
avg_bot[stds >= l * avgs] = avgs[stds >= l * avgs] - l * stds[stds >= l * avgs]

# create the main axis
fig, main_ax = plt.subplots()

# plot mean and std
main_ax.plot(beta, avgs, linestyle="-", label="err")
main_ax.fill_between(beta, avgs - stds, avgs + stds, alpha=0.2)

# main_ax.set_xlabel("Signal noise ($\\alpha$)")
main_ax.set_xlabel("$\\varepsilon$")
main_ax.set_ylabel("$C~~~$", rotation=0)

main_ax.set_yscale('log')
main_ax.set_xscale('log')

# left arrow
main_ax.annotate('', xy=(1.1e-3, 3e-5), xycoords='data',
                 xytext=(3e-3, 1e-4), textcoords='data',
                 arrowprops=dict(linestyle="dotted", arrowstyle="->",
                                 connectionstyle="arc3"))

# right arrow
main_ax.annotate('', xy=(1e-1, 7e-5), xycoords='data',
                 xytext=(3.5e-2, 8.5e-5), textcoords='data',
                 arrowprops=dict(linestyle="dotted", arrowstyle="<-",
                                 connectionstyle="arc3"))

# right subplot
right_inset_ax = fig.add_axes([.7, .15, .3, .3])

right_inset_ax.tick_params(left=False,
                           bottom=False,
                           labelleft=False,
                           labelbottom=False)

right_inset_ax.spines["right"].set_linestyle("dotted")
right_inset_ax.spines["left"].set_linestyle("dotted")
right_inset_ax.spines["top"].set_linestyle("dotted")
right_inset_ax.spines["bottom"].set_linestyle("dotted")

right_inset_ax.plot(func, ".-",
                    markersize=2, alpha=0.7, lw=0.5)
right_inset_ax.plot(range(4, len(func)), pred_all[2][-1], ".-",
                    markersize=2, alpha=0.5, lw=0.5)
right_inset_ax.plot(range(90, 90 + len(fore_all[3][-1])), fore_all[3][-1],
                    ".-", markersize=2, alpha=0.5, lw=0.5)

right_inset_ax.axvline(90, color="black", ls="--", alpha=0.5, linewidth=1)

# left subplot
left_inset_ax = fig.add_axes([.2, .45, .3, .3])

left_inset_ax.tick_params(left=False,
                          bottom=False,
                          labelleft=False,
                          labelbottom=False)

left_inset_ax.spines["right"].set_linestyle("dotted")
left_inset_ax.spines["left"].set_linestyle("dotted")
left_inset_ax.spines["top"].set_linestyle("dotted")
left_inset_ax.spines["bottom"].set_linestyle("dotted")

left_inset_ax.plot(func, ".-",
                   markersize=2, alpha=0.7, lw=0.5)
left_inset_ax.plot(range(4, len(func)), pred_all[0][-1], ".-",
                   markersize=2, alpha=0.5, lw=0.5)
left_inset_ax.plot(range(90, 90 + len(fore_all[0][-1])), fore_all[0][-1], ".-",
                   markersize=2, alpha=0.5, lw=0.5)

left_inset_ax.axvline(90, color="black", ls="--", alpha=0.5, linewidth=1)

# enable scientific notation
# plt.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0),
#                      useOffset=False, useMathText=True)

main_ax.set_yticks([3e-5, 10e-5, 30e-5])
main_ax.get_yaxis().set_major_formatter(ScalarFormatter())
main_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# plt.tight_layout()
plt.savefig(f"../comparison/{experiment}_data_noise_dependence.pdf")
# plt.show()
