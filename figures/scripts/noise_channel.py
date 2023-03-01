import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.style as style

style.use('tableau-colorblind10')

res_path = f"{Path(__file__).parent.parent.parent}/results/quantum"
plt.rcParams.update({'font.size': 13})

avgs, stds = [], []
pred_all, fore_all = [], []

experiment = "sine_wave"
alpha = [1, 0.8, 0.6, 0.4, 0.2]

for i in alpha:
    with open(f"{res_path}/{experiment}_modes_3_cut_6_lr_0.01"
              f"_inlen_4_outlen_1_datalen_200_gamma_10_layers_1"
              f"_nonlinear_0_noise_0_channel_loss_{i}.pkl",
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

# transform from energy transitivity to channel loss
x = 1 - np.array(alpha)

# if std deviation goes below zero try this:
l = 0.01
avg_bot = avgs - stds
avg_bot[stds >= l * avgs] = avgs[stds >= l * avgs] - l * stds[stds >= l * avgs]

# create the main axis
fig, main_ax = plt.subplots()

# plot mean and std
main_ax.plot(x, avgs, linestyle="-", label="err")
main_ax.fill_between(x, avgs - stds, avgs + stds, alpha=0.2)

# main_ax.set_xlabel("Signal noise ($\\alpha$)")
main_ax.set_xlabel("$\\beta$")
main_ax.set_ylabel("$C~~~~$", rotation=0)

# main_ax.set_yscale('log')
# main_ax.set_xscale('log')

# # left arrow
main_ax.annotate('', xy=(0.1, 4e-4), xycoords='data',
            xytext=(0, 4e-5), textcoords='data',
            arrowprops=dict(linestyle="dotted", arrowstyle="<-",
                connectionstyle="arc3"))

# main_ax.annotate('', xy=(0.1, 1e-4), xycoords='data',
#             xytext=(0, 2e-5), textcoords='data',
#             arrowprops=dict(linestyle="dotted", arrowstyle="<-",
#                 connectionstyle="arc3"))

# right arrow
main_ax.annotate('', xy=(0.45, 6e-4), xycoords='data',
            xytext=(0.4, 1e-4), textcoords='data',
            arrowprops=dict(linestyle="dotted", arrowstyle="<-",
                connectionstyle="arc3"))

# main_ax.annotate('', xy=(0.45, 2.5e-4), xycoords='data',
#             xytext=(0.4, 0.5e-4), textcoords='data',
#             arrowprops=dict(linestyle="dotted", arrowstyle="<-",
#                 connectionstyle="arc3"))

# right subplot
right_inset_ax = fig.add_axes([.4, .61, .3, .3])

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
right_inset_ax.plot(range(90, 90 + len(fore_all[2][-1])), fore_all[2][-1], ".-",
                    markersize=2, alpha=0.5, lw=0.5)

right_inset_ax.axvline(90, color="black", ls="--", alpha=0.5, linewidth=1)

# left subplot
left_inset_ax = fig.add_axes([.15, .3, .3, .3])

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
# left_inset_ax.plot(func, "-",
#                     alpha=0.2)
left_inset_ax.plot(range(4, len(func)), pred_all[0][-1], ".-",
                   markersize=2, alpha=0.5, lw=0.5)
left_inset_ax.plot(range(90, 90 + len(fore_all[0][-1])), fore_all[0][-1], ".-",
                   markersize=2, alpha=0.5, lw=0.5)

left_inset_ax.axvline(90, color="black", ls="--", alpha=0.5, linewidth=1)

# enable scientific notation
main_ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0),
                     useOffset=False, useMathText=True)

# plt.tight_layout()
plt.savefig(f"../comparison/{experiment}_channel_noise_dependence.pdf")
# plt.show()
