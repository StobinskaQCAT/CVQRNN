# Script to recreate subfigures in Fig. 3. It plots the prediction
# of the train network (for chosen epoch) with forecast and
# original function. Works for both classical and quantum RNN.

import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.style as style

plt.rcParams.update({'font.size': 13})
style.use('tableau-colorblind10')

# name of the experiment 
experiment = "triangle_wave"
# classical or quantum experiment
c_or_q = "q"
# specify the epoch for which will be plotted
epoch = 5

for c_or_q in ["c", "q"]:
    for epoch in [5, 50]:
        # choose the proper file (if its classical or quantum)
        if c_or_q == "q":
            f_name = f"{Path(__file__).parent.parent.parent}" \
                     f"/results/quantum/{experiment}" \
                     f"_modes_3_cut_6_lr_0.01" \
                     f"_inlen_4_outlen_1_datalen_200_gamma_10_layers_1" \
                     f"_nonlinear_0_noise_0_channel_loss_1"
        elif c_or_q == "c":
            f_name = f"{Path(__file__).parent.parent.parent}" \
                     f"/results/classical/{experiment}" \
                     f"_units_4_lr_0.01" \
                     f"_inlen_4_outlen_1_datalen_200_noise_0"

        with open(f"{f_name}.pkl", "rb") as data:
            run = pickle.load(data)
            function = run[4]
            pred, fore = run[5], run[6]

        # create the main axis
        fig, ax = plt.subplots()

        msize = 10

        # True data points
        ax.plot(function, ".-",
                markersize=msize, alpha=0.7, label="True data")

        # # True data line, with small alpha
        # ax.plot(function, "-", alpha=0.1)

        # Prediction for all the points
        ax.plot(range(4, len(function)), pred[epoch // 5], ".-",
                markersize=msize, alpha=0.5,
                label="Prediction")

        # Forecast for test set
        ax.plot(range(90, 90 + len(fore[epoch // 5])), fore[epoch // 5],
                ".-", markersize=msize, alpha=0.5,
                label="Forecast")

        # Separation between the train and test dataset
        ax.axvline(90, color="black", ls="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("$i$")
        ax.set_ylabel("$f(x_i)~~~~$", rotation=0)
        ax.set_title(f"Epoch {epoch}")

        if c_or_q == "q" and epoch == 5:
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'{Path(__file__).parent.parent}/comparison/' \
                    f'{experiment}_epoch_{epoch}_{c_or_q}.pdf')
    # plt.show()
