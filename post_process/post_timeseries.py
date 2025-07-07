from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})

CONFIG = {
    "data_path": Path("/Users/leo/python/thesis/sync_results/"),
    "lookup_csv": Path("/Users/leo/python/thesis/post_process/HAWC2_Output_Channels.csv"),
    "fatigue_columns": {0: "del_1e+07_3", 1: "del_1e+07_10"},
}

channels = [10, 13, 15, 17, 26, 44]
names = [
    "Omega [rad/s]",
    "Ae rot. thrust [kN]",
    "WSP gl. coo.,Vy [m/s]",
    "Mx coo: tower [kNm]",
    "Mx coo: blade1 [kNm]",
    "Mx coo: blade2 [kNm]",
    "Mx coo: blade3 [kNm]",
    "tower top fa displ [m]",
    "tower top fa acc [m/s^2]",
    "blade 1 tip pos x [m]"
]

channel_dict = dict(zip(channels, names))

def main():
    # Construct full filepath using Path
    filename_wave2 = CONFIG["data_path"] / "TimeSeries_wave2_study_ws_10.0_ti_0.16_n_0.06_a_2.0_seed_1.txt"
    filename_cnbl = CONFIG["data_path"] / "TimeSeries_cnbl_study_ws_10.0_ti_0.16_h_150.0_w_30.0_s_8_seed_1.txt"
    filename_alpha = CONFIG["data_path"] / "TimeSeries_alpha_study_ws_10_ti_0.16_shear_0.2_seed_1.txt"

    # Load the data assuming tab-separated values and first line as header
    dfwave2 = pd.read_csv(filename_wave2, sep="\t")
    dfcnbl = pd.read_csv(filename_cnbl, sep="\t")
    dfalpha = pd.read_csv(filename_alpha, sep="\t")

    # Show first rows to verify
    print(dfwave2.head())
    print(dfcnbl.head())
    print(dfalpha.head())

    c_lst = ["cornflowerblue", "darkorange", "mediumseagreen", "#d62728"]

    window_size = 50  # smoothing window

    for ch in channels:
        ch_str = str(ch)
        if ch_str not in dfwave2.columns or ch_str not in dfcnbl.columns or ch_str not in dfalpha.columns:
            print(f"Warning: Channel {ch} not found in all DataFrames columns, skipping.")
            continue
        if "time" not in dfwave2.columns or "time" not in dfcnbl.columns or "time" not in dfalpha.columns:
            print("Warning: 'time' column missing in one or more DataFrames, cannot plot with time axis.")
            break

        plt.figure(figsize=(12, 6))

        mean_alpha = np.mean(dfalpha[ch_str])
        std_alpha = np.std(dfalpha[ch_str])
        lower_bound = mean_alpha - std_alpha
        upper_bound = mean_alpha + std_alpha

        """
        # Fill between mean-sigma and mean+sigma for alpha study in background
        plt.fill_between(
            dfwave2["time"],  # x-axis (time, same length as wave studies)
            lower_bound,
            upper_bound,
            color=c_lst[2],
            alpha=0.3
        )
        plt.axhline(lower_bound, color=c_lst[2], linestyle="--", label=r"$\alpha-study$ mean$\pm$σ")
        plt.axhline(upper_bound, color=c_lst[2], linestyle="--")
        """

        # Plot alpha study
        plt.plot(dfalpha["time"], dfalpha[ch_str], color=c_lst[2], label=r"$\alpha-study$ $U=10.0$ [m/s]; $TI=0.16$ [-]; $\alpha=0.2$ [-]")
        # Plot wave study
        #plt.plot(dfwave2["time"], dfwave2[ch_str], color=c_lst[0], label=r"$wave-study$ $U=10.0$ [m/s]; $TI=0.16$ [-]; $N=0.06$ [Hz] and $A=2.0$ [m/s]; Seed=1")
        # Plot cnbl study
        plt.plot(dfcnbl["time"], dfcnbl[ch_str], color=c_lst[0], label=r"$jet-study$ $U=10.0$ [m/s]; $TI=0.16$ [-]; $H$=150 [m]; $W$=30 [m]; $S$=8 [m/s]; Seed=1")

        #plt.title(fr"Time Series Comparison for Channel {ch}: {channel_dict[ch]}")
        plt.xlabel("Time [s]")
        plt.ylabel(fr"${channel_dict[ch]}$")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        plt.grid(True)
        plt.tight_layout() #rect=[0, 0.05, 1, 1]

    plt.show()



if __name__ == "__main__":
    main()
