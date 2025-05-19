import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

def plot_channelstats(stats_df, iec_stats_df):
    # Blade Figures
    blade_channels = [26, 29, 32]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    box_colors = ["lavender", "cornflowerblue"]
    labels = ["IEC", "Case"]

    for idx, channel_number in enumerate(blade_channels):
        if channel_number not in stats_df["channel"].values:
            print(f"Channel '{channel_number}' not found. Skipping.")
            continue

        iec_data = iec_stats_df[iec_stats_df["channel"] == channel_number].iloc[0]
        data = stats_df[stats_df["channel"] == channel_number].iloc[0]
        variable_name = data["variable"]
        unit = data["unit"]

        iec_box_stats = {
            'med': iec_data["quant_0.5"],
            'q1': iec_data["quant_0.25"],
            'q3': iec_data["quant_0.75"],
            'whislo': iec_data["quant_0.01"],
            'whishi': iec_data["quant_0.99"],
            'fliers': [],
            'label': 'IEC'
        }

        box_stats = {
            'med': data["quant_0.5"],
            'q1': data["quant_0.25"],
            'q3': data["quant_0.75"],
            'whislo': data["quant_0.01"],
            'whishi': data["quant_0.99"],
            'fliers': [],
            'label': 'Case'
        }

        ax = axes[idx]
        
        bxp1 = ax.bxp([iec_box_stats], positions=[1], widths=0.6, showfliers=False, patch_artist=True)
        bxp2 = ax.bxp([box_stats], positions=[2], widths=0.6, showfliers=False, patch_artist=True)

        bxp1["boxes"][0].set_facecolor(box_colors[0])
        bxp2["boxes"][0].set_facecolor(box_colors[1])

        ax.set_title(f"{variable_name}\nChannel {channel_number}")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel(f"Load [{unit}]")
        ax.grid(True)

        fatigue_col = "del_1e+07_10"
        if fatigue_col not in data:
            print(f"Fatigue column '{fatigue_col}' not found in data.")
            continue

        ax.text(1.5, ax.get_ylim()[0] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"{fatigue_col}\nIEC:{iec_data[fatigue_col]:.2f}\nCase:{data[fatigue_col]:.2f}",
                ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(box_colors, labels)]
    fig.legend(handles=legend_patches, loc="upper center", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Tower figure
    tower_channel = 17
    fig, ax = plt.subplots()

    iec_data = iec_stats_df[iec_stats_df["channel"] == tower_channel].iloc[0]
    data = stats_df[stats_df["channel"] == tower_channel].iloc[0]
    variable_name = data["variable"]
    unit = data["unit"]

    iec_box_stats = {
        'med': iec_data["quant_0.5"],
        'q1': iec_data["quant_0.25"],
        'q3': iec_data["quant_0.75"],
        'whislo': iec_data["quant_0.01"],
        'whishi': iec_data["quant_0.99"],
        'fliers': [],
        'label': 'IEC'
    }

    box_stats = {
        'med': data["quant_0.5"],
        'q1': data["quant_0.25"],
        'q3': data["quant_0.75"],
        'whislo': data["quant_0.01"],
        'whishi': data["quant_0.99"],
        'fliers': [],
        'label': 'Case'
    }
    
    bxp1 = ax.bxp([iec_box_stats], positions=[1], widths=0.6, showfliers=False, patch_artist=True)
    bxp2 = ax.bxp([box_stats], positions=[2], widths=0.6, showfliers=False, patch_artist=True)

    bxp1["boxes"][0].set_facecolor(box_colors[0])
    bxp2["boxes"][0].set_facecolor(box_colors[1])

    ax.set_title(f"{variable_name}\nChannel {tower_channel}")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Load [{unit}]")
    ax.grid(True)

    fatigue_col = "del_1e+07_3"
    if fatigue_col not in data:
        print(f"Fatigue column '{fatigue_col}' not found in data.")

    ax.text(1.5, ax.get_ylim()[0] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"{fatigue_col}\nIEC:{iec_data[fatigue_col]:.2f}\nCase:{data[fatigue_col]:.2f}",
            ha="center", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(box_colors, labels)]
    fig.legend(handles=legend_patches, loc="upper center", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

def plot_channel_ws_overview(stats_df, iec_stats_df, channel_number):
    iec_channel_df = iec_stats_df[iec_stats_df["channel"] == channel_number]
    channel_df = stats_df[stats_df["channel"] == channel_number]
    variable = channel_df["variable"].iloc[0]

    if channel_df.empty:
        print(f"Channel '{channel_number}' not found in data.")
        return

    iec_channel_df = iec_channel_df.sort_values("ws")
    channel_df = channel_df.sort_values("ws")

    iec_ws, ws = iec_channel_df["ws"], channel_df["ws"]
    iec_mean, mean = iec_channel_df["mean"], channel_df["mean"]
    iec_stdev, stdev = iec_channel_df["stdev"], channel_df["stdev"]
    iec_p90, p90 = iec_channel_df["quant_0.9"], channel_df["quant_0.9"]
    iec_p99, p99 = iec_channel_df["quant_0.99"], channel_df["quant_0.99"]

    # Figure for mean and stdev
    plt.figure()

    # Shaded region for mean ± stdev
    plt.plot(iec_ws, iec_mean - iec_stdev, color="grey")
    plt.plot(iec_ws, iec_mean + iec_stdev, color="grey")
    plt.fill_between(iec_ws, iec_mean - iec_stdev, iec_mean + iec_stdev, color="grey", alpha=0.3, label=f"IEC Mean ±1 Stdev - Channel {channel_number}")
    plt.plot(ws, mean - stdev, color="cornflowerblue")
    plt.plot(ws, mean + stdev, color="cornflowerblue")
    plt.fill_between(ws, mean - stdev, mean + stdev, color="cornflowerblue", alpha=0.3, label=f"Case Mean ±1 Stdev - Channel {channel_number}")

    plt.title(f"{variable}\n Channel {channel_number}")
    plt.xlabel("Wind Speed [m/s]")
    plt.ylabel("Mean Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Figure for P90/P99
    plt.figure()

    # Shaded region between P90 and P99
    plt.plot(iec_ws, iec_p90, color="grey")
    plt.plot(iec_ws, iec_p99, color="grey")
    plt.fill_between(iec_ws, iec_p90, iec_p99, color="grey", alpha=0.3, label=f"IEC P90/99 - Channel {channel_number}")
    plt.plot(ws, p90, color="cornflowerblue")
    plt.plot(ws, p99, color="cornflowerblue")
    plt.fill_between(ws, p90, p99, color="cornflowerblue", alpha=0.3, label=f"Case P90/99 - Channel {channel_number}")

    plt.title(f"{variable}\n Channel {channel_number}")
    plt.xlabel("Wind Speed [m/s]")
    plt.ylabel("Mean Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def _plot_single_channel(ax, df, iec_df, channel_number):
    # Filter data for the channel
    channel_df = df[df["channel"] == channel_number]
    iec_channel_df = iec_df[iec_df["channel"] == channel_number]

    for _, row in iec_channel_df.iterrows():
        pdf_array = row["pdf"]
        mids = pdf_array[:, 1]
        probs = pdf_array[:, 3]
        label_parts = [f"IEC {col}={row[col]}" for col in iec_channel_df.columns if col not in ["pdf", "channel"]]
        ax.plot(mids, probs, label=", ".join(label_parts), color="grey")

    for _, row in channel_df.iterrows():
        pdf_array = row["pdf"]
        mids = pdf_array[:, 1]
        probs = pdf_array[:, 3]
        label_parts = [f"{col}={row[col]}" for col in channel_df.columns if col not in ["pdf", "channel"]]
        ax.plot(mids, probs, label=", ".join(label_parts))

    ax.set_title(f"PDFs for channel {channel_number}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.grid()
    ax.legend(title="Parameters", loc='center left', bbox_to_anchor=(1, 0.5))


def plot_selected_channel_pdfs(pdf_df, iec_pdf_df):
    if "channel" not in pdf_df.columns:
        print("The DataFrame does not contain a 'channel' column.")
        return

    default_channels = [17, 26, 29, 32]

    # Plot channel 17 alone
    channel_17 = default_channels[0]
    if channel_17 in pdf_df["channel"].values:
        fig, ax = plt.subplots(figsize=(12, 6))
        _plot_single_channel(ax, pdf_df, iec_pdf_df, channel_17)
        plt.tight_layout()
    else:
        print(f"Channel {channel_17} not found in pdf data.")

    # Plot channels 26, 29, 32 in vertical subplots
    remaining_channels = default_channels[1:]
    available_channels = [ch for ch in remaining_channels if ch in pdf_df["channel"].values]

    if available_channels:
        fig, axes = plt.subplots(nrows=len(available_channels), ncols=1, figsize=(12, 3 * len(available_channels)))
        if len(available_channels) == 1:
            axes = [axes]  # Ensure it's iterable
        for ax, ch in zip(axes, available_channels):
            _plot_single_channel(ax, pdf_df, iec_pdf_df, ch)
        plt.tight_layout()
    else:
        print("None of the remaining channels (26, 29, 32) were found in pdf data.")

def plot_other_channel(pdf_df, iec_pdf_df, channel_number):
    if channel_number in pdf_df["channel"].values:
        fig, ax = plt.subplots(figsize=(12, 6))
        _plot_single_channel(ax, pdf_df, iec_pdf_df, channel_number)
        plt.tight_layout()
    else:
        print(f"Channel {channel_number} not found in pdf data.")

def plot_joint_probs(stats_df, s_choice, metric="quant_0.9"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    channels = [17, 26, 29, 32]
    
    for idx, channel in enumerate(channels):
        ax = axes[idx // 2, idx % 2]  # position in 2x2 grid
        ch_df = stats_df[(stats_df["channel"] == channel) & (stats_df["s"] == s_choice)]
        
        if ch_df.empty:
            ax.set_title(f"Channel {channel} (no data)")
            ax.axis("off")
            continue
        
        load_map = ch_df.pivot_table(index="h", columns="w", values=metric)
        sns.heatmap(load_map, cmap="viridis", ax=ax, cbar=True)
        
        ax.set_title(f"Channel {channel}")
        ax.set_xlabel("w")
        ax.set_ylabel("h")

    fig.suptitle(f"Heat Maps ({metric}) for s = {s_choice}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

def plot_joint_probs2(stats_df, metric="quant_0.9"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    channels = [17, 26, 29, 32]
    
    for idx, channel in enumerate(channels):
        ax = axes[idx // 2, idx % 2]  # position in 2x2 grid
        ch_df = stats_df[(stats_df["channel"] == channel)]
        ch_df = ch_df.drop("w", axis=1)

        if ch_df.empty:
            ax.set_title(f"Channel {channel} (no data)")
            ax.axis("off")
            continue
        
        load_map = ch_df.pivot_table(index="h", columns="s", values=metric, aggfunc="mean")
        sns.heatmap(load_map, cmap="viridis", ax=ax, cbar=True)
        
        ax.set_title(f"Channel {channel}")
        ax.set_xlabel("s")
        ax.set_ylabel("h")

    fig.suptitle(f"Heat Maps ({metric})")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

