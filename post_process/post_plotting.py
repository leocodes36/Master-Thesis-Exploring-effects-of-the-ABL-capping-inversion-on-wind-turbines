import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

def plot_channelstats(stats_df, iec_stats_df):
    channels = [17, 26, 29, 32]
    fig, axes = plt.subplots(1, len(channels), figsize=(4.5 * len(channels), 5), sharey=False)

    box_colors = ["lavender", "cornflowerblue"]
    labels = ["IEC", "Case"]

    excluded_cols = ["channel", "variable", "unit", "mean", "stdev",
                     "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75",
                     "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in stats_df.columns if col not in excluded_cols]

    # Create one twinx axis for the last three subplots, based on the second subplot
    right_ax = axes[3].twinx()

    for idx, channel_number in enumerate(channels):
        ax = axes[idx]

        if channel_number not in stats_df["channel"].values:
            ax.set_title(f"Channel {channel_number} not found")
            ax.axis("off")
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

        case_box_stats = {
            'med': data["quant_0.5"],
            'q1': data["quant_0.25"],
            'q3': data["quant_0.75"],
            'whislo': data["quant_0.01"],
            'whishi': data["quant_0.99"],
            'fliers': [],
            'label': 'Case'
        }

        bxp1 = ax.bxp([iec_box_stats], positions=[1], widths=0.6, showfliers=False, patch_artist=True)
        bxp2 = ax.bxp([case_box_stats], positions=[2], widths=0.6, showfliers=False, patch_artist=True)

        bxp1["boxes"][0].set_facecolor(box_colors[0])
        bxp2["boxes"][0].set_facecolor(box_colors[1])

        ax.set_title(f"{variable_name}\nChannel {channel_number}")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels)
        ax.grid(True)

        if idx == 0:
            # First subplot: normal left y-axis with ticks and label
            ax.set_ylabel(f"Load [{unit}]")
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
        else:
            # Last three subplots: hide left y-axis ticks and labels
            ax.tick_params(left=False, labelleft=False)

            # Plotting limits and ticks for last three subplots will be controlled by right_ax
            # So set limits for ax to match those of right_ax later

    # Now configure the shared right y-axis for the last three plots:
    # Set limits and labels based on combined data range (optional - or you can set fixed limits)
    # For simplicity, compute the min/max over last three channels in stats_df & iec_stats_df

    last_three_channels = channels[1:]
    vals = []

    for ch in last_three_channels:
        d = stats_df[stats_df["channel"] == ch].iloc[0]
        i = iec_stats_df[iec_stats_df["channel"] == ch].iloc[0]
        vals.extend([d["quant_0.01"], d["quant_0.99"], i["quant_0.01"], i["quant_0.99"]])

    ymin, ymax = 1.1 * min(vals), 0.9 * max(vals)

    right_ax.set_ylim(ymin, ymax)
    right_ax.set_ylabel(f"Load [{unit}]")  # Use unit of last channel or make unified
    right_ax.yaxis.set_label_position("right")
    right_ax.yaxis.tick_right()

    # Sync limits on the last three subplots
    for ax in axes[1:]:
        ax.set_ylim(ymin, ymax)

    info_text = " | ".join(f"{col}: {data[col]:.2f}" for col in included_cols if col in data)

    fig.text(0.5, 0.0, f"Case Data – {info_text}", ha="center", va="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(box_colors, labels)]
    fig.legend(handles=legend_patches, loc="upper center", ncol=2)

    plt.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])

def plot_channel_ws_overview(stats_df, iec_stats_df, channel_number):
    iec_channel_df = iec_stats_df[iec_stats_df["channel"] == channel_number]
    channel_df = stats_df[stats_df["channel"] == channel_number]
    variable = channel_df["variable"].iloc[0]
    data = channel_df.iloc[0]

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

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # First subplot: mean ± stdev
    axs[0].plot(iec_ws, iec_mean - iec_stdev, color="grey", linestyle="--")
    axs[0].plot(iec_ws, iec_mean + iec_stdev, color="grey", linestyle="--")
    axs[0].fill_between(iec_ws, iec_mean - iec_stdev, iec_mean + iec_stdev, color="grey", alpha=0.3,
                        label=f"IEC Mean ±1 Stdev - Channel {channel_number}")
    axs[0].plot(ws, mean - stdev, color="cornflowerblue", linestyle="--")
    axs[0].plot(ws, mean + stdev, color="cornflowerblue", linestyle="--")
    axs[0].fill_between(ws, mean - stdev, mean + stdev, color="cornflowerblue", alpha=0.3,
                        label=f"Case Mean ±1 Stdev - Channel {channel_number}")

    axs[0].set_title(f"{variable}\nChannel {channel_number}")
    axs[0].set_xlabel("Nominal Wind Speed [m/s]")
    axs[0].set_ylabel("Load [kNm]")
    axs[0].grid(True)
    axs[0].legend()

    # Second subplot: P90 / P99
    axs[1].plot(iec_ws, iec_p90, color="grey", linestyle="--")
    axs[1].plot(iec_ws, iec_p99, color="grey", linestyle="--")
    axs[1].fill_between(iec_ws, iec_p90, iec_p99, color="grey", alpha=0.3,
                        label=f"IEC P90/99 - Channel {channel_number}")
    axs[1].plot(ws, p90, color="cornflowerblue", linestyle="--")
    axs[1].plot(ws, p99, color="cornflowerblue", linestyle="--")
    axs[1].fill_between(ws, p90, p99, color="cornflowerblue", alpha=0.3,
                        label=f"Case P90/99 - Channel {channel_number}")

    axs[1].set_title(f"{variable}\nChannel {channel_number}")
    axs[1].set_xlabel("Nominal Wind Speed [m/s]")
    # No ylabel here to avoid repetition
    axs[1].grid(True)
    axs[1].legend()

    # Info text once at bottom center
    excluded_cols = ["ws", "channel", "variable", "unit", "mean", "stdev", "quant_0.01", "quant_0.25",
                    "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in channel_df.columns if col not in excluded_cols]
    info_text = " | ".join(f"{col}: {data[col]:.2f}" for col in included_cols)

    fig.text(0.5, 0.02, f"Case Data – {info_text}", ha="center", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for the text

def _plot_single_channel(ax, df, iec_df, channel_number, show_legend=True, legend_loc='best'):
    # Filter data for the channel
    channel_df = df[df["channel"] == channel_number]
    iec_channel_df = iec_df[iec_df["channel"] == channel_number]

    for _, row in iec_channel_df.iterrows():
        pdf_array = row["pdf"]
        mids = pdf_array[:, 1]
        probs = pdf_array[:, 3]
        ax.plot(mids, probs, label="IEC", color="grey")

    for _, row in channel_df.iterrows():
        pdf_array = row["pdf"]
        mids = pdf_array[:, 1]
        probs = pdf_array[:, 3]
        ax.plot(mids, probs, label="Case", color="cornflowerblue")

    ax.set_title(f"PDFs for channel {channel_number}")
    ax.set_xlabel("Load [kNm]")
    ax.set_ylabel("Probability Density [-]")
    ax.grid()
    if show_legend:
        ax.legend(loc=legend_loc)

def plot_selected_channel_pdfs(pdf_df, iec_pdf_df):
    if "channel" not in pdf_df.columns:
        print("The DataFrame does not contain a 'channel' column.")
        return

    default_channels = [17, 26, 29, 32]

    # Info text once at bottom center
    data = pdf_df.iloc[0]
    excluded_cols = ["channel", "variable", "unit", "mean", "stdev", "quant_0.01", "quant_0.25",
                    "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10", "pdf"]
    included_cols = [col for col in pdf_df.columns if col not in excluded_cols]
    info_text = " | ".join(f"{col}: {data[col]:.2f}" for col in included_cols)

    # Plot channel 17 alone
    channel_17 = default_channels[0]
    if channel_17 in pdf_df["channel"].values:
        fig, ax = plt.subplots(figsize=(12, 6))
        _plot_single_channel(ax, pdf_df, iec_pdf_df, channel_17)
        fig.text(0.5, 0.02, f"Case Data – {info_text}", ha="center", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        print(f"Channel {channel_17} not found in pdf data.")

    # Plot channels 26, 29, 32 in vertical subplots
    remaining_channels = default_channels[1:]
    available_channels = [ch for ch in remaining_channels if ch in pdf_df["channel"].values]

    if available_channels:
        fig, axes = plt.subplots(nrows=len(available_channels), ncols=1, figsize=(12, 3 * len(available_channels)), sharex=True)
        if len(available_channels) == 1:
            axes = [axes]  # Ensure it's iterable
        for i, (ax, ch) in enumerate(zip(axes, available_channels)):
            _plot_single_channel(ax, pdf_df, iec_pdf_df, ch, show_legend=(i == 0))
        fig.text(0.5, 0.02, f"Case Data – {info_text}", ha="center", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
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

def plot_contours_sliders(stats_df, metric="quant_0.9"):
    heights = [150.0, 200.0, 250.0, 300.0]
    channel_values = [17, 26, 29, 32]
    ws_values = sorted(stats_df["ws"].unique())

    # Initial values
    init_channel = channel_values[0]
    init_ws = ws_values[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.85)
    axes = axes.flatten()

    # Shared colorbar setup (dummy to initialize)
    norm = plt.Normalize(vmin=stats_df[metric].min(), vmax=stats_df[metric].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(metric)

    def get_pivot(df, h):
        sub = df[df["h"] == h]
        if sub.empty:
            return None, None, None
        pivot = sub.pivot_table(index="s", columns="w", values=metric)
        if pivot.empty:
            return None, None, None
        X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
        Z = pivot.values
        return X, Y, Z

    contour_plots = []

    def update(val=None):
        nonlocal contour_plots

        ws = slider_ws.val
        channel = slider_channel.val

        df = stats_df[(stats_df["ws"] == ws) & (stats_df["channel"] == channel)]
        if df.empty:
            for ax in axes:
                ax.clear()
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
            fig.canvas.draw_idle()
            return

        vmin = df[metric].min()
        vmax = df[metric].max()
        for ax in axes:
            ax.clear()

        for i, h in enumerate(heights):
            ax = axes[i]
            X, Y, Z = get_pivot(df, h)
            if Z is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            cs = ax.contourf(X, Y, Z, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"Height: {h} m")
            ax.set_xlabel("w")
            ax.set_ylabel("s")

        # Update shared colorbar
        sm.set_clim(vmin, vmax)
        cbar.update_normal(sm)
        fig.suptitle(f"{metric} | ws: {ws}, channel: {channel}")
        fig.canvas.draw_idle()

    # Slider axes
    ax_ws = fig.add_axes([0.1, 0.15, 0.7, 0.03])
    ax_channel = fig.add_axes([0.1, 0.1, 0.7, 0.03])

    slider_ws = Slider(ax_ws, "Wind Speed (ws)", min(ws_values), max(ws_values), valinit=init_ws, valstep=ws_values)
    slider_channel = Slider(ax_channel, "Channel", min(channel_values), max(channel_values), valinit=init_channel, valstep=channel_values)

    slider_ws.on_changed(update)
    slider_channel.on_changed(update)

    update()  # Initial plot
    plt.show()

def plot_channelstats_sliders(stats_df, iec_stats_df):
    channels = [17, 26, 29, 32]

    box_colors = ["lavender", "cornflowerblue"]
    labels = ["IEC", "Case"]

    # Columns to filter by
    excluded_cols = ["channel", "variable", "unit", "mean", "stdev", "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in stats_df.columns if col not in excluded_cols]

    # Create main figure and axes
    fig, axes = plt.subplots(1, len(channels), figsize=(5 * len(channels), 5))
    fig.subplots_adjust(bottom=0.35)

    # Ensure axes is always a list
    if len(channels) == 1:
        axes = [axes]

    # Make the last three subplots share the y-axis
    for ax in axes[2:]:
        ax.get_shared_y_axes().joined(ax, axes[1])

    # Create axes for sliders
    slider_axes = []
    sliders = []
    for i, col in enumerate(included_cols):
        ax_slider = fig.add_axes([0.1, 0.25 - i * 0.05, 0.8, 0.03])
        slider = Slider(
            ax=ax_slider,
            label=col,
            valmin=stats_df[col].min(),
            valmax=stats_df[col].max(),
            valinit=stats_df[col].iloc[0],
            valstep=sorted(stats_df[col].unique())
        )
        sliders.append(slider)
    
    # Plot updating function
    def update(val=None):
        # Get current values from sliders
        filters = {col: slider.val for col, slider in zip(included_cols, sliders)}
        
        # Filter dataframes
        filtered_df = stats_df.copy()
        for k, v in filters.items():
            filtered_df = filtered_df[filtered_df[k] == v]

        filtered_iec_df = iec_stats_df.copy()
        filtered_iec_df = filtered_iec_df[filtered_iec_df["ws"] == filters["ws"]]
        
        # Check for empty filtered data
        if filtered_df.empty or filtered_iec_df.empty:
            for ax in axes:
                ax.clear()
                ax.text(0.5, 0.5, "No matching data for filters.", 
                        ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
            fig.canvas.draw_idle()
            return

        # Update each subplot
        for idx, channel in enumerate(channels):
            ax = axes[idx]
            ax.clear()

            if channel not in filtered_df["channel"].values:
                ax.set_title(f"Channel {channel} not found")
                continue

            case_data = filtered_df[filtered_df["channel"] == channel].iloc[0]
            iec_data = filtered_iec_df[filtered_iec_df["channel"] == channel].iloc[0]

            variable_name = case_data["variable"]
            unit = case_data["unit"]

            iec_box = {
                'med': iec_data["quant_0.5"],
                'q1': iec_data["quant_0.25"],
                'q3': iec_data["quant_0.75"],
                'whislo': iec_data["quant_0.01"],
                'whishi': iec_data["quant_0.99"],
                'fliers': [],
                'label': 'IEC'
            }

            case_box = {
                'med': case_data["quant_0.5"],
                'q1': case_data["quant_0.25"],
                'q3': case_data["quant_0.75"],
                'whislo': case_data["quant_0.01"],
                'whishi': case_data["quant_0.99"],
                'fliers': [],
                'label': 'Case'
            }

            bxp1 = ax.bxp([iec_box], positions=[1], widths=0.6, showfliers=False, patch_artist=True)
            bxp2 = ax.bxp([case_box], positions=[2], widths=0.6, showfliers=False, patch_artist=True)

            bxp1["boxes"][0].set_facecolor(box_colors[0])
            bxp2["boxes"][0].set_facecolor(box_colors[1])

            ax.set_title(f"{variable_name}\nChannel {channel}")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(labels)
            ax.set_ylabel(f"Load [{unit}]")
            ax.grid(True)
            # Y-axis label and tick logic
            if idx == 0:
                ax.set_ylabel(f"Load [{unit}]")  # Leftmost
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()
            elif idx == len(channels) - 1:
                ax.set_ylabel(f"Load [{unit}]")  # Rightmost
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            else:
                ax.set_ylabel("")                # Middle plots
                ax.set_yticklabels([])
                ax.tick_params(axis='y', length=0)  # Hide tick lines

        fig.canvas.draw_idle()

    # Attach the update function to sliders
    for slider in sliders:
        slider.on_changed(update)

    # Initial plot
    update()

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(box_colors, labels)]
    fig.legend(handles=legend_patches, loc="upper center", ncol=2)

    plt.show()

def plot_selected_blade_pdfs_with_sliders(pdf_df, iec_pdf_df):
    blade_channels = [26, 29, 32]

    if "channel" not in pdf_df.columns:
        print("The DataFrame does not contain a 'channel' column.")
        return

    # Determine filterable columns
    excluded_cols = ["channel", "bin", "pdf", "variable", "unit"]
    included_cols = [col for col in pdf_df.columns if col not in excluded_cols]

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Assign channel axes
    channel_axes = axes[:3]
    slider_ax = axes[3]
    slider_ax.axis('off')  # We'll place sliders manually here

    # Create slider axes in the 4th subplot
    sliders = []
    for i, col in enumerate(included_cols):
        # [x, y, width, height] relative to slider_ax's coordinate system (0–1)
        ax_slider = slider_ax.inset_axes([0.1, 0.8 - i * 0.1, 0.8, 0.07])
        unique_vals = sorted(pdf_df[col].dropna().unique())
        slider = Slider(
            ax=ax_slider,
            label=col,
            valmin=min(unique_vals),
            valmax=max(unique_vals),
            valinit=unique_vals[0],
            valstep=unique_vals if len(unique_vals) < 50 else None
        )
        sliders.append(slider)

    def update(val=None):
        # Filter based on slider values
        filters = {col: slider.val for col, slider in zip(included_cols, sliders)}
        filtered_df = pdf_df.copy()
        filtered_iec_df = iec_pdf_df.copy()

        for col, val in filters.items():
            filtered_df = filtered_df[filtered_df[col] == val]
        if "ws" in filters:
            filtered_iec_df = filtered_iec_df[filtered_iec_df["ws"] == filters["ws"]]

        # Clear and update plots
        for ax, ch in zip(channel_axes, blade_channels):
            ax.clear()
            if ch in filtered_df["channel"].values:
                _plot_single_channel_sliders(ax, filtered_df, filtered_iec_df, ch)
            else:
                ax.text(0.5, 0.5, f"No data for channel {ch}", ha="center", va="center")

        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    update()  # Initial plot
    plt.show()

def plot_channel_pdf_with_sliders(pdf_df, iec_pdf_df, channel_number):
    if "channel" not in pdf_df.columns:
        print("The DataFrame does not contain a 'channel' column.")
        return

    # Determine filterable columns
    excluded_cols = ["channel", "bin", "pdf", "variable", "unit"]
    included_cols = [col for col in pdf_df.columns if col not in excluded_cols]

    # Create 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    plot_ax, slider_ax = axes
    slider_ax.axis('off')  # We'll place sliders manually

    # Create sliders in the right subplot
    sliders = []
    for i, col in enumerate(included_cols):
        ax_slider = slider_ax.inset_axes([0.1, 0.8 - i * 0.1, 0.8, 0.07])
        unique_vals = sorted(pdf_df[col].dropna().unique())
        slider = Slider(
            ax=ax_slider,
            label=col,
            valmin=min(unique_vals),
            valmax=max(unique_vals),
            valinit=unique_vals[0],
            valstep=unique_vals if len(unique_vals) < 50 else None
        )
        sliders.append(slider)

    def update(val=None):
        # Filter based on slider values
        filters = {col: slider.val for col, slider in zip(included_cols, sliders)}
        filtered_df = pdf_df.copy()
        filtered_iec_df = iec_pdf_df.copy()

        for col, val in filters.items():
            filtered_df = filtered_df[filtered_df[col] == val]
        if "ws" in filters:
            filtered_iec_df = filtered_iec_df[filtered_iec_df["ws"] == filters["ws"]]

        # Clear and update the plot
        plot_ax.clear()
        if channel_number in filtered_df["channel"].values:
            _plot_single_channel_sliders(plot_ax, filtered_df, filtered_iec_df, channel_number)
        else:
            plot_ax.text(0.5, 0.5, f"No data for channel {channel_number}", ha="center", va="center")

        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    update()  # Initial plot
    plt.tight_layout()
    plt.show()

def _plot_single_channel_sliders(ax, df, iec_df, channel_number):
    # Filter data for the channel
    channel_df = df[df["channel"] == channel_number]
    iec_channel_df = iec_df[iec_df["channel"] == channel_number]

    if iec_channel_df.empty and channel_df.empty:
        ax.text(0.5, 0.5, f"No data for channel {channel_number}", ha="center", va="center")
        return

    # Plot IEC curves
    for _, row in iec_channel_df.iterrows():
        pdf_array = row["pdf"]
        mids = pdf_array[:, 1]
        probs = pdf_array[:, 3]
        label_parts = [f"{col}={row[col]}" for col in iec_channel_df.columns if col not in ["pdf", "channel"]]
        label = "IEC "# + ", ".join(label_parts)
        ax.plot(mids, probs, label=label, color="grey", linestyle="--", alpha=0.7)

    # Plot case curves
    for _, row in channel_df.iterrows():
        pdf_array = row["pdf"]
        mids = pdf_array[:, 1]
        probs = pdf_array[:, 3]
        label_parts = [f"{col}={row[col]}" for col in channel_df.columns if col not in ["pdf", "channel"]]
        label = "Case "# + ", ".join(label_parts)
        ax.plot(mids, probs, label=label)

    ax.set_title(f"PDFs for Channel {channel_number}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend(loc="upper right")

def plot_channel_ws_overview_sliders(stats_df, iec_stats_df):
    # Columns that are not filters
    excluded_cols = ["ws", "variable", "unit", "mean", "stdev", "quant_0.01", "quant_0.25",
                     "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in stats_df.columns if col not in excluded_cols]

    # Create figure with 2 subplots (mean ± stdev, P90–P99)
    fig, (ax_mean, ax_p) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(bottom=0.35)

    sliders = []

    # Create sliders at the bottom
    for i, col in enumerate(included_cols):
        ax_slider = fig.add_axes([0.1, 0.25 - i * 0.05, 0.8, 0.03])
        slider = Slider(
            ax=ax_slider,
            label=col,
            valmin=stats_df[col].min(),
            valmax=stats_df[col].max(),
            valinit=stats_df[col].iloc[0],
            valstep=sorted(stats_df[col].unique())
        )
        sliders.append((col, slider))

    def update(val=None):
        # Filter stats_df based on current slider values
        filtered_df = stats_df.copy()
        iec_filtered_df = iec_stats_df.copy()

        for col, slider in sliders:
            selected_val = slider.val
            filtered_df = filtered_df[np.isclose(filtered_df[col], selected_val, atol=1e-3)]
            if col == "channel":
                iec_filtered_df = iec_filtered_df[np.isclose(iec_filtered_df[col], selected_val, atol=1e-3)]

        if filtered_df.empty:
            ax_mean.clear()
            ax_mean.set_title("No matching data")
            ax_p.clear()
            ax_p.set_title("No matching data")
            fig.canvas.draw_idle()
            return

        variable = filtered_df["variable"].iloc[0]
        unit = filtered_df["unit"].iloc[0]

        # Sort for consistent plotting
        iec_sorted = iec_filtered_df.sort_values("ws")
        case_sorted = filtered_df.sort_values("ws")

        ws_iec, mean_iec, stdev_iec = iec_sorted["ws"], iec_sorted["mean"], iec_sorted["stdev"]
        p90_iec, p99_iec = iec_sorted["quant_0.9"], iec_sorted["quant_0.99"]
        ws_case, mean_case, stdev_case = case_sorted["ws"], case_sorted["mean"], case_sorted["stdev"]
        p90_case, p99_case = case_sorted["quant_0.9"], case_sorted["quant_0.99"]

        # Plot Mean ± Stdev
        ax_mean.clear()
        ax_mean.plot(ws_iec, mean_iec, color="black", linestyle="--", label="IEC Mean")
        ax_mean.fill_between(ws_iec, mean_iec - stdev_iec, mean_iec + stdev_iec, color="gray", alpha=0.3)

        ax_mean.plot(ws_case, mean_case, color="cornflowerblue", linestyle="--", label="Case Mean")
        ax_mean.fill_between(ws_case, mean_case - stdev_case, mean_case + stdev_case, color="cornflowerblue", alpha=0.3)

        ax_mean.set_title(f"{variable} – Mean ± Stdev")
        ax_mean.set_xlabel("Nominal Wind Speed [m/s]")
        ax_mean.set_ylabel(f"Value [{unit}]")
        ax_mean.grid(True)
        ax_mean.legend()

        # Plot P90–P99
        ax_p.clear()
        ax_p.plot(ws_iec, p90_iec, color="black", linestyle="--", label="IEC P90")
        ax_p.plot(ws_iec, p99_iec, color="black", linestyle="--", label="IEC P99")
        ax_p.fill_between(ws_iec, p90_iec, p99_iec, color="gray", alpha=0.3)

        ax_p.plot(ws_case, p90_case, color="cornflowerblue", linestyle="--", label="Case P90")
        ax_p.plot(ws_case, p99_case, color="cornflowerblue", linestyle="--", label="Case P99")
        ax_p.fill_between(ws_case, p90_case, p99_case, color="cornflowerblue", alpha=0.3)

        ax_p.set_title(f"{variable} – P90–P99")
        ax_p.set_xlabel("Nominal Wind Speed [m/s]")
        ax_p.set_ylabel(f"Value [{unit}]")
        ax_p.grid(True)
        ax_p.legend()

        fig.canvas.draw_idle()

    # Attach update function to sliders
    for _, slider in sliders:
        slider.on_changed(update)

    update()
    plt.show()

def plot_joint_probabilities(stats_df, channel):
    ch_df = stats_df[stats_df["channel"] == channel]
    variable = ch_df["variable"].iloc[0][-6:]
    unit = ch_df["unit"].iloc[0]

    params = {
    "ws [m/s]": ch_df["ws"],
    "h [m]": ch_df["h"],
    "w [m]": ch_df["w"],
    "s [m/s]": ch_df["s"],
    f"P50 {variable} [{unit}]": ch_df["quant_0.5"],
    f"P90 {variable} [{unit}]": ch_df["quant_0.9"]}

    param_keys = list(params.keys())
    
    fig, axs = plt.subplots(len(params), len(params))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for i in range(len(params)):
        for j in range(len(params)):
            ax = axs[i, j]

            if j > i:
                ax.axis("off")
                continue

            xi = params[param_keys[j]]
            yi = params[param_keys[i]]
            
            if i == j:
                # Marginal distribution
                sns.kdeplot(xi, ax=ax, color=plt.cm.viridis(0.2), fill=True)
            else:
                # Joint PDF with contours
                xy = np.vstack([xi, yi])
                kde = gaussian_kde(xy)
                xmin, xmax = xi.min(), xi.max()
                ymin, ymax = yi.min(), yi.max()
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contourf(xx, yy, zz, levels=10, cmap="viridis")
            
            if i == len(params)-1:
                ax.set_xlabel(param_keys[j])
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(param_keys[i])
            else:
                ax.set_yticks([])

    plt.suptitle("Joint and Marginal Distributions of Wind Profile Parameters")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def plot_del_ws_overview(stats_df, iec_stats_df):
    channels = [17, 26]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    del_tag = "del_1e+07_3"

    for i, channel in enumerate(channels):
        iec_channel_df = iec_stats_df[iec_stats_df["channel"] == channel]
        channel_df = stats_df[stats_df["channel"] == channel]
        variable = channel_df["variable"].iloc[0]
        data = channel_df.iloc[0]

        iec_channel_df = iec_channel_df.sort_values("ws")
        channel_df = channel_df.sort_values("ws")

        # Get common ws values
        common_ws = iec_channel_df["ws"].isin(channel_df["ws"])
        common_ws_channel = channel_df["ws"].isin(iec_channel_df["ws"])

        # Filter both dataframes to only keep rows with common ws
        iec_channel_df_matched = iec_channel_df[common_ws]
        channel_df_matched = channel_df[common_ws_channel]

        # Optional: Check if the common ws values are exactly the same and aligned
        if not iec_channel_df_matched["ws"].reset_index(drop=True).equals(channel_df_matched["ws"].reset_index(drop=True)):
            raise ValueError(f"Mismatch in common ws values for channel {channel}")

        if channel == 26:
            del_tag = "del_1e+07_10"

        ax = axes[i]
        width = 0.35  # Bar width

        # X locations (ws) and slight shift for IEC
        x_cnbl = channel_df_matched["ws"]
        x_cnbl_pos = range(len(channel_df_matched))
        x_iec_pos = [x + width for x in x_cnbl_pos]

        # Midpoints for xticks
        xtick_pos = [x + width / 2 for x in x_cnbl_pos]

        # Plot bars
        ax.set_axisbelow(True)
        ax.grid(True, zorder=0)
        ax.bar(x_cnbl_pos, channel_df_matched[del_tag], width=width, label="Case", color="cornflowerblue", edgecolor="black", alpha=0.8, zorder=3)
        ax.bar(x_iec_pos, iec_channel_df_matched[del_tag], width=width, label="IEC", color="grey", edgecolor="black", alpha=0.8, zorder=3)

        # Set labels and ticks
        ax.set_title(f"Channel {channel}")
        ax.set_xlabel("Nominal wind speed [m/s]")
        ax.set_ylabel(f"{del_tag} [kNm]")
        # Set labels and ticks
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels([f"{ws:.1f}" for ws in x_cnbl], rotation=45)
        ax.legend()

    # Info text once at bottom center
    excluded_cols = ["ws", "channel", "variable", "unit", "mean", "stdev", "quant_0.01", "quant_0.25",
                    "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in channel_df.columns if col not in excluded_cols]
    info_text = " | ".join(f"{col}: {data[col]:.2f}" for col in included_cols)
    fig.text(0.5, 0.02, f"Case Data – {info_text}", ha="center", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout(rect=[0, 0.05, 1, 1])