import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from post_common import *
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
    "default_channels": [10, 12, 15, 17, 26, 29, 32],
    "fatigue_columns": {0: "del_1e+07_3", 1: "del_1e+07_10"},
    "shearlist": Path("/Users/leo/python/thesis/post_process/shearlist.csv")
}

def JetShear(U, zhub, TurbNz, Turbdz, JetHeight, JetWidth, JetStrength, alpha=0.2):
    """
    Computes the vertical wind shear profile influenced by a supergeostrophic jet,
    using a power-law base and a Gaussian jet centered at specified height.

    Parameters:
    U (float): Reference wind speed at hub height (m/s)
    zhub (float): Hub height (m)
    TurbNz (int): Number of vertical points in the turbulence box
    Turbdz (float): Vertical spacing in the turbulence box (m)
    JetHeight (float): Height of the jet center (m)
    JetWidth (float): Characteristic width of the jet (m)
    JetStrength (float): Maximum wind speed contribution from the jet (m/s)
    alpha (float): Wind shear defaults to a value of 0.2 (-)

    Returns:
    np.ndarray: Wind shear box (array of wind speeds) over the turbine rotor span
    """

    # Define vertical space
    Nz = 1000                           # 1000 grid points
    dz = Turbdz                         # Spacing of dz 1 [m]
    z = np.arange(dz, Nz*dz, dz)        # vertical space array

    # Make vertical wind speed profile and add gaussian jet
    PowerLaw = U * (z / zhub)**alpha
    Jet = PowerLaw + JetStrength * np.exp(-((z - JetHeight) / JetWidth)**2)

    # Sample from hub height ± half of the turbulence box
    TurbRadius = TurbNz // 2
    hub_index = int(zhub / dz)
    lower, upper = max(0, int(hub_index - TurbRadius)), min(len(Jet), int(hub_index + TurbRadius))
    shearBox = Jet[lower:upper] #- U

    return shearBox

def PowerLawShear(U, zhub, TurbNz, Turbdz, alpha):
    """
    Computes the vertical wind shear profile based on the power law wind profile,
    scaling the wind speed as a function of height relative to the hub height.

    Parameters:
    U (float): Reference wind speed at hub height (m/s)
    zhub (float): Hub height (m)
    TurbNz (int): Number of vertical points in the turbulence box
    Turbdz (float): Vertical spacing in the turbulence box (m)
    alpha (float): Power law exponent characterizing wind shear

    Returns:
    np.ndarray: Wind shear box (array of wind speeds) over the turbine rotor span
    """
    # Define vertical space
    Nz = 1000                           # 1000 grid points
    dz = Turbdz                         # Spacing of dz 1 [m]
    z = np.arange(dz, Nz*dz, dz)        # vertical space array starting from 0

    # Make vertical wind speed profile
    PowerLaw = U * (z / zhub)**alpha

    # Sample from hub height ± half of the turbulence box
    TurbRadius = TurbNz // 2
    hub_index = int(zhub / dz)
    lower, upper = max(0, hub_index - TurbRadius), min(len(PowerLaw), hub_index + TurbRadius)
    shearBox = PowerLaw[lower:upper] #- U

    return shearBox

def getAlpha(WindSpeeds, zhub, dz):
    """
    Estimates the power-law shear exponent (alpha) over the full rotor span.

    Fits a power-law model to a vertical wind speed profile using log-log regression.
    The height levels are assumed to be evenly spaced and symmetrically distributed 
    around the hub height.

    Parameters:
    WindSpeeds (array-like): 1D array of wind speeds sampled over the rotor span.
    zhub (float):            Hub height in meters.
    dz (float):              Vertical spacing between wind speed samples (m).

    Returns:
    float: Estimated shear exponent (alpha) from the full profile.
    """
    z = zhub + dz * (np.arange(len(WindSpeeds)) - len(WindSpeeds) // 2)
    alpha, _ = np.polyfit(np.log(z), np.log(WindSpeeds), 1)
    return alpha

def getAlphaTopHalf(WindSpeeds, zhub, dz):
    """
    Estimates the power-law shear exponent (alpha) for the top half of the rotor.

    Fits a power-law model using only the upper half of the vertical wind speed 
    profile, based on log-log regression. The height levels are assumed to be evenly 
    spaced and centered around the hub height.

    Parameters:
    WindSpeeds (array-like): 1D array of wind speeds sampled over the rotor span.
    zhub (float):            Hub height in meters.
    dz (float):              Vertical spacing between wind speed samples (m).

    Returns:
    float: Estimated shear exponent (alpha) from the top half of the profile.
    """
    z = zhub + dz * (np.arange(len(WindSpeeds)) - len(WindSpeeds) // 2)
    top_half_start = len(WindSpeeds) // 2
    z_top_half = z[top_half_start:]
    WindSpeeds_top_half = WindSpeeds[top_half_start:]
    
    alphatop50, _ = np.polyfit(np.log(z_top_half), np.log(WindSpeeds_top_half), 1)
    return alphatop50

def getAlphaTopQuarter(WindSpeeds, zhub, dz):
    """
    Estimates the power-law shear exponent (alpha) for the top quarter of the rotor.

    Fits a power-law model using only the top 25% of the vertical wind speed profile,
    based on log-log regression. The height levels are assumed to be evenly spaced 
    and centered around the hub height.

    Parameters:
    WindSpeeds (array-like): 1D array of wind speeds sampled over the rotor span.
    zhub (float):            Hub height in meters.
    dz (float):              Vertical spacing between wind speed samples (m).

    Returns:
    float: Estimated shear exponent (alpha) from the top quarter of the profile.
    """
    z = zhub + dz * (np.arange(len(WindSpeeds)) - len(WindSpeeds) // 2)
    top_quarter_start = 3 * len(WindSpeeds) // 4
    z_top_quarter = z[top_quarter_start:]
    WindSpeeds_top_quarter = WindSpeeds[top_quarter_start:]
    
    alphatop25, _ = np.polyfit(np.log(z_top_quarter), np.log(WindSpeeds_top_quarter), 1)
    return alphatop25

def apply_query_filters(df, query):
    """Filter DataFrame `df` by values in `query` if the key exists as a column."""
    for key, val in query.items():
        if key in df.columns:
            df = df[df[key] == val]
    return df

def compute_alphas(row):
    shear = JetShear(row["ws"], zhub, TurbNz, Turbdz, row["h"], row["w"], row["s"])
    return pd.Series({
        "alpha": getAlpha(shear, zhub, Turbdz),
        "alphatop50": getAlphaTopHalf(shear, zhub, Turbdz),
        "alphatop25": getAlphaTopQuarter(shear, zhub, Turbdz),
        "wshub": shear[len(shear)//2]
    })

def compute_alphas_iec(row):
    shear = PowerLawShear(row["ws"], zhub, TurbNz, Turbdz, row["shear"])
    return pd.Series({
        "alpha": row["shear"],
        "alphatop50": getAlphaTopHalf(shear, zhub, Turbdz),
        "alphatop25": getAlphaTopQuarter(shear, zhub, Turbdz),
        "wshub": row["ws"]
    })

stats_files = list_stats_files("cnbl", CONFIG["data_path"])
stats_df = parse_all_stats(stats_files)

iec_stats_files = list_iec_stats_files(CONFIG["data_path"])
iec_stats_df = parse_iec_stats(iec_stats_files)

lookupdf = pd.read_csv(CONFIG["lookup_csv"])
lookupdf.columns = lookupdf.columns.str.strip().str.lower()
lookupdf = lookupdf[["channel", "variable", "unit"]]
stats_df = stats_df.merge(lookupdf, on="channel", how="left")

# Constants
zhub = 119.0
TurbNz = 180
Turbdz = 1

# Apply to all rows
stats_df[["alpha", "alphatop50", "alphatop25", "wshub"]] = stats_df.apply(compute_alphas, axis=1)
iec_stats_df[["alpha", "alphatop50", "alphatop25", "wshub"]] = iec_stats_df.apply(compute_alphas_iec, axis=1)

# Show results
pd.set_option('display.max_columns', None)
print(stats_df.head())
print(iec_stats_df.head())

print_query = {
    "ws": 10.0,
    "ti": 0.16,
    "h": 150.0,
    "w": 30.0,
    "s": 8,
    "shear": 0.2,
    "channel": 17
}

# Print filtered results
print_df = apply_query_filters(stats_df, print_query)
print("Filtered CNBL stats_df:")
print(print_df[["ws", "h", "w", "s", "alpha", "alphatop50", "alphatop25", "wshub"]])

iec_print_df = apply_query_filters(iec_stats_df, print_query)
print("\nFiltered IEC stats_df:")
print(iec_print_df[["ws", "alpha", "alphatop50", "alphatop25", "wshub"]])

query = {
    "ti": 0.16,
    "h": 200.0,
    "w": 40.0,
    "s": 8,
    "shear": 0.2
}

# Apply filters
filtered_stats_df = apply_query_filters(stats_df, query)
filtered_iec_stats_df = apply_query_filters(iec_stats_df, query)

pd.set_option('display.max_columns', None)
print(filtered_stats_df.head())
print(filtered_iec_stats_df.head())

# DEL plot over matched hub height wind speeds
def match_with_tolerance(source_df, target_df, key, tol=0.2):
    matched_source = []
    matched_target = []

    if target_df.empty:
        print("Warning: target_df is empty.")
        return pd.DataFrame(), pd.DataFrame()

    for _, source_row in source_df.iterrows():
        diffs = (target_df[key] - source_row[key]).abs()

        # Skip if no values within tolerance
        if diffs.min() > tol:
            continue

        # Use .loc because idxmin returns index label, not position
        closest_match = target_df.loc[diffs.idxmin()]

        matched_source.append(source_row)
        matched_target.append(closest_match)

    return pd.DataFrame(matched_source), pd.DataFrame(matched_target)

def plot_del_wshub_overview(stats_df, iec_stats_df):
    channels = [17, 26]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    del_tag = "del_1e+07_3"
    
    for i, channel in enumerate(channels):
        # Filter for current channel
        cnbl = stats_df[stats_df["channel"] == channel]
        iec = iec_stats_df[iec_stats_df["channel"] == channel]
        data = cnbl.iloc[0]
        # Match only wshub values that both have
        cnbl_matched, iec_matched = match_with_tolerance(cnbl, iec, key="wshub")

        cnbl_matched = cnbl_matched.sort_values(by="wshub").reset_index(drop=True)
        iec_matched = iec_matched.sort_values(by="wshub").reset_index(drop=True)

        if channel == 26:
            del_tag = "del_1e+07_10"

        ax = axes[i]
        width = 0.35  # Bar width

        # X locations (wshub) and slight shift for IEC
        x_cnbl = cnbl_matched["wshub"]
        x_cnbl_pos = range(len(cnbl_matched))
        x_iec_pos = [x + width for x in x_cnbl_pos]

        # Midpoints for xticks
        xtick_pos = [x + width / 2 for x in x_cnbl_pos]

        # Plot bars
        ax.set_axisbelow(True)
        ax.grid(True, zorder=0)
        ax.bar(x_cnbl_pos, cnbl_matched[del_tag], width=width, label="Case", color="cornflowerblue", edgecolor="black", alpha=0.8, zorder=3)
        ax.bar(x_iec_pos, iec_matched[del_tag], width=width, label="IEC", color="grey", edgecolor="black", alpha=0.8, zorder=3)

        # Set labels and ticks
        ax.set_title(f"Channel {channel}")
        ax.set_xlabel("Hub height wind speed [m/s]")
        ax.set_ylabel(f"{del_tag} [kNm]")
        # Set labels and ticks
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels([f"{ws:.1f}" for ws in x_cnbl], rotation=45)
        ax.legend()

    # Info text once at bottom center
    excluded_cols = ["alpha", "alphatop50", "alphatop25", "wshub", "ws", "channel", "variable", "unit", "mean", "stdev", "quant_0.01", "quant_0.25",
                    "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in stats_df.columns if col not in excluded_cols]
    info_text = " | ".join(f"{col}: {data[col]:.2f}" for col in included_cols)
    fig.text(0.5, 0.02, f"Case Data – {info_text}", ha="center", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

plot_del_wshub_overview(filtered_stats_df, filtered_iec_stats_df)

# Manual query plots
def plot_channel_wshub_overview(stats_df, iec_stats_df, channel_number):
    # Filter by channel number
    channel_df = stats_df[stats_df["channel"] == channel_number]
    iec_channel_df = iec_stats_df[iec_stats_df["channel"] == channel_number]

    if channel_df.empty or iec_channel_df.empty:
        print(f"Channel '{channel_number}' not found in one of the datasets.")
        return

    variable = channel_df["variable"].iloc[0]
    data = channel_df.iloc[0]

    # Sort Values
    channel_df = channel_df.sort_values("wshub")
    iec_channel_df = iec_channel_df.sort_values("wshub")

    # Extract values
    wshub = channel_df["wshub"]
    iec_wshub = iec_channel_df["wshub"]

    mean, stdev = channel_df["mean"], channel_df["stdev"]
    p90, p99 = channel_df["quant_0.9"], channel_df["quant_0.99"]
    iec_mean, iec_stdev = iec_channel_df["mean"], iec_channel_df["stdev"]
    iec_p90, iec_p99 = iec_channel_df["quant_0.9"], iec_channel_df["quant_0.99"]

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # First subplot: mean ± stdev
    axs[0].fill_between(iec_wshub, iec_mean - iec_stdev, iec_mean + iec_stdev, color="grey", alpha=0.3,
                        label=r"IEC Mean $\pm$1 Stdev")
    axs[0].fill_between(wshub, mean - stdev, mean + stdev, color="cornflowerblue", alpha=0.3,
                        label=r"Case Mean $\pm$1 Stdev")

    axs[0].set_title(f"{variable} – Channel {channel_number}")
    axs[0].set_xlabel("Hub-Height Windspeed [m/s]")
    axs[0].set_ylabel("Load [kNm]")
    axs[0].grid(True)
    axs[0].legend()

    # Second subplot: P90 / P99
    axs[1].fill_between(iec_wshub, iec_p90, iec_p99, color="grey", alpha=0.3,
                        label="IEC P90–P99")
    axs[1].fill_between(wshub, p90, p99, color="cornflowerblue", alpha=0.3,
                        label="Case P90–P99")

    axs[1].set_title(f"{variable} – Channel {channel_number}")
    axs[1].set_xlabel("Hub-Height Windspeed [m/s]")
    axs[1].grid(True)
    axs[1].legend()

    # Info text
    excluded_cols = ["ws", "wshub", "alpha", "alphatop50", "alphatop25", "channel", "variable", "unit", "mean", "stdev", "quant_0.01", "quant_0.25",
                     "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in channel_df.columns if col not in excluded_cols]
    info_text = " | ".join(f"{col}: {data[col]:.2f}" for col in included_cols)

    fig.text(0.5, 0.02, f"Case Data – {info_text}", ha="center", va="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for bottom text
    plt.show()
def plot_shear_wshub_overview(stats_df, iec_stats_df):
    if stats_df.empty or iec_stats_df.empty:
        print("Filtered data is empty for either stats_df or iec_stats_df.")
        return

    stats_df = stats_df.sort_values("wshub")
    iec_stats_df = iec_stats_df.sort_values("wshub")

    wshub = stats_df["wshub"]
    iec_wshub = iec_stats_df["wshub"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Case lines
    ax.plot(wshub, stats_df["alpha"], label=r"Case $\alpha$ (full)", color="cornflowerblue")
    ax.plot(wshub, stats_df["alphatop25"], label=r"Case $\alpha$ (top 25%)", color="mediumseagreen")
    ax.plot(wshub, stats_df["alphatop50"], label=r"Case $\alpha$ (top 50%)", color="darkorange")

    # IEC lines
    ax.plot(iec_wshub, iec_stats_df["alpha"], "--", label=r"IEC $\alpha$ (full)", color="cornflowerblue", alpha=0.5)
    ax.plot(iec_wshub, iec_stats_df["alphatop25"], "--", label=r"IEC $\alpha$ (top 25%)", color="mediumseagreen", alpha=0.5)
    ax.plot(iec_wshub, iec_stats_df["alphatop50"], "--", label=r"IEC $\alpha$ (top 50%)", color="darkorange", alpha=0.5)

    ax.set_title("Shear Parameters vs Hub-Height Wind Speed")
    ax.set_xlabel("Hub-Height Wind Speed [m/s]")
    ax.set_ylabel(r"Shear Exponent ($\alpha$)")
    ax.grid(True)
    ax.legend()

    # Optional: Add info box from query
    excluded_cols = ["channel", "ws", "wshub", "alpha", "alphatop50", "alphatop25", "variable", "unit", "mean", "stdev",
                     "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75", "quant_0.9", "quant_0.99",
                     "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in stats_df.columns if col not in excluded_cols]
    if not stats_df.empty:
        info_text = " | ".join(f"{col}: {stats_df.iloc[0][col]:.2f}" for col in included_cols)
        fig.text(0.5, 0.02, f"Case Data – {info_text}", ha="center", va="bottom",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

plot_channel_wshub_overview(filtered_stats_df, filtered_iec_stats_df, 10)
plot_shear_wshub_overview(filtered_stats_df, filtered_iec_stats_df)

# Plots with sliders
def plot_channel_wshub_overview_sliders(stats_df, iec_stats_df):
    # Columns that are not sliders
    excluded_cols = ["ws", "wshub", "alpha", "alphatop50", "alphatop25", "variable", "unit",
                     "mean", "stdev", "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75",
                     "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in stats_df.columns if col not in excluded_cols]

    # Create figure and axes
    fig, (ax_mean, ax_p) = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    sliders = []

    # Add sliders below the plots
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
        # Filter the datasets by slider values
        filtered_df = stats_df.copy()
        iec_filtered_df = iec_stats_df.copy()

        for col, slider in sliders:
            selected_val = slider.val
            filtered_df = filtered_df[np.isclose(filtered_df[col], selected_val, atol=1e-3)]
            if col == "channel":
                iec_filtered_df = iec_filtered_df[np.isclose(iec_filtered_df[col], selected_val, atol=1e-3)]

        if filtered_df.empty or iec_filtered_df.empty:
            ax_mean.clear()
            ax_p.clear()
            ax_mean.set_title("No matching data")
            ax_p.set_title("No matching data")
            fig.canvas.draw_idle()
            return

        variable = filtered_df["variable"].iloc[0]
        unit = filtered_df["unit"].iloc[0]
        data = filtered_df.iloc[0]

        # Sort by wshub
        filtered_df = filtered_df.sort_values("wshub")
        iec_filtered_df = iec_filtered_df.sort_values("wshub")

        wshub = filtered_df["wshub"]
        mean, stdev = filtered_df["mean"], filtered_df["stdev"]
        p90, p99 = filtered_df["quant_0.9"], filtered_df["quant_0.99"]

        iec_wshub = iec_filtered_df["wshub"]
        iec_mean, iec_stdev = iec_filtered_df["mean"], iec_filtered_df["stdev"]
        iec_p90, iec_p99 = iec_filtered_df["quant_0.9"], iec_filtered_df["quant_0.99"]

        # Plot mean ± stdev
        ax_mean.clear()
        ax_mean.fill_between(iec_wshub, iec_mean - iec_stdev, iec_mean + iec_stdev, color="grey", alpha=0.3,
                             label="IEC Mean ±1 Stdev")
        ax_mean.fill_between(wshub, mean - stdev, mean + stdev, color="cornflowerblue", alpha=0.3,
                             label="Case Mean ±1 Stdev")
        ax_mean.set_title(f"{variable} – Mean ± Stdev")
        ax_mean.set_xlabel("Hub-Height Windspeed [m/s]")
        ax_mean.set_ylabel(f"Value [{unit}]")
        ax_mean.grid(True)
        ax_mean.legend()

        # Plot P90–P99
        ax_p.clear()
        ax_p.fill_between(iec_wshub, iec_p90, iec_p99, color="grey", alpha=0.3,
                          label="IEC P90–P99")
        ax_p.fill_between(wshub, p90, p99, color="cornflowerblue", alpha=0.3,
                          label="Case P90–P99")
        ax_p.set_title(f"{variable} – P90–P99")
        ax_p.set_xlabel("Hub-Height Windspeed [m/s]")
        ax_p.grid(True)
        ax_p.legend()

        # Info text below plots
        info_excluded = excluded_cols + ["channel"]
        info_cols = [col for col in filtered_df.columns if col not in info_excluded]
        info_text = " | ".join(f"{col}: {data[col]:.2f}" for col in info_cols)

        """fig.texts.clear()
        fig.text(0.5, 0.02, f"Case Data – {info_text}", ha="center", va="bottom",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))"""

        fig.canvas.draw_idle()

    # Attach update to sliders
    for _, slider in sliders:
        slider.on_changed(update)

    update()
    plt.tight_layout(rect=[0, 0.25, 1, 1])
    plt.show()
def plot_shear_over_wshub_sliders(stats_df, iec_stats_df):
    # Columns we do not want sliders for
    excluded_cols = ["ws", "wshub", "alpha", "alphatop50", "alphatop25", "variable", "unit",
                     "mean", "stdev", "quant_0.01", "quant_0.25", "quant_0.5", "quant_0.75",
                     "quant_0.9", "quant_0.99", "del_1e+07_3", "del_1e+07_10"]
    included_cols = [col for col in stats_df.columns if col not in excluded_cols]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.1 + 0.05 * len(included_cols))

    sliders = []

    # Add sliders
    for i, col in enumerate(included_cols):
        ax_slider = fig.add_axes([0.15, 0.25 - i * 0.05, 0.7, 0.03])
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
        filtered_df = stats_df.copy()
        filtered_iec = iec_stats_df.copy()

        for col, slider in sliders:
            val = slider.val
            filtered_df = filtered_df[np.isclose(filtered_df[col], val, atol=1e-3)]
            if col in filtered_iec.columns:
                filtered_iec = filtered_iec[np.isclose(filtered_iec[col], val, atol=1e-3)]

        if filtered_df.empty or filtered_iec.empty:
            ax.clear()
            ax.set_title("No matching data")
            fig.canvas.draw_idle()
            return

        # Sort by wshub for clean plotting
        filtered_df = filtered_df.sort_values("wshub")
        filtered_iec = filtered_iec.sort_values("wshub")

        wshub = filtered_df["wshub"]
        iec_wshub = filtered_iec["wshub"]

        ax.clear()
        ax.plot(wshub, filtered_df["alpha"], label="Case α (full)", color="cornflowerblue")
        ax.plot(wshub, filtered_df["alphatop25"], label="Case α (top 25%)", color="mediumseagreen")
        ax.plot(wshub, filtered_df["alphatop50"], label="Case α (top 50%)", color="darkorange")

        ax.plot(iec_wshub, filtered_iec["alpha"], "--", label="IEC α (full)", color="cornflowerblue", alpha=0.5)
        ax.plot(iec_wshub, filtered_iec["alphatop25"], "--", label="IEC α (top 25%)", color="mediumseagreen", alpha=0.5)
        ax.plot(iec_wshub, filtered_iec["alphatop50"], "--", label="IEC α (top 50%)", color="darkorange", alpha=0.5)

        ax.set_title("Shear Parameters vs Hub-Height Wind Speed")
        ax.set_xlabel("Hub-Height Wind Speed [m/s]")
        ax.set_ylabel("Shear Exponent (α)")
        ax.grid(True)
        ax.legend()

        fig.canvas.draw_idle()

    for _, slider in sliders:
        slider.on_changed(update)

    update()
    plt.show()

#plot_channel_wshub_overview_sliders(stats_df, iec_stats_df)
#plot_shear_over_wshub_sliders(stats_df, iec_stats_df)
