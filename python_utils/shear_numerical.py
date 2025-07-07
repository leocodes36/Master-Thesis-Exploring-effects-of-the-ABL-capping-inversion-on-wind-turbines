# Import relevant libraries
import numpy as np
import pylab as plt
import seaborn as sns
import pandas as pd
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

def calculate_alpha(H, S, W):
    U = 10
    zhub = 119
    kappa = 0.4
    z0 = 0.0001
    ustar = (U * kappa) / np.log(zhub / z0)
    term1 = (zhub / (ustar / kappa * np.log(zhub / z0) + S * np.exp(-(zhub - H) ** 2 / W ** 2)))
    term2 = (1 / zhub * (ustar / kappa) - (2 * S / W**2) * np.exp(-(zhub - H) ** 2 / W ** 2) * (zhub - H))
    return term1 * term2

# Define Simulation parameters
zhub, TurbNz, Turbdz = 119.0, 180, 1

Winds = np.array([5.0, 8.0, 10.0, 11.0, 12.0, 14.0])
Heights = np.arange(100, 410, 10) #np.array([150.0, 200.0, 250.0, 300.0, 400.0, 500.0])
Widths = np.arange(20, 150, 5)#np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25])
Strengths = np.arange(0, 11, 1) #np.array([4, 6, 8, 10])

print(f"Testing {len(Winds)} Windspeeds, {len(Heights)} Heights, {len(Widths)} Widths and {len(Strengths)} Strengths.")
print(f"Total number simulations: {len(Winds)*len(Heights)*len(Widths)*len(Strengths)}")

ShearList = []

for i, Wind in enumerate(Winds):
    for j, JetHeight in enumerate(Heights):
        for k, JetWidth in enumerate(Widths):
            for l, JetStrength in enumerate(Strengths):
                WindArray = JetShear(Wind, zhub, TurbNz, Turbdz, JetHeight, JetWidth, JetStrength)
                WindShear = getAlpha(WindArray, zhub, Turbdz)
                WindShearTopHalf = getAlphaTopHalf(WindArray, zhub, Turbdz)
                WindShearTopQuarter = getAlphaTopQuarter(WindArray, zhub, Turbdz)
                ShearList.append((Wind, JetHeight, JetWidth, JetStrength, WindShear, WindShearTopHalf, WindShearTopQuarter))

ShearArray = np.array(ShearList)
print("Simluation done.")

# Make the array into a dataframe
df = pd.DataFrame(ShearList, columns=["ws", "h", "w", "s", "alpha", "alphatop50", "alphatop25"])

"""print("Plotting sample profiles...")
# Sample Wind profiles and plotting power law approximations
sample_cases = [
    (150, 30, 6),
    (200, 60, 8),
    (300, 100, 10)
]
sample_Wind = 10
plt.figure()

# Define a color map (using seaborn for more color variety)
colors = sns.color_palette("RdBu_r", len(sample_cases))
my_colors = ["cornflowerblue", "mediumseagreen", "darkorange"]

for idx, (JetHeight, JetWidth, JetStrength) in enumerate(sample_cases):
    # Calculate WindArray and shear exponent for the current case
    WindArray = JetShear(sample_Wind, zhub, TurbNz, Turbdz, JetHeight, JetWidth, JetStrength)
    alpha = getAlpha(WindArray, zhub, Turbdz)
    z = zhub + Turbdz * (np.arange(len(WindArray)) - len(WindArray) // 2)

    # Define vertical space and reference wind speed at hub height
    PowerLawArray = PowerLawShear(sample_Wind, zhub, TurbNz, Turbdz, alpha)

    # Use the same color for both the actual wind profile and the power law approximation
    color = my_colors[idx]
    
    # Plotting the actual wind profile and power law approximation
    plt.plot(WindArray, z, label=f"Jet Profile: H={JetHeight}, W={JetWidth}, S={JetStrength}", color=color)
    plt.plot(PowerLawArray, z, linestyle='--', label=fr"Power Law Approx. ($\alpha$={alpha:.2f})", color=color)

plt.xlabel("Wind Speed U [m/s]")
plt.ylabel("Height z [m]")
plt.legend()
plt.grid()
plt.show()

print("Making 3D Scatter Plot...")
# 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df["h"], df["w"], df["alpha"], c=df["s"], cmap='RdBu_r')

ax.set_xlabel("Jet Height H [m]")
ax.set_ylabel("Jet Width W [m]")
ax.set_zlabel(r"Wind Shear $\alpha$ [-]")
plt.colorbar(sc, label="Jet Strength S [m/s]")
plt.show()

print("Making Heat Map (fixing JetStrength to 8) at U 10 m/s...")
# Heatmap
# Filter for ws=10 and pivot the data to create a heatmap (fixing JetStrength to 6)
idf = df[np.isclose(df["ws"], 10, atol=1e-3)]
heatmap_data = idf[idf["s"] == 8].pivot(index="h", columns="w", values="alpha")

plt.figure()
sns.heatmap(heatmap_data, cmap="RdBu_r", annot=False, fmt=".2f", cbar_kws={'label': r"$\alpha$ [-]"})
plt.xlabel("Jet Width W [m]")
plt.ylabel("Jet Height H [m]")
plt.show()"""

print("Plotting the histogram for Wind Shear values...")
# Histogram for both WindShears and WindShearsTopHalf
bins = np.linspace(-1.5, 1.5, 30)
xticks = np.linspace(-1.5, 1.5, 4)

fig, axs = plt.subplots(2, 2)
axs = axs.flatten()

# Top Quarter
axs[0].hist(df["alphatop25"], bins=bins, color='cornflowerblue', edgecolor='black', alpha=0.8)
axs[0].set_ylabel("Frequency [-]")
axs[0].set_title(r"$\alpha$ over top 25% of rotor")
axs[0].set_xticks(xticks)

# Top Half
axs[1].hist(df["alphatop50"], bins=bins, color='cornflowerblue', edgecolor='black', alpha=0.8)
axs[1].set_ylabel("Frequency [-]")
axs[1].set_title(r"$\alpha$ over top 50% of rotor")
axs[1].sharex(axs[0])

# All Heights
axs[2].hist(df["alpha"], bins=bins, color='cornflowerblue', edgecolor='black', alpha=0.8)
axs[2].set_xlabel(r"Wind Shear $\alpha$ [-]")
axs[2].set_ylabel("Frequency [-]")
axs[2].set_title(r"$\alpha$ over whole rotor")
axs[2].sharex(axs[0])

# Alpha Distribution
H_samples = np.random.gamma(2.3, 3.9, 1000) * 50
S_samples = np.random.normal(loc=2.12, scale=0.5, size=H_samples.shape)
W_samples = H_samples * 0.25
alpha_samples = calculate_alpha(H_samples, S_samples, W_samples)
axs[3].hist(alpha_samples, bins=bins, color='darkorange', edgecolor='black', alpha=0.8)
axs[3].set_xlabel(r"Wind Shear $\alpha$ [-]")
axs[3].set_ylabel("Frequency [-]")
axs[3].set_title(r"Analytical $\alpha$ distribution")
axs[3].set_xticks(np.linspace(-1.5, 1.5, 6))


for ax in axs:
    ax.set_yscale('log')
    ax.grid(True)

plt.tight_layout()
plt.show()

# Categorize all three jet parameters
def categorize(series, labels=["Low", "Mid", "High"]):
    return pd.qcut(series, q=3, labels=labels)

df["JetHeightCat"] = categorize(df["h"])
df["JetWidthCat"] = categorize(df["w"])
df["JetStrengthCat"] = categorize(df["s"])

height_levels = ["Low", "Mid", "High"]
width_levels = ["Low", "Mid", "High"]
strength_levels = ["Low", "Mid", "High"]

for h_cat in height_levels:
    min_h = df[df["JetHeightCat"] == h_cat]["h"].min()
    max_h = df[df["JetHeightCat"] == h_cat]["h"].max()
    min_alpha = df[df["JetHeightCat"] == h_cat]["alpha"].min()
    max_alpha = df[df["JetHeightCat"] == h_cat]["alpha"].max()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle(fr"Wind Shear $\alpha$ Distribution - Jet Height: {h_cat} ({min_h:.1f} [m] – {max_h:.1f} [m])")
    bins = np.linspace(np.min(min_alpha), np.max(max_alpha), 21)

    for i, w_cat in enumerate(width_levels):
        for j, s_cat in enumerate(strength_levels):
            ax = axes[i, j]
            subset = df[
                (df["JetHeightCat"] == h_cat) &
                (df["JetWidthCat"] == w_cat) &
                (df["JetStrengthCat"] == s_cat)
            ]
            ax.hist(subset["alpha"], bins=bins, color="cornflowerblue", edgecolor="black", alpha=0.8)
            ax.set_title(f"W: {w_cat} [m], S: {s_cat} [m/s]")
            ax.set_xlabel(r"$\alpha$ [-]")
            ax.set_ylabel("Frequency [-]")
            xticks = np.arange(0.0, 1.0, 0.2)
            ax.set_xticks(xticks)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()