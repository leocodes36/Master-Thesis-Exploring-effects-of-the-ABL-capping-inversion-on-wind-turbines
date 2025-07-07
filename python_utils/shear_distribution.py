import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import norm
from scipy.stats import gamma
from functions import LogLawustar, LogLawU, fCoriolis, GDL, JetWidth, BruntVäisälä
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

"""
This script performs a Monte Carlo simulation to explore the variability in wind shear (α) 
within the atmospheric boundary layer (ABL), driven by stochastic variations in jet characteristics.

The core steps and logic are:
1. **Parameter Initialization**:
   - Uses meteorological and geographical parameters (e.g., hub height, roughness, Coriolis parameter).
   - Computes key stability and flow metrics (e.g., friction velocity `ustar`, Coriolis frequency `fc`, Brunt-Väisälä frequency `N`, and gradient Richardson-based length scale `G`).

2. **Sampling**:
   - Jet height `H` is modeled using a Gamma distribution.
   - Jet width `W` is linearly dependent on `H` with added relative Gaussian noise (5% of the nominal width).
   - Jet strength `S` follows a Gaussian distribution around its mean value derived from `G` and the background wind speed `U`.

3. **Shear Calculation**:
   - A nonlinear function estimates wind shear α for each (`H`, `W`, `S`) sample combination.
   - The result is a distribution of α values representing different plausible ABL states.

4. **Visualization**:
   - Four histograms show the distributions of `H`, `W`, `S`, and α.
   - A 2D heatmap shows the mean α conditioned on `H` and `W` for fixed `S ≈ μ ± 0.1 m/s`.
   - A second heatmap shows the 90th percentile (P₉₀) of α given `H` and `W`, highlighting extreme shear conditions.
   - Overlay includes ±3σ range of `H` per `W` bin for added interpretability.

5. **Statistical Output**:
   - Prints the mean, variance, and standard deviation of α.
   - Computes and displays the cumulative probability P(H ≤ 200 m) from the Gamma distribution.

This tool helps characterize how variability in LLJ-like features impacts shear at turbine hub height, supporting sensitivity analysis and robust wind turbine design.
"""

# Parameters
U = 10
zhub = 119
kappa = 0.4
z0 = 0.0001                         
theta0 = 290
dthetadz = 0.003
ustar = LogLawustar(U, zhub, z0)
fc = fCoriolis(55)
Nc = BruntVäisälä(theta0, dthetadz)
G = GDL(ustar, fc, z0)
N = 10000 # Number of Monte Carlo samples

# ABL Height Distribution (gamma)
k_H = 2.3  # shape
s_H = 3.9  # scale

# Initial approximations
H_init = np.mean(np.random.gamma(k_H, s_H, N) * 50)
W_init = JetWidth(ustar, fc, Nc)
S_init = 1.07 * G - U

# CDF Estimate
# Gamma distribution parameters for ABL height
k_H = 2.3   # shape parameter
s_H = 3.9   # scale parameter

# Target height (e.g., H = 200 m)
H_target = 200

# Calculate cumulative probability P(H <= 200)
# Note: If the sampled values are multiplied by 50, the actual scale becomes s_H * 50
scale_adjusted = s_H * 50
cdf_value = gamma.cdf(H_target, a=k_H, scale=scale_adjusted)

print(f"Cumulative probability P(H ≤ {H_target}) = {cdf_value:.4f}")

print("H_init:", H_init, "m")
print("W_init:", W_init, "m")
print("S_init:", S_init, "m/s")

# Jet Strength distribution (normal)
mu_S = S_init  # Mean of S
sigma_S = 0.5  # Standard deviation of S

# Generate samples
H_samples = np.random.gamma(k_H, s_H, N) * 50
# Define W based on H and add Gaussian noise proportional to H
relative_noise_std = 0.05  # 5% of the mean W value as standard deviation
W_nominal = 0.25 * H_samples
W_noise = np.random.normal(0, relative_noise_std * W_nominal)
W_samples = W_nominal + W_noise
S_samples = np.random.normal(mu_S, sigma_S, N)

# Function to calculate shear alpha
def calculate_alpha(H, S, W):
    term1 = (zhub / (ustar / kappa * np.log(zhub / z0) + S * np.exp(-(zhub - H) ** 2 / W ** 2)))
    term2 = (1 / zhub * (ustar / kappa) - (2 * S / W**2) * np.exp(-(zhub - H) ** 2 / W ** 2) * (zhub - H))
    return term1 * term2

# Compute alpha for all samples
alpha_samples = np.array([calculate_alpha(H, S, W) for H, S, W in zip(H_samples, S_samples, W_samples)])

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2)

n_bin = 30
# Histogram of Height
axs[0, 0].hist(H_samples, bins=n_bin, color='cornflowerblue', edgecolor='black', alpha=0.8)
axs[0, 0].set_xlabel("Jet Height H [m]")
axs[0, 0].set_ylabel("Frequency [-]")

# Histogram of Width
axs[0, 1].hist(W_samples, bins=n_bin, color='cornflowerblue', edgecolor='black', alpha=0.8)
axs[0, 1].set_xlabel("Jet Width W [m]")
axs[0, 1].set_ylabel("Frequency [-]")

# Histogram of Strength
axs[1, 0].hist(S_samples, bins=n_bin, color='cornflowerblue', edgecolor='black', alpha=0.8)
axs[1, 0].set_xlabel("Jet Strength S [m/s]")
axs[1, 0].set_ylabel("Frequency [-]")

# Histogram of Shear
axs[1, 1].hist(alpha_samples, bins=n_bin, color='darkorange', edgecolor='black', alpha=0.8)
axs[1, 1].set_xlabel(r"Wind Shear $\alpha$ [-]")
axs[1, 1].set_ylabel("Frequency [-]")

for ax in axs.flatten():
    ax.set_yscale('log')
    ax.grid(True)

plt.tight_layout()

# Estimate the mean and variance of the shear:
mean_alpha = np.mean(alpha_samples)
variance_alpha = np.var(alpha_samples)
sigma_alpha = np.std(alpha_samples)

print(f"Mean shear: {mean_alpha}")
print(f"Variance of shear: {variance_alpha}")
print(f"Standard deviation of shear: {sigma_alpha}")

plt.show()

# Fix S to a narrow band around its mean
S_fixed = mu_S
tolerance = 0.1  # +/- range

# Filter samples
mask = (S_samples >= S_fixed - tolerance) & (S_samples <= S_fixed + tolerance)
H_sel = H_samples[mask]
W_sel = W_samples[mask]
alpha_sel = alpha_samples[mask]

# Create 2D bins
n_bins = 50
H_bins = np.linspace(H_sel.min(), H_sel.max(), n_bins + 1)
W_bins = np.linspace(W_sel.min(), W_sel.max(), n_bins + 1)

# Digitize
H_idx = np.digitize(H_sel, H_bins) - 1
W_idx = np.digitize(W_sel, W_bins) - 1

mean_H_per_W = np.full(n_bins, np.nan)
std_H_per_W = np.full(n_bins, np.nan)

for w_bin in range(n_bins):
    H_in_bin = H_sel[W_idx == w_bin]
    if len(H_in_bin) > 0:
        mean_H_per_W[w_bin] = np.mean(H_in_bin)
        std_H_per_W[w_bin] = np.std(H_in_bin)

# Create heatmap array
heatmap = np.full((n_bins, n_bins), np.nan)
counts = np.zeros((n_bins, n_bins))

# Fill with mean alpha values per bin
for h, w, a in zip(H_idx, W_idx, alpha_sel):
    if 0 <= h < n_bins and 0 <= w < n_bins:
        if np.isnan(heatmap[h, w]):
            heatmap[h, w] = a
            counts[h, w] = 1
        else:
            heatmap[h, w] += a
            counts[h, w] += 1

# Average
heatmap = heatmap / np.where(counts == 0, 1, counts)

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(
    heatmap.T, origin='lower', aspect='auto',
    extent=[H_bins[0], H_bins[-1], W_bins[0], W_bins[-1]],
    cmap='RdBu_r'
)
plt.colorbar(label=r"Mean Wind Shear $\alpha$")
plt.xlabel("Jet Height H [m]")
plt.ylabel("Jet Width W [m]")
plt.title(fr"Shear $\alpha$ with S$\approx${np.round(S_fixed, 2)}$\pm${tolerance} m/s")
plt.grid(True)
plt.tight_layout()
plt.show()

alpha_per_bin = defaultdict(list)

# Collect alpha samples per bin
for h, w, a in zip(H_idx, W_idx, alpha_sel):
    if 0 <= h < n_bins and 0 <= w < n_bins:
        alpha_per_bin[(h, w)].append(a)

# Initialize an array to hold P90 values
alpha_p90_map = np.full((n_bins, n_bins), np.nan)

# Compute 90th percentile for each bin
for (h, w), a_list in alpha_per_bin.items():
    alpha_p90_map[h, w] = np.percentile(a_list, 90)

plt.figure(figsize=(8, 6))
fig, ax = plt.subplots(figsize=(8, 6))

W_bin_centers = 0.5 * (W_bins[:-1] + W_bins[1:])

ax.fill_betweenx(
    W_bin_centers,
    mean_H_per_W - 3*std_H_per_W,
    mean_H_per_W + 3*std_H_per_W,
    color='k',
    alpha=0.3,
    label=r'$\pm$3 $\sigma$ interval of $H$'
)

c = ax.imshow(
    alpha_p90_map.T, origin='lower', aspect='auto',
    extent=[H_bins[0], H_bins[-1], W_bins[0], W_bins[-1]],
    cmap='RdBu_r'
)
plt.colorbar(c, label=r"90th Percentile of Wind Shear $\alpha$")

ax.set_xlabel("Jet Height H [m]")
ax.set_ylabel("Jet Width W [m]")
ax.set_title(r"Conditional 90th Percentile of Wind Shear $\alpha$ given H, W: $P_{90}(\alpha \mid H, W)$")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
