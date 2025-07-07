import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression

from functions import LogLawustar, fCoriolis, GDL, BruntVäisälä
from post_common import *
from post_plotting import *

CONFIG = {
    "data_path": Path("/Users/leo/python/thesis/sync_results/"),
    "lookup_csv": Path("/Users/leo/python/thesis/post_process/HAWC2_Output_Channels.csv"),
    "default_channels": [10, 12, 15, 17, 26, 29, 32],
    "fatigue_columns": {0: "del_1e+07_3", 1: "del_1e+07_10"},
}

def main():
    args = parse_arguments()
    
    print("cnbl only for joint plots!")
    study_names = load_study_names(CONFIG["data_path"])

    if args.interactive or not args.study:
        selected_study = select_study(study_names)
    else:
        if args.study not in study_names:
            raise ValueError(f"Study '{args.study}' not found.")
        selected_study = args.study
    
    if selected_study != "cnbl":
        return
    
    stats_files = list_stats_files(selected_study, CONFIG["data_path"])
    stats_df = parse_all_stats(stats_files)

    lookupdf = pd.read_csv(CONFIG["lookup_csv"])
    lookupdf.columns = lookupdf.columns.str.strip().str.lower()
    lookupdf = lookupdf[["channel", "variable", "unit"]]
    stats_df = stats_df.merge(lookupdf, on="channel", how="left")

    print("\nMerged data preview:")
    pd.set_option('display.max_columns', None)
    print(stats_df.head())

    ch_choice = int(input("Which channel?").strip())

    ch_df = stats_df[stats_df["channel"] == ch_choice]
    variable = ch_df["variable"].iloc[0][-6:]
    unit = ch_df["unit"].iloc[0]
    print(ch_df.head())

    ws = np.array(ch_df["ws"])
    h = np.array(ch_df["h"])
    w = np.array(ch_df["w"])
    s = np.array(ch_df["s"])
    p90 = np.array(ch_df["quant_0.9"])

    X_train = np.column_stack((ws, h, w, s))
    y_train = p90

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Parameters
    U_mean = 10        # Mean wind speed [m/s]
    zhub = 119         # Hub height [m]
    kappa = 0.4        # von Kármán constant
    z0 = 0.0001        # Surface roughness length [m]
    theta0 = 290       # Potential temperature [K]
    dthetadz = 0.003   # Vertical temp gradient [K/m]

    ustar = LogLawustar(U_mean, zhub, z0)
    fc = fCoriolis(55)
    Nc = BruntVäisälä(theta0, dthetadz)
    G = GDL(ustar, fc, z0)
    N = 1000

    # Generate Distributions
    k_U, s_U = 2.5, 10
    U = s_U * np.random.weibull(k_U, N)

    k_H, s_H = 2.3, 3.9
    H = np.random.gamma(k_H, s_H, N) * 50  # ABL Height

    W = 0.25 * H * np.random.normal(0, 1, N)  # Add noise to break perfect collinearity # Jet Width

    mu_S, sigma_S = 1.07 * G - U, 0.5
    S = np.random.normal(mu_S, sigma_S, N)

    # Function to calculate shear alpha
    def calculate_alpha(H, W, S):
        term1 = (zhub / (ustar / kappa * np.log(zhub / z0) + S * np.exp(-(zhub - H) ** 2 / W ** 2)))
        term2 = (1 / zhub * (ustar / kappa) - (2 * S / W**2) * np.exp(-(zhub - H) ** 2 / W ** 2) * (zhub - H))
        return term1 * term2

    # Compute shear
    alpha = np.array([calculate_alpha(Hi, Wi, Si) for Hi, Wi, Si in zip(H, S, W)])

    # Predict load metric for simulated points
    sim = np.column_stack((U, H, W, S))
    p90_sim = model.predict(sim)
    print("Coefficients:", model.coef_)  # array of 4 values corresponding to U,H,W,S
    print("Intercept:", model.intercept_)
    r2 = model.score(X_train, y_train)
    print(f"R^2 score: {r2:.3f}")

    params = {"U [m/s]": U, "H [m]": H, "W [m]": W, "S [m/s]": S, "alpha [-]": alpha, f"P90 {variable} [{unit}]": p90_sim}
    param_keys = list(params.keys())

    print("Plotting...")
    fig, axs = plt.subplots(len(params), len(params))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

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

if __name__ == "__main__":
    main()
