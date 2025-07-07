import sys
import os.path
import matplotlib.pyplot as plt
from hipersim import MannTurbulenceField
from generate_box_common import *


# Main entry point for the script
if __name__ == "__main__":

    # Determine that we have all the arguments, otherwise print a helpful message
    if len(sys.argv)!=15:
        print("Give me the following, in this order:")
        print("\tThe nominal speed of the flow")
        print("\tThe turbulence intenisty of the flow")
        print("\tTime step in data")
        print("\tNx")
        print("\tNy")
        print("\tNz")
        print("\tdx")
        print("\tdy")
        print("\tdz")
        print("\tHigh-freq-comp")
        print("\tBase-name")
        print("\tSeeds as a space seperated list")
        print("\tBrunt-Vaisala Frequency")
        print("\tWave structure Amplitude")

        quit()

    # Assign all arguments to a variable
    nominal_U = float(sys.argv[1])
    nominal_TI = float(sys.argv[2])
    delta_t = float(sys.argv[3])
    Nx = int(sys.argv[4])
    Ny = int(sys.argv[5])
    Nz = int(sys.argv[6])
    dx = float(sys.argv[7])
    dy = float(sys.argv[8])
    dz = float(sys.argv[9])
    High_freq_comp = int(sys.argv[10])
    Base_name = sys.argv[11]
    Seeds_arg = int(sys.argv[12])
    N = float(sys.argv[13])
    A = float(sys.argv[14])

    # Print a helpful message explaining what we are about to do
    print("Generating the turbulence box with dimesnions","Nx:",Nx,"Ny:",Ny,"Nz:",Nz,"dx:",dx,"dy:",dy,"dz:",dz,"High-Freq-Comp:",High_freq_comp,"Base-name:",Base_name,"Seed(s):",Seeds_arg)

    # Make dir for plots
    outdir = "./plots/"
    os.makedirs(outdir, exist_ok=True)

    # Define simulation paramters
    zhub = 119.0
    params = {}
    params["L"] = get_length(zhub)                   # Turbulence length based on DTU-10MW turbine [m]
    params["alphaepsilon"] = get_strength()         # Turbulence strength [m^2 s^-3], default set to 1 and scale timeseries
    params["Gamma"] = get_anisotropy()              # Anisotropy [-]
    params["Nxyz"] = (Nx,Ny,Nz)                     # Number of points in box [-]
    params["dxyz"] = (dx,dy,dz)                     # Spacing between points [m]
    params["seed"] = Seeds_arg                      # Seed for random number gen [-]
    params["HighFreqComp"] = High_freq_comp         # Compensation at high frequencies to make sure 5/3 law [-]
    params["double_xyz"] = (True, False, False)     # Doubling along given axis for bigger box [-]
    params["BaseName"] = Base_name

    # Generate turbulence box
    mtf = MannTurbulenceField.generate(alphaepsilon = params["alphaepsilon"],
                                        L = params["L"],
                                        Gamma = params["Gamma"],
                                        Nxyz = params["Nxyz"],
                                        dxyz = params["dxyz"],
                                        seed = Seeds_arg,
                                        HighFreqComp = params["HighFreqComp"],
                                        double_xyz = params["double_xyz"])
    
    # Scale box to match the IEC wind and turbulence class and print info
    print('Scaling turbulence to match IEC class 1A.')
    print (f'Before: Box TI={mtf.uvw[0].std(0).mean()/nominal_U:.3f}, alphaepsilon:{mtf.alphaepsilon:.3f}, theoretical spectrum TI {mtf.spectrum_TI(nominal_U):.2f}')
    mtf.scale_TI(TI=nominal_TI, U=nominal_U)
    print (f'After: Box TI={mtf.uvw[0].std(0).mean()/nominal_U:.3f}, alphaepsilon:{mtf.alphaepsilon:.3f}, theoretical spectrum TI {mtf.spectrum_TI(nominal_U):.2f}')
    print(f'Iref = {nominal_TI}, TI = {mtf.uvw[0].std(0).mean()/nominal_U:.3f} (Difference due to uncertainty in lower frequencies and seed-to-seed differences.)')

    # Get the turbulence spectra
    k_rea, spectra_rea = mtf.spectra()
    k_int, spectra_int = mtf.spectra_integrated()
    k_lut, spectra_lut = mtf.spectra_lookup()

    # Colors, labels and plotting
    c_lst = ["cornflowerblue", "darkorange", "mediumseagreen", "#d62728"]
    s_lst = ["uu","vv","ww","uw"]

    # Plot of spectra
    plt.figure()
    for phi_rea, phi_int, phi_lut, c, l in zip(spectra_rea, spectra_int, spectra_lut, c_lst, s_lst):
        plt.semilogx(k_rea, phi_rea*k_rea, color=c, label=l)
        plt.semilogx(k_int, phi_int*k_int, '--', color=c)
        plt.semilogx(k_lut, phi_lut*k_lut, ':', color=c)

    plt.plot([], '-', color='gray', label='Realization')
    plt.plot([], '--', color='gray', label='LookupTable')
    plt.plot([], ':', color='gray', label='Integrated')
    plt.xlabel('Wave number, $k_1$ $  [m^{-1}$]')
    plt.ylabel('$k_1 S(k_1)[m^2s^{-2}]$')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outdir, f"Spectra_{Base_name}{params['seed']}.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Export the data to an array
    da = mtf.to_xarray()

    print("Before constraints:")
    print(f"var(u): {np.var(da.sel(uvw='u').values)}")
    print(f"var(v): {np.var(da.sel(uvw='v').values)}")
    print(f"var(w): {np.var(da.sel(uvw='w').values)}")

    z_vals = da['z'].values
    hub = z_vals[np.abs(z_vals - zhub).argmin()]
    y_vals = da.coords['y'].values
    mid_y_index = Ny // 2
    mid_y_val = y_vals[mid_y_index]
    constraints = CoherentWaveConstraint3(da, nominal_U, zhub, params["Nxyz"], params["dxyz"], A, N)

    # Use .constrain() method of the mtf object
    mtf.constrain(constraints)

    # Difference plot between raw and constrained windfields
    diff = da - mtf.to_xarray()

    fig = plt.figure(figsize=(10, 6))
    diff.sel(uvw='u', y=mid_y_val).plot(x='x')

    plt.title(f"Difference of Windfields of u at y = {mid_y_val:.1f} m")
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.grid(True)

    plt.savefig(os.path.join(outdir, f"Diff_{Base_name}{params['seed']}.pdf"), dpi=150, bbox_inches='tight')
    plt.close()

    # Add implicit shear of mann model parameters
    print('Adding implicit shear from mann model.')
    shear = ImplicitShear(nominal_U, nominal_TI, Nz, dz, zhub, params["L"])
    mtf.uvw[0, :, :, :] += shear[np.newaxis, np.newaxis, :]
    
    # Plot to compare raw turbulence and constraints
    axes = plt.subplots(3, 1, figsize=(12, 6))[1]
    for ax, c in zip(axes, 'uvw'):
        da.sel(uvw=c, y=mid_y_val, z=z_vals[-1]).plot(ax=ax, label='Raw turbulence')

    # Plot "Raw turbulence" and "Constrained turbulence"
    for i, (ax, c) in enumerate(zip(axes, 'uvw'), 3):
        mtf.to_xarray().sel(uvw=c, y=mid_y_val, z=z_vals[-1]).plot(ax=ax, label='Constrained turbulence')

        ax.plot([], '.k', label='Constraints')
        constraints_filtered = [con for con in constraints if con[1] == mid_y_val and con[2] == z_vals[-1]]

        for con in constraints_filtered:
            ax.plot(con[0], con[i], '.k', markersize=1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"Constraints_{Base_name}{params['seed']}.pdf"), dpi=150, bbox_inches='tight')
    plt.close()

    # Get the turbulence spectra
    k_rea, spectra_rea = mtf.spectra()
    k_int, spectra_int = mtf.spectra_integrated()
    k_lut, spectra_lut = mtf.spectra_lookup()

    # Plot of spectra
    plt.figure()
    for phi_rea, phi_int, phi_lut, c, l in zip(spectra_rea, spectra_int, spectra_lut, c_lst, s_lst):
        plt.semilogx(k_rea, phi_rea*k_rea, color=c, label=l)
        plt.semilogx(k_int, phi_int*k_int, '--', color=c)
        plt.semilogx(k_lut, phi_lut*k_lut, ':', color=c)

    plt.plot([], '-', color='gray', label='Realization')
    plt.plot([], '--', color='gray', label='LookupTable')
    plt.plot([], ':', color='gray', label='Integrated')
    plt.xlabel('Wave number, $k_1$ $  [m^{-1}$]')
    plt.ylabel('$k_1 S(k_1)[m^2s^{-2}]$')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outdir, f"cSpectra_{Base_name}{params['seed']}.pdf"), dpi=150, bbox_inches='tight')
    plt.close()

    # dac with the constrained mtf object for last plots
    dac = mtf.to_xarray()

    # Plot of windfield
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 4)  # Create 4-column grid for flexible width ratio

    # Left subplot: Windfield (occupies 3/4 of figure width)
    ax0 = fig.add_subplot(gs[0, :3])
    dac.sel(uvw='u', y=mid_y_val).plot(x='x', ax=ax0)
    ax0.set_title("Windfield at y = {:.1f} m".format(mid_y_val))
    ax0.grid(True)

    # Right subplot: Vertical wind profile over x at y=90
    ax1 = fig.add_subplot(gs[0, 3], sharey=ax0)
    u_slice = dac.sel(uvw='u').isel(y=mid_y_index)  # Shape: (x, z)

    u_mean = u_slice.mean(dim='x')
    u_std = u_slice.std(dim='x')
    z_vals = dac.coords['z'].values

    ax1.errorbar(u_mean, z_vals, xerr=u_std, fmt='o', color='cornflowerblue', ecolor='cornflowerblue', capsize=3)
    ax1.set_xlabel('U [m/s]')
    ax1.set_ylabel('Height z [m]')
    ax1.set_title("Profile at y = {:.1f} m".format(mid_y_val))
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"Windfield_{Base_name}{params['seed']}.pdf"), dpi=150, bbox_inches='tight')
    plt.close()

    # Topview plot
    fig = plt.figure(figsize=(10, 6))
    dac.sel(uvw='u', z=hub).plot(x='x')

    plt.title(f"Top View of u at z = {hub:.1f} m (hub height)")
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)

    plt.savefig(os.path.join(outdir, f"TopView_{Base_name}{params['seed']}.pdf"), dpi=150, bbox_inches='tight')
    plt.close()

    print("After constraints:")
    print(f"var(u): {np.var(dac.sel(uvw='u').values)}")
    print(f"var(v): {np.var(dac.sel(uvw='v').values)}")
    print(f"var(w): {np.var(dac.sel(uvw='w').values)}")

