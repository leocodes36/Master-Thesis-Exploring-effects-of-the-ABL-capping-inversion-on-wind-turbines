import sys
import os.path
import matplotlib.pyplot as plt
from hipersim import MannTurbulenceField
from generate_box_common import *


# Main entry point for the script
if __name__ == "__main__":

    # Determine that we have all the arguments, otherwise print a helpful message
    if len(sys.argv)!=14:
        print("Give me the following, in this order:")
        print("\tThe nominal speed of the flow")
        print("\tThe turbulence intensity of the flow")
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
        print("\tWind Shear")

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
    Seeds_arg = sys.argv[12]
    nominal_shear = float(sys.argv[13])

    # Define Simulation parameters
    zhub = 119.0

    # Extract the seed from the argument
    seed_func_arg = None

    if Seeds_arg and ' ' in Seeds_arg:  # Ensure Seeds_arg is not empty/None
        try:
            seed_func_arg = [int(seed_str) for seed_str in Seeds_arg.split()]
        except ValueError:
            raise ValueError(f"Invalid seed value in input: {Seeds_arg}")
    else:
        try:
            seed_func_arg = int(Seeds_arg)
        except ValueError:
            raise ValueError(f"Invalid seed value: {Seeds_arg}")

    # Make dir for plots
    outdir = "./plots/"
    os.makedirs(outdir, exist_ok=True)

    # Print a helpful message explaining what we are about to do
    print("Generating the turbulence box with added shear according to IEC standard.","Nx:",Nx,"Ny:",Ny,"Nz:",Nz,"dx:",dx,"dy:",dy,"dz:",dz,"High-Freq-Comp:",High_freq_comp,"Base-name:",Base_name,"Seed(s):",seed_func_arg)

    # Define simulation paramters
    params = {}
    params["L"] = get_length(119)                   # Turbulence length based on DTU-10MW turbine [m]
    params["alphaepsilon"] = get_strength()         # Turbulence strength [m^2 s^-3], default set to 1 and scale timeseries
    params["Gamma"] = get_anisotropy()              # Anisotropy [-]
    params["Nxyz"] = (Nx,Ny,Nz)                     # Number of points in box [-]
    params["dxyz"] = (dx,dy,dz)                     # Spacing between points [m]
    params["seed"] = seed_func_arg                  # Seed for random number gen [-]
    params["HighFreqComp"] = High_freq_comp         # Compensation at high frequencies to make sure 5/3 law [-]
    params["double_xyz"] = (True, False, False)     # Doubling along given axis for bigger box [-]

    if isinstance(seed_func_arg, list):
        for s in seed_func_arg:
            # Generate turbulence box
            mtf = MannTurbulenceField.generate(alphaepsilon = params["alphaepsilon"],
                                        L = params["L"],
                                        Gamma = params["Gamma"],
                                        Nxyz = params["Nxyz"],
                                        dxyz = params["dxyz"],
                                        seed = s,
                                        HighFreqComp = params["HighFreqComp"],
                                        double_xyz = params["double_xyz"],
                                        cache_spectral_tensor=True)

            # Scale box to match the IEC wind and turbulence class and print info
            print('Scaling turbulence to match IEC class 1A.')
            print (f'Before: Box TI={mtf.uvw[0].std(0).mean()/nominal_U:.3f}, alphaepsilon:{mtf.alphaepsilon:.3f}, theoretical spectrum TI {mtf.spectrum_TI(nominal_U):.2f}')
            mtf.scale_TI(TI=nominal_TI, U=nominal_U)
            print (f'After: Box TI={mtf.uvw[0].std(0).mean()/nominal_U:.3f}, alphaepsilon:{mtf.alphaepsilon:.3f}, theoretical spectrum TI {mtf.spectrum_TI(nominal_U):.2f}')
            print(f'Iref = {nominal_TI}, TI = {mtf.uvw[0].std(0).mean()/nominal_U:.3f} (Difference due to uncertainty in lower frequencies and seed-to-seed differences.)')


            # Generate a vertical wind speed profile based on the wind shear
            print('Adding shear from power law.')
            shear = PowerLawShear(nominal_U, zhub, Nz, dz, nominal_shear)

            # Add the additional shear profile to the uvw components of the mtf object
            mtf.uvw[0, :, :, :] += shear[np.newaxis, np.newaxis, :]

            # Export the data to an array
            da = mtf.to_xarray() # xarray dataarray

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
            plt.savefig(os.path.join(outdir, f"Spectra_{Base_name}{s}.pdf"), dpi=150, bbox_inches='tight')
            plt.close()

            # Plot of windfield
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(1, 4)  # Create 4-column grid for flexible width ratio

            # Left subplot: Windfield (occupies 3/4 of figure width)
            ax0 = fig.add_subplot(gs[0, :3])
            da.sel(uvw='u', y=(Ny*dy)//2).plot(x='x', ax=ax0)
            ax0.set_title("Windfield at y = {:.1f} m".format((Ny*dy)//2))
            ax0.grid(True)

            # Right subplot: Vertical wind profile over x at y=90
            ax1 = fig.add_subplot(gs[0, 3])
            u_slice = da.sel(uvw='u').isel(y=Ny // 2)  # Shape: (x, z)

            u_mean = u_slice.mean(dim='x')
            u_std = u_slice.std(dim='x')
            z_vals = da.coords['z'].values

            ax1.errorbar(u_mean, z_vals, xerr=u_std, fmt='o', color='cornflowerblue', ecolor='cornflowerblue', capsize=3)
            ax1.set_xlabel('U [m/s]')
            ax1.set_ylabel('Height z [m]')
            ax1.set_title("Profile at y = {:.1f} m".format((Ny*dy)//2))
            ax1.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"Windfield_{Base_name}{s}.pdf"), dpi=150, bbox_inches='tight')
            plt.close()
    else:
            # Generate turbulence box
            mtf = MannTurbulenceField.generate(alphaepsilon = params["alphaepsilon"],
                                        L = params["L"],
                                        Gamma = params["Gamma"],
                                        Nxyz = params["Nxyz"],
                                        dxyz = params["dxyz"],
                                        seed = params["seed"],
                                        HighFreqComp = params["HighFreqComp"],
                                        double_xyz = params["double_xyz"])

            # Scale box to match the IEC wind and turbulence class and print info
            print('Scaling turbulence to match IEC class 1A.')
            print (f'Before: Box TI={mtf.uvw[0].std(0).mean()/nominal_U:.3f}, alphaepsilon:{mtf.alphaepsilon:.3f}, theoretical spectrum TI {mtf.spectrum_TI(nominal_U):.2f}')
            mtf.scale_TI(TI=nominal_TI, U=nominal_U)
            print (f'After: Box TI={mtf.uvw[0].std(0).mean()/nominal_U:.3f}, alphaepsilon:{mtf.alphaepsilon:.3f}, theoretical spectrum TI {mtf.spectrum_TI(nominal_U):.2f}')
            print(f'Iref = {nominal_TI}, TI = {mtf.uvw[0].std(0).mean()/nominal_U:.3f} (Difference due to uncertainty in lower frequencies and seed-to-seed differences.)')

            # Generate a CNBL profile based on the case
            print('Adding shear from power law.')
            shear = PowerLawShear(nominal_U, zhub, Nz, dz, nominal_shear)

            # Add the additional shear profile to the uvw components of the mtf object
            mtf.uvw[0, :, :, :] += shear[np.newaxis, np.newaxis, :]

            # Export the data to an array
            da = mtf.to_xarray() # xarray dataarray

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

            # Plot of windfield
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(1, 4)  # Create 4-column grid for flexible width ratio

            # Left subplot: Windfield (occupies 3/4 of figure width)
            ax0 = fig.add_subplot(gs[0, :3])
            da.sel(uvw='u', y=(Ny*dy)//2).plot(x='x', ax=ax0)
            ax0.set_title("Windfield at y = {:.1f} m".format((Ny*dy)//2))
            ax0.grid(True)

            # Right subplot: Vertical wind profile over x at y=90
            ax1 = fig.add_subplot(gs[0, 3])
            u_slice = da.sel(uvw='u').isel(y=Ny // 2)  # Shape: (x, z)

            u_mean = u_slice.mean(dim='x')
            u_std = u_slice.std(dim='x')
            z_vals = da.coords['z'].values

            ax1.errorbar(u_mean, z_vals, xerr=u_std, fmt='o', color='cornflowerblue', ecolor='cornflowerblue', capsize=3)
            ax1.set_xlabel('U [m/s]')
            ax1.set_ylabel('Height z [m]')
            ax1.set_title("Profile at y = {:.1f} m".format((Ny*dy)//2))
            ax1.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"Windfield_{Base_name}{params['seed']}.pdf"), dpi=150, bbox_inches='tight')
            plt.close()

            print(f"var(u): {np.var(da.sel(uvw='u').values)}")
            print(f"var(v): {np.var(da.sel(uvw='v').values)}")
            print(f"var(w): {np.var(da.sel(uvw='w').values)}")