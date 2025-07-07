import sys
import os.path
from hipersim import MannTurbulenceField
from generate_box_common import *


# Main entry point for the script
if __name__ == "__main__":

    # Determine that we have all the arguments, otherwise print a helpful message
    if len(sys.argv)!=13:
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

    # Define Simulation parameters
    zhub = 119.0
    latitude = 55
    theta0 = 290
    dthetadz = 0.003

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

    # Print a helpful message explaining what we are about to do
    print("Generating the turbulence box with added shear for a CNBL profile and dimesnions","Nx:",Nx,"Ny:",Ny,"Nz:",Nz,"dx:",dx,"dy:",dy,"dz:",dz,"High-Freq-Comp:",High_freq_comp,"Base-name:",Base_name,"Seed(s):",seed_func_arg)

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


            # Generate a CNBL profile based on the case
            print('Adding shear from jet.')
            shear = GenProfile(nominal_U, zhub, latitude, theta0, dthetadz)

            # Add the additional shear profile to the uvw components of the mtf object
            mtf.uvw[0, :, :, :] += shear[np.newaxis, np.newaxis, :]

            # Make temporary folder and store resulting binary files
            os.makedirs('tmp', exist_ok=True)
            mtf.to_hawc2(folder='tmp', basename=f'{Base_name}{s}_')
            print(f'Seed {s} out {len(seed_func_arg)} done.')
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
            print('Adding shear from jet.')
            shear = GenProfile(nominal_U, zhub, latitude, theta0, dthetadz)

            # Add the additional shear profile to the uvw components of the mtf object
            mtf.uvw[0, :, :, :] += shear[np.newaxis, np.newaxis, :]

            # Make temporary folder and store resulting binary files
            os.makedirs('tmp', exist_ok=True)
            mtf.to_hawc2(folder='tmp', basename=f'{Base_name}{params["seed"]}_')
            print(f'Done.')
