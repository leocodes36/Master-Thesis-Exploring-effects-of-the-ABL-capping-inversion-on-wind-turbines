import sys
import os.path
import matplotlib.pyplot as plt
from hipersim import MannTurbulenceField
from generate_box_common import *

# Main entry point for the script
if __name__ == "__main__":

    # Determine that we have all the arguments, otherwise print a helpful message
    if len(sys.argv)!=17:
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
        print("\tWave structure Height")
        print("\tWave structure Width")

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
    H = float(sys.argv[15])
    W = float(sys.argv[16])

    # Print a helpful message explaining what we are about to do
    print("Generating the turbulence box with dimensions","Nx:",Nx,"Ny:",Ny,"Nz:",Nz,"dx:",dx,"dy:",dy,"dz:",dz,"High-Freq-Comp:",High_freq_comp,"Base-name:",Base_name,"Seed(s):",Seeds_arg)

    # Define simulation paramters
    zhub = 119.0
    params = {}
    params["L"] = get_length(zhub)                  # Turbulence length based on DTU-10MW turbine [m]
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

    # Export the data to an array
    da = mtf.to_xarray()

    # Get constraints array
    R = 90
    H_local = H - zhub + R
    constraints = CoherentWaveConstraint1(da, nominal_U, zhub, params["Nxyz"], params["dxyz"], H_local, W, A, N)

    # Use .constrain() method of the mtf object
    print(f'Constraining u to represent a laterally coherent wave of N:{N}s-^1 A:{A}m/s at H:{H}m over a vertical width W:{W}')
    mtf.constrain(constraints)

    # Add implicit shear of mann model parameters
    print('Adding implicit shear from mann model.')
    shear = ImplicitShear(nominal_U, nominal_TI, Nz, dz, zhub, params["L"])
    mtf.uvw[0, :, :, :] += shear[np.newaxis, np.newaxis, :]

    # Make temporary folder and store resulting binary files
    os.makedirs('tmp', exist_ok=True)
    mtf.to_hawc2(folder='tmp', basename=f'{Base_name}{params["seed"]}_')
    print('Done.')