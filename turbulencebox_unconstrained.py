"""
Synthesizing an isotropic turbulence box coherent with IEC 61400-1: 2019, utilising Hipersim.

Leonard Riemer - 13/02/2025
"""

# Import relevant libraries
from hipersim import MannTurbulenceField
import pylab as plt
from iec_definitions import *
from common import *
import os.path

# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
fp = lambda x: os.path.join(inputVariables,x)

dtu10mw = loadFromJSON(fp("dtu10mw.json"))  # Includes wind & turbulence class with ref values and hub height

# Define wind parameters based on DTU 10 MW reference
U = dtu10mw["V_ave"]
TI = dtu10mw["I_ref"]

# Define simulation paramters
params = {}
params["L"] = get_length(dtu10mw["zhub"])                   # Turbulence length [m]
params["alphaepsilon"] = get_strength()                     # Turbulence strength [m^(4/3)/s^2], default set to 1 and scale timeseries
params["Gamma"] = get_anisotropy()                          # Anisotropy [-]
params["Nxyz"] = (1024,120,120)                             # Dimensions of turbulence box, for now rather short (4096m correspinding to ca. 400s with U=10m/s)
params["dxyz"] = (2,2,2)                                    # Spacing between points [m]
params["seed"] = 1                                          # Seed for random number gen
params["HighFreqComp"] = 1                                  # Compensation at high frequencies to make sure 5/3 law?
params["double_xyz"] = (False, True, True)                  # Doubling along given axis for bigger box

# Generate turbulence box
mtf = MannTurbulenceField.generate(alphaepsilon = params["alphaepsilon"],
                                   L = params["L"],
                                   Gamma = params["Gamma"],
                                   Nxyz = params["Nxyz"],
                                   dxyz = params["dxyz"],
                                   seed = params["seed"],
                                   HighFreqComp = params["HighFreqComp"],
                                   double_xyz = params["double_xyz"])

# Store turbulence data in variables (uvw correspond to xyz directions)
u,v,w = mtf.uvw

# Scale box to match the IEC wind and turbulence class
print (f'Before: Box TI={mtf.uvw[0].std(0).mean()/U:.3f}, alphaepsilon:{mtf.alphaepsilon:.3f}, theoretical spectrum TI {mtf.spectrum_TI(U):.2f}')
mtf.scale_TI(TI=TI, U=U)
print (f'After: Box TI={mtf.uvw[0].std(0).mean()/U:.3f}, alphaepsilon:{mtf.alphaepsilon:.3f}, theoretical spectrum TI {mtf.spectrum_TI(U):.2f}')
print(f'Iref = {TI}, TI = {mtf.uvw[0].std(0).mean()/U:.3f}; Difference due to uncertainty in lower frequencies and seed-to-seed differences.')

# Export the data to an array
da = mtf.to_xarray() # xarray dataarray

# Get the turbulence spectra
k_rea, spectra_rea = mtf.spectra()
k_int, spectra_int = mtf.spectra_integrated()
k_lut, spectra_lut = mtf.spectra_lookup()

# Colors, labels and plotting
c_lst = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
s_lst = ["uu","vv","ww","uw"]

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

# Variance of uu spectrum vs variance of realization
var_uu_spec = mtf.spectrum_variance()
print("Spectrum Variance", var_uu_spec)
var_uu_real = mtf.uvw[0].var(0).mean()
print("Realization Variance", var_uu_real)

# Plot a front view of the turbulence at x = 0 plane
plt.figure()
da.sel(uvw='u',x=0).plot(y='y')
plt.axis('scaled')
plt.grid()

# Plot a side view of the turbulence at y = 0 plane
plt.figure()
da.sel(uvw='u',y=0).plot(x='x')
plt.axis('scaled')
plt.grid()

# Plot a time series of u at hub height
plt.figure()
da.sel(uvw='u', y=0, z=120).plot(x='x')
plt.grid()

make_plots("unconstr")
