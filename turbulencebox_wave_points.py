"""
Synthesizing an isotropic turbulence box coherent with IEC 61400-1: 2019, utilising Hipersim.
Constraining it with a synthetic, Kaimal spectrum-based time series at the beginning.

Leonard Riemer - 07/02/2025
"""

# Import relevant libraries
import os.path
from hipersim import MannTurbulenceField
import pylab as plt
from iec_definitions import *
from common import *
from constraint import constraint_wave


# Location of input files
# Shorten the imports
inputVariables = "inputVariables"
fp = lambda x: os.path.join(inputVariables,x)

dtu10mw = loadFromJSON(fp("dtu10mw.json"))  # Includes wind & turbulence class with ref values and hub height

# Define wind parameters based on DTU 10 MW reference
U = dtu10mw["V_ave"]
TI = dtu10mw["I_ref"]

# Reference point
yr, zr = 90, 160

# Define simulation paramters
params = {}
params["L"] = get_length(dtu10mw["zhub"])                   # Turbulence length [m]
params["alphaepsilon"] = get_strength()                     # Turbulence strength [m^(4/3)/s^2], default set to 1 and scale timeseries
params["Gamma"] = get_anisotropy()                          # Anisotropy [-]
params["Nxyz"] = (200,180,180)                             # Dimensions of turbulence box
params["dxyz"] = (2,1,1)                                    # Spacing between points [m]
params["seed"] = 1                                          # Seed for random number gen
params["HighFreqComp"] = 1                                  # Compensation at high frequencies to make sure 5/3 law?
params["double_xyz"] = (False, True, True)                  # Doubling along given axis for bigger box

# Constraint positions
positions = [(52, 160), (128, 160)]
y1, z1 = positions[0]
y2, z2 = positions[1]

# Generate turbulence box
mtf = MannTurbulenceField.generate(alphaepsilon = params["alphaepsilon"],
                                   L = params["L"],
                                   Gamma = params["Gamma"],
                                   Nxyz = params["Nxyz"],
                                   dxyz = params["dxyz"],
                                   seed = params["seed"],
                                   HighFreqComp = params["HighFreqComp"],
                                   double_xyz = params["double_xyz"])

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

# --- Figure 1 ---
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

# --- Figure 2 ---
# Plot a front view of the turbulence at x = 100 plane
plt.figure()
da.sel(uvw='u',x=100).plot(y='z')
plt.axis('scaled')
plt.grid()

# --- Figure 3 ---
# Plot a side view of the turbulence at y1 plane
plt.figure()
da.sel(uvw='u',y=y1).plot(x='x')
#plt.axis('scaled')
plt.grid()

# --- Figure 4 ---
# Plot a side view of the turbulence at reference plane before constraint
plt.figure()
da.sel(uvw='u', y=yr).plot(x='x')
#plt.axis('scaled')
plt.grid()

# Get the constraints data
# Using theta0 = 290 and dtheta/dz = 0.003 based on Kelly M, Cersosimo RA, Berg J. A universal wind profile for the inversion-capped neutral atmospheric boundary layer. Q J R Meteorol Soc. 2019;145:982–992. https://doi.org/10.1002/qj.3472
constraints = constraint_wave(positions, U, params["Nxyz"], params["dxyz"], 290, 0.003)

# --- Figure 5 ---
# Plotting "Raw turbulence" before constraining
axes = plt.subplots(3, 1)[1]
for ax, c in zip(axes, 'uvw'):
    da.sel(uvw=c, y=y1, z=z1).plot(ax=ax, label='Raw turbulence')

# Use .constrain() method of the mtf object
mtf.constrain(constraints)

# Plot "Constraints" (synthetic, Kaimal spectrum-based time series) and "Constrained turbulence"
for i, (ax, c) in enumerate(zip(axes, 'uvw'), 3):
    mtf.to_xarray().sel(uvw=c, y=y1, z=z1).plot(ax=ax, label='Constrained turbulence')

    ax.plot([], '.k', label='Constraints')
    constraints_filtered = [con for con in constraints if con[1] == y1 and con[2] == z1]

    for con in constraints_filtered:
        ax.plot(con[0], con[i], '.k', markersize=1.5)


da1 = mtf.to_xarray()
# --- Figure 6 ---
# Plot a front view of the turbulence at x = 100 plane
plt.figure()
da1.sel(uvw='u',x=100).plot(y='z')
plt.axis('scaled')
plt.grid()

# --- Figure 7 ---
# Plot a side view of the turbulence at y1 plane
plt.figure()
da1.sel(uvw='u',y=y1).plot(x='x')
#plt.axis('scaled')
plt.grid()

# --- Figure 8 ---
# Plot a side view of the turbulence at reference plane after constraint
plt.figure()
da1.sel(uvw='u',y=yr).plot(x='x')
#plt.axis('scaled')
plt.grid()

# --- Figure 9 ---
# Plot a time series at y1, z1 before and after constraint
plt.figure()
da.sel(uvw='u', y=y1, z=z1).plot(x='x', label='before constraint')
da1.sel(uvw='u', y=y1, z=z1).plot(x='x', label='after constraint')
plt.grid()
plt.legend()

# --- Figure 10 ---
# Plot a time series at yr, zr before and after constraint
plt.figure()
da.sel(uvw='u', y=yr, z=zr).plot(x='x', label='before constraint')
da1.sel(uvw='u', y=yr, z=zr).plot(x='x', label='after constraint')
plt.grid()
plt.legend()

make_plots("wave_points")