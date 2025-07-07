"""Library of common .py functions for thesis -- Leonard Riemer -- 19/05/2025"""
import numpy as np
import matplotlib.pyplot as plt

def get_length(zhub):
    """
    Computes the turbulence length scale based on hub height.
    
    Parameters:
    zhub (float): Hub height in meters
    
    Returns:
    float: Turbulence length scale (m)
    
    Source: IEC-61400-1:2019 6.3.1 Eq.(5), Annex C.2 Eq.(C.12)
    """
    if zhub <= 60:
        Lambda = 0.7 * zhub
    else:
        Lambda = 42
    return 0.8 * Lambda

def get_strength():
    """
    Returns the turbulence intensity parameter for wind turbulence modeling.
    
    Returns:
    int: Turbulence intensity parameter
    
    Source: IEC-61400-1:2019 Annex C.2 Eq.(C.12)
    """
    return 1

def get_anisotropy():
    """
    Returns the anisotropy factor for turbulence modeling.
    
    Returns:
    float: Anisotropy factor
    
    Source: IEC-61400-1:2019 Annex C.2 Eq.(C.12)
    """
    return 3.9

def fCoriolis(latitude_deg):
    """
    Computes the Coriolis parameter based on latitude.
    
    Parameters:
    latitude_deg (float): Latitude in degrees
    
    Returns:
    float: Coriolis parameter (s⁻¹)
    
    Source: BERG et al. (2022), DTU 46100
    """
    return 2 * 7.2921e-5 * np.sin(np.radians(latitude_deg))

def LogLawustar(U, z, z0, kappa=0.4):
    """
    Estimates friction velocity (ustar) from wind speed using the logarithmic wind profile.
    
    Parameters:
    U (float): Wind speed at height z (m/s)
    z (float): Measurement height (m)
    z0 (float): Roughness length (m)
    kappa (float, optional): von Kármán constant (default: 0.4)
    
    Returns:
    float: Friction velocity u_* (m/s)
    
    Source: BERG et al. (2022), DTU 46100
    """
    return (U * kappa) / np.log(z / z0)

def LogLawU(ustar, z, z0, kappa=0.4):
    """
    Computes wind speed at height z using the logarithmic wind profile.
    
    Parameters:
    ustar (float): Friction velocity (m/s)
    z (float): Height above ground (m)
    z0 (float): Roughness length (m)
    kappa (float, optional): von Kármán constant (default: 0.4)
    
    Returns:
    float: Wind speed U (m/s)
    
    Source: BERG et al. (2022), DTU 46100
    """
    return (ustar / kappa) * np.log(z / z0)

def BruntVäisälä(theta0, dthetadz, g=9.81):
    """
    Computes the Brunt–Väisälä frequency, indicating atmospheric stability.
    
    Parameters:
    theta0 (float): Potential temperature at reference height (K)
    dthetadz (float): Vertical temperature gradient (K/m)
    g (float, optional): Acceleration due to gravity (m/s², default: 9.81)
    
    Returns:
    float: Brunt–Väisälä frequency N (s⁻¹)
    
    Source: KELLY et al. (2019)
    """
    return np.sqrt((g / theta0) * dthetadz)

def GDL(ustar, fc, z0, kappa=0.4, A=1.8, B=4.5):
    """
    Computes the geostrophic wind speed using the geostrophic drag law.
    
    Parameters:
    ustar (float): Friction velocity (m/s)
    fc (float): Coriolis parameter (s⁻¹)
    z0 (float): Roughness length (m)
    kappa (float, optional): von Kármán constant (default: 0.4)
    A (float, optional): Empirical constant (default: 1.8)
    B (float, optional): Empirical constant (default: 4.5)
    
    Returns:
    float: Geostrophic wind speed G (m/s)
    
    Source: BERG et al. (2022), DTU 46100
    """
    return ustar / kappa * np.sqrt((np.log((ustar / fc) / z0) - A)**2 + B**2)

def ABLHeight(ustar, f, N):
    """
    Computes the atmospheric boundary layer (ABL) height.
    
    Parameters:
    ustar (float): Friction velocity (m/s)
    f (float): Coriolis parameter (s⁻¹)
    N (float): Brunt–Väisälä frequency (s⁻¹)
    
    Returns:
    float: Boundary layer height h_e (m)
    
    Source: ZILITINKEVICH et al. (2012)
    """
    CCN, CR = 1.36, 0.6
    Ch = CCN * (f / N)**0.5  
    return Ch * ustar / f

def JetWidth(ustar, f, N):
    """
    Computes the width of a supergeostrophic jet in a conventionally neutral boundary layer (CNBL).
    
    Parameters:
    ustar (float): Friction velocity (m/s)
    f (float): Coriolis parameter (s⁻¹)
    N (float): Brunt–Väisälä frequency (s⁻¹)
    
    Returns:
    float: Jet width (m)
    
    Source: LIBIANCHI et al. (2023)
    """
    epsilon, CCN = 0.12, 1.36
    return 2 * epsilon * CCN * ustar / ((1 - 0.05**(2/3)) * np.sqrt(f * N))

def GenProfile(U, zhub, lat, theta0, dthetadz):
    """
    Generates a wind speed profile including a supergeostrophic jet in a conventionally neutral boundary layer (CNBL),
    based on log-law scaling and an added Gaussian-shaped jet.

    Parameters:
    U (float): Reference wind speed at hub height (m/s)
    zhub (float): Hub height (m)
    lat (float): Latitude (degrees)
    theta0 (float): Reference potential temperature (K)
    dthetadz (float): Vertical gradient of potential temperature (K/m)

    Returns:
    np.ndarray: Wind shear box (array of wind speed deviations) across ±90 m from hub height
    """
    # Calculation of log-law for Reference wind speed at hub height
    z0 = 0.0001                         # Roughness length over water, also possible through Charnock relation.
    ustar = LogLawustar(U, zhub, z0)    # Friction velocity defined at the surface (For now just fitted to U=10 m/s at hub height, Else possibly average from Sprog 0.2580)

    # Define vertical space
    Nz = 1000                           # 1000 grid points
    dz = 1                              # Spacing of dz 1 [m]
    z = np.arange(dz, Nz*dz, dz)        # vertical space array

    LogLaw = LogLawU(ustar, z, z0)

    # Calculations for ABL Height
    f = fCoriolis(lat)                  # Coriolis paramter based on latitude of 55°
    N = BruntVäisälä(theta0, dthetadz)  # Brunt Väisälä Frequency, using theta0 = 290 and dtheta/dz = 0.003 based on Kelly M, Cersosimo RA, Berg J. (2019)
    he = ABLHeight(ustar, f, N)         # ABL Height based on Zilitinkevich (2012)

    # Calculations for Geostrophic wind -> Jet strength
    G = GDL(ustar, f, z0)                       # Geostrophic Drag Law Troen, I. and Petersen, E. L. (1989)
    UJet =  1.07 * G - np.interp(he, z, LogLaw) # Approximation according to PEDERSEN, J. G., Gryning, S.-E., and Kelly, M. (2014)
    Width = JetWidth(ustar, f, N)               # Relation according to Libianchi (2023)
    
    LogLaw = np.where(LogLaw <= G, LogLaw, G)   # Make sure that base profile doesn't exceed G

    # Adding a Gaussian shaped jet at ABL height
    Jet = LogLaw + UJet * np.exp(-((z - he) / Width)**2)

    # Sample from hub height - radius + radius
    lower, upper = int(zhub - 1 - 90), int(zhub - 1 + 90)
    shearBox = Jet[lower:upper] - U

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
    shearBox = PowerLaw[lower:upper] - U

    return shearBox

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
    shearBox = Jet[lower:upper] - U

    return shearBox

def CoherentWaveConstraint(U, TurbNxyz, Turbdxyz, Height, Field, Amp, N, n_con_sample=2):
    """
    Generates a spatiotemporal constraint array representing a coherent wave disturbance
    across all lateral (y) positions within a specified vertical height interval in a 
    turbulence grid. Intended for use with older version of HiperSim/TurbGen module.

    The function simulates a sinusoidal longitudinal velocity disturbance (u') evolving
    in time, coherent across y at selected z levels.

    Parameters:
    U (float): Mean longitudinal flow speed (m/s)
    TurbNxyz (tuple): Number of grid points in (x, y, z) directions (Nx, Ny, Nz)
    Turbdxyz (tuple): Grid spacing in (x, y, z) directions (dx, dy, dz) (m)
    Height (float): Center height (z) of the vertical constraint band (m)
    Field (float): Total vertical extent (height) of the constraint region (m)
    Amp (float): Amplitude of the sinusoidal disturbance (m/s)
    N (float): Buoyancy (Brunt-Väisälä) frequency in Hz
    n_con_sample (int): Subsampling factor for constraint array (default is 2)

    Returns:
    np.ndarray: Array of shape (n_points, 4) containing constraint entries with columns:
                [x, y, z, u'] — where u' is the wave-induced velocity perturbation

    Raises:
    ValueError: If the specified vertical interval is out of grid bounds or if
                sampling frequency violates wave resolution constraints (k·dx ≳ 1)

    Notes:
    - The constraint is coherent in time (sinusoidal) and space (uniform across y),
      within the vertical band defined by Height ± Field/2.
    - To ensure the wave is properly resolved, k·dx should be ≪ 1.
    """
    
    # Define Parameters
    Nx, Ny, Nz = TurbNxyz
    dx, dy, dz = Turbdxyz

    # Brunt-Väisälä frequency -> wave number
    k = 2 * np.pi / (U / N)
    print(f"Wave frequency fw {np.round(N, 3)} [Hz]; Wave number k {np.round(k, 3)} [m^-1]")

    # Define Temporal Grid
    dt = dx / U
    Nt = Nx
    t_start = 0.1 * Nt * dt
    t = np.arange(t_start, Nt * dt, dt)

    # Define Spatial Grid
    y = np.arange(0, Ny * dy, dy)
    z = np.arange(0, Nz * dz, dz)
    Y, Z = np.meshgrid(y, z, indexing='ij')

    # Generate sinusoidal time-series turbulence
    u_turb = Amp * np.sin(2 * np.pi * N * t)

    # Determine z bounds for constraint region
    lower_z = Height - Field / 2
    upper_z = Height + Field / 2
    if lower_z < 0 or upper_z > Nz * dz:
        raise ValueError(f"Height range {lower_z:.2f}–{upper_z:.2f}m out of z bounds 0–{Nz*dz:.2f}m")

    # Identify z positions within range
    z_indices = np.where((z >= lower_z) & (z <= upper_z))[0]
    if len(z_indices) == 0:
        raise ValueError(f"No grid z-values found between {lower_z:.2f} and {upper_z:.2f}")

    # Constraint across all y positions and selected z positions
    constraint_positions = [(yy, zz) for yy in y for zz in z[z_indices]]

    # Generate time-dependent constraints
    constraints_list = []
    for ypos, zpos in constraint_positions:
        Uconstraints = u_turb.reshape(-1, 1)
        Xconstraints = (t[:, None] * U)
        Yconstraints = np.full_like(Xconstraints, ypos)
        Zconstraints = np.full_like(Xconstraints, zpos)
        constraints_data = np.hstack((Xconstraints, Yconstraints, Zconstraints, Uconstraints))
        constraints_list.append(constraints_data)

    constraints_array = np.vstack(constraints_list)

    # Subsample constraint array
    if not isinstance(n_con_sample, int):
        raise ValueError("n_con_sample must be an integer.")

    constraints_array = constraints_array[::n_con_sample]
    dtCon = n_con_sample * dx / U
    fCon = 1 / dtCon
    dxCon = U * dtCon

    print(f"Frequency of constraint {np.round(fCon, 3)} [Hz]")
    print(f"dt of constraint {np.round(dtCon, 3)} [s]")
    print(f"dx of constraint {np.round(dxCon, 3)} [m]; k*dx {np.round(k*dxCon, 3)}")

    if k * dxCon >= 0.4:
        raise ValueError("k*dxCon should be << 1 to ensure a wave signal is apparent!")

    """# Plots
    plt.figure()
    plt.scatter(Y.flatten(), Z.flatten(), color='lightgray', s=10, label='Mesh')
    constraint_yz = np.array(constraint_positions)
    plt.scatter(constraint_yz[:, 0], constraint_yz[:, 1], color='blue', s=10, label='Constraints')
    plt.gca().set_aspect('equal')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.legend()
    plt.title('Wave Constraint Positions (all y, limited z)')
    
    # Plot u'(x) at center height
    z_center = Height
    mask = np.isclose(constraints_array[:, 2], z_center)
    x_vals = constraints_array[mask, 0]
    u_vals = constraints_array[mask, 3]

    plt.figure()
    plt.scatter(x_vals, u_vals, label=f'z = {z_center} m')
    plt.xlabel('x [m]')
    plt.ylabel("u' [m/s]")
    plt.title("Wave Signal at Center Height")
    plt.grid(True)
    plt.legend()
    plt.show()"""

    return constraints_array

def CoherentWaveConstraint1(mtf, U, zhub, TurbNxyz, Turbdxyz, Height, Width, Amp, N, n_con_sample=18):
    """
    Constructs a 6D spatiotemporal constraint array representing a coherent wave-like
    velocity disturbance within a turbulent flow field, superimposed on pre-existing 
    Mann turbulence data.

    This function generates a longitudinal sinusoidal wave (u') that evolves in time
    and extracts the transverse (v) and vertical (w) velocity components from the
    turbulence field (`mtf`) at corresponding locations. The wave signal starts 10%
    into the time domain to avoid spin-up effects.

    Parameters:
    ----------
    mtf : xr.DataArray
        4D Mann turbulence field with dimensions (uvw, x, y, z).
        Must contain 'u', 'v', and 'w' components. The x-dimension is assumed
        to represent time, scaled by the mean wind speed U.
    U : float
        Mean longitudinal wind speed (m/s).
    zhub : float
        Unused in this function, included for interface compatibility. Assumes mtf is centered around hub height.
    TurbNxyz : tuple of int
        Number of grid points in (x, y, z) directions (Nx, Ny, Nz).
    Turbdxyz : tuple of float
        Grid spacing in (x, y, z) directions (dx, dy, dz) in meters.
    Height : float
        Center height of the vertical constraint band (m).
    Width : float
        Vertical extent (width) of the constraint region (m).
    Amp : float
        Amplitude of the imposed sinusoidal wave disturbance (m/s).
    N : float
        Buoyancy (Brunt-Väisälä) frequency of the wave (Hz).
    n_con_sample : int, optional (default=6)
        Initial temporal subsampling factor for the constraint.
        The function will adapt this value downward automatically
        if it does not sufficiently resolve the wave.

    Returns:
    -------
    constraints_array : np.ndarray
        Array of shape (n_points, 6) with columns:
        [x, y, z, u', v, w], where:
        - x, y, z are spatial coordinates (m)
        - u' is the imposed longitudinal wave disturbance (m/s)
        - v, w are the transverse and vertical components from the Mann field (m/s)

    Notes:
    -----
    - The wave signal is sinusoidal in time and coherent across y and selected z positions.
    - Only the last 90% of the x/time domain is used to avoid spin-up effects.
    - The constraint is automatically resampled to ensure sufficient spatial resolution:
        * The spatial resolution dxCon = n_con_sample * dx must satisfy:
            k · dxCon < 0.1.571  -> at least 4 points per wavelength.
        * If not, n_con_sample is reduced iteratively until this is met or reaches 1.
    - A ValueError is raised if no suitable resolution is found (e.g., too coarse a grid or too high a frequency).
    """
    
    # Define Parameters
    Nx, Ny, Nz = TurbNxyz
    dx, dy, dz = Turbdxyz

    # Brunt-Väisälä frequency -> wave number
    k = 2 * np.pi / (U / N)
    print(f"Wave frequency fw {np.round(N, 3)} [Hz]; Wave number k {np.round(k, 3)} [m^-1]")

    # Define Temporal Grid
    dt = dx / U
    Nt = Nx
    t_start = 0.1 * Nt * dt
    t_end = Nt * dt
    t = np.arange(t_start, t_end, dt)
    x_start_index = int(t_start / dt)
    x_end_index = int(t_end / dt)   


    # Define Spatial Grid
    y = np.arange(0, Ny * dy, dy)
    z = np.arange(0, Nz * dz, dz)
    Y, Z = np.meshgrid(y, z, indexing='ij')

    # Determine suitable n_con_sample by checking wave resolution
    if not isinstance(n_con_sample, int):
        raise ValueError("n_con_sample must be an integer.")

    while n_con_sample >= 1:
        dtCon = n_con_sample * dx / U
        dxCon = U * dtCon
        fCon = 1 / dtCon
        kdx = k * dxCon

        if kdx < 0.785:
            break
        else:
            print(f"Warning: k*dxCon = {kdx:.3f} too large, decreasing n_con_sample to improve resolution.")
            n_con_sample -= 1

    if n_con_sample < 1:
        raise ValueError("Unable to find suitable n_con_sample to resolve the wave. Try reducing N or increasing resolution.")

    # Print final constraint sampling info
    print(f"Final n_con_sample: {n_con_sample}")
    print(f"Frequency of constraint {np.round(fCon, 3)} [Hz]")
    print(f"dt of constraint {np.round(dtCon, 3)} [s]")
    print(f"dx of constraint {np.round(dxCon, 3)} [m]; k*dx {np.round(kdx, 3)}")

    # Generate sinusoidal time-series turbulence
    u_turb = Amp * np.sin(2 * np.pi * N * t)

    # Determine z bounds for constraint region
    lower_z = Height - Width / 2
    upper_z = Height + Width / 2
    if lower_z < 0 or upper_z > Nz * dz:
        raise ValueError(f"Height range {lower_z:.2f}–{upper_z:.2f}m out of z bounds 0–{Nz*dz:.2f}m")

    # Identify z positions within range
    z_indices = np.where((z >= lower_z) & (z <= upper_z))[0]
    if len(z_indices) == 0:
        raise ValueError(f"No grid z-values found between {lower_z:.2f} and {upper_z:.2f}")

    # Constraint across all y positions and selected z positions
    constraint_positions = [(yy, zz) for yy in y for zz in z[z_indices]]

    # Generate time-dependent constraints
    constraints_list = []
    for ypos, zpos in constraint_positions:
        Uconstraints = u_turb.reshape(-1, 1)
        Vconstraints = mtf.sel(uvw='v', y=ypos, z=zpos).isel(x=slice(x_start_index, x_end_index)).values.reshape(-1, 1)
        Wconstraints = mtf.sel(uvw='w', y=ypos, z=zpos).isel(x=slice(x_start_index, x_end_index)).values.reshape(-1, 1)
        Xconstraints = (t[:, None] * U)
        Yconstraints = np.full_like(Xconstraints, ypos)
        Zconstraints = np.full_like(Xconstraints, zpos)
        constraints_data = np.hstack((Xconstraints, Yconstraints, Zconstraints, Uconstraints, Vconstraints, Wconstraints))
        constraints_list.append(constraints_data)

    constraints_array = np.vstack(constraints_list)

    # Subsample constraint array
    constraints_array = constraints_array[::n_con_sample]

    return constraints_array

def getAlpha(WindSpeeds, zhub, dz):
    """
    Estimates the power-law shear exponent (alpha) from a vertical wind speed profile.

    This function fits a power-law model to the input wind speed profile using 
    a log-log regression, assuming the vertical coordinates are evenly spaced around 
    the hub height.

    Parameters:
    WindSpeeds (array-like): 1D array of wind speed values sampled vertically.
    zhub (float):            Hub height (m) around which the wind speeds are centered.
    dz (float):              Vertical spacing between wind speed samples (m).

    Returns:
    float: Estimated power-law shear exponent (alpha), representing how wind speed 
           changes with height according to the power-law model.
    """
    z = np.arange(zhub - (len(WindSpeeds) * dz) // 2, 
              zhub + (len(WindSpeeds) * dz) // 2, dz)
    alpha, intercept = np.polyfit(np.log(z),np.log(WindSpeeds), 1)
    return alpha

def ImplicitShear(U, TI, TurbNz, Turbdz, zhub, TurbLength):
    """
    Estimates a vertical wind shear profile using turbulence intensity and a 
    Mann-model-based length scale, following the approach of Kelly (2018), and 
    returns the deviation from hub-height wind speed.

    The function assumes linear shear, where the vertical shear gradient (dU/dz) 
    is estimated from the nominal wind speed (U), turbulence intensity (TI), 
    and a characteristic turbulence length scale (TurbLength), based on a formulation 
    presented in Kelly (2018). A power-law exponent (alpha) is then fitted to 
    this linear profile to construct a full vertical wind speed distribution. 
    The function extracts a window centered at hub height to represent the shear box.

    Parameters:
    U (float):        Nominal (hub-height) wind speed (m/s).
    TI (float):       Turbulence intensity at hub height (unitless).
    TurbNz (int):     Number of vertical grid points in the turbulence domain.
    Turbdz (float):   Vertical spacing of turbulence grid (m).
    zhub (float):     Hub height of the wind turbine (m).
    TurbLength (float): Turbulence length scale [m], typically from Mann model settings.

    Returns:
    np.ndarray: 1D array representing wind speed deviations [m/s] from hub-height 
                wind speed, sampled over a vertical range of size `TurbNz`.

    Notes:
    - dU/dz is computed as: (U × TI) / TurbLength, following Kelly (2018).
    - The power-law profile is reconstructed using a log-log fit of the linear profile.
    - The result can be used as an implicit shear input in synthetic turbulence simulations.
    """
    # Define vertical space
    Nz = 1000                           # 1000 grid points
    dz = Turbdz                         # Spacing of dz 1 [m]
    z = np.arange(dz, Nz*dz, dz)        # vertical space array

    # Find shear exponent
    dUdz = (U * TI) / (TurbLength)
    LinearShear = U + (z - zhub) * dUdz

    # Sample from hub height ± half of the turbulence box
    TurbRadius = TurbNz // 2
    hub_index = int(zhub / dz)
    lower, upper = max(0, int(hub_index - TurbRadius)), min(len(LinearShear), int(hub_index + TurbRadius))

    # Find shear exponent over rotor
    alpha = getAlpha(LinearShear[lower:upper], zhub, dz)

    # Make vertical wind speed profile and add gaussian jet
    PowerLaw = U * (z / zhub)**alpha
    shearBox = PowerLaw[lower:upper] - U

    return shearBox

def CoherentWaveConstraint2(mtf, U, zhub, TurbNxyz, Turbdxyz, Amp, N, n_con_sample=6):
    """
    Constructs a 6D spatiotemporal constraint array representing a coherent wave-like
    velocity disturbance within a turbulent flow field, superimposed on a pre-existing 
    Mann turbulence data cube.

    This version imposes a longitudinal sinusoidal wave (`u'`) that evolves over time
    at the top of the turb box and samples multiple lateral positions across the `y`
    direction. The disturbance is coherent across these lateral positions.

    Parameters
    ----------
    mtf : xr.DataArray
        4D Mann turbulence field with dimensions ('uvw', 'x', 'y', 'z').
        Must contain 'u', 'v', and 'w' components.
        The x-dimension is assumed to represent time, scaled by mean wind speed U.
    U : float
        Mean longitudinal wind speed (m/s).
    zhub : float
        Not used in this function. Included for compatibility.
    TurbNxyz : tuple of int
        Number of grid points in (x, y, z) directions: (Nx, Ny, Nz).
    Turbdxyz : tuple of float
        Grid spacing in (x, y, z) directions: (dx, dy, dz) in meters.
    Amp : float
        Amplitude of the imposed sinusoidal wave disturbance (m/s).
    N : float
        Buoyancy (Brunt-Väisälä) frequency of the wave (Hz).
    n_con_sample : int, optional (default=18)
        Initial subsampling factor for the constraint in time (x-direction).
        Automatically adjusted downward to ensure sufficient wave resolution.

    Returns
    -------
    constraints_array : np.ndarray
        Array of shape (n_points, 6), where columns are:
        [x, y, z, u', v, w], with:
            x : streamwise coordinate (m)
            y : lateral coordinate (m)
            z : vertical coordinate (m)
            u' : imposed longitudinal velocity disturbance (m/s)
            v : transverse component from Mann field (m/s)
            w : vertical component from Mann field (m/s)

    Notes
    -----
    - The imposed wave is coherent across multiple y-locations and fixed at the z-level
      closest to the specified Height.
    - Only the last 90% of the x/time domain is used to avoid initial spin-up effects.
    - The constraint is sampled every 3rd y-position to reduce density and redundancy.
    - The wave resolution is automatically validated:
        * The constraint spacing dx_con must satisfy: k * dx_con < 0.785
        * If not, the function iteratively reduces `n_con_sample` until this condition is met.
    - A ValueError is raised if no valid sampling factor is found.

    """
    # Define Parameters
    Nx, Ny, Nz = TurbNxyz
    dx, dy, dz = Turbdxyz

    # Brunt-Väisälä frequency -> wave number
    k = 2 * np.pi / (U / N)
    print(f"Wave frequency fw {np.round(N, 3)} [Hz]; Wave number k {np.round(k, 3)} [m^-1]")

    # Define Temporal Grid
    dt = dx / U
    Nt = Nx
    t_start = 0.1 * Nt * dt
    t_end = Nt * dt
    t = np.arange(t_start, t_end, dt)
    x_start_index = int(t_start / dt)
    x_end_index = int(t_end / dt)   


    # Define Spatial Grid
    y = np.arange(0, Ny * dy, dy)
    z = np.arange(0, Nz * dz, dz)
    Y, Z = np.meshgrid(y, z, indexing='ij')

    # Determine suitable n_con_sample by checking wave resolution
    if not isinstance(n_con_sample, int):
        raise ValueError("n_con_sample must be an integer.")

    while n_con_sample >= 1:
        dtCon = n_con_sample * dx / U
        dxCon = U * dtCon
        fCon = 1 / dtCon
        kdx = k * dxCon

        if kdx < 0.785:
            break
        else:
            print(f"Warning: k*dxCon = {kdx:.3f} too large, decreasing n_con_sample to improve resolution.")
            n_con_sample -= 1

    if n_con_sample < 1:
        raise ValueError("Unable to find suitable n_con_sample to resolve the wave. Try reducing N or increasing resolution.")

    # Print final constraint sampling info
    print(f"Final n_con_sample: {n_con_sample}")
    print(f"Frequency of constraint {np.round(fCon, 3)} [Hz]")
    print(f"dt of constraint {np.round(dtCon, 3)} [s]")
    print(f"dx of constraint {np.round(dxCon, 3)} [m]; k*dx {np.round(kdx, 3)}")

    # Generate sinusoidal time-series turbulence
    u_turb = Amp * np.sin(2 * np.pi * N * t)

    # Use the highest z position in the grid
    z_idx = -1
    zpos = z[z_idx] 
    # technically zvals = mtf['z'].values and then zvals[-1] would be better

    # Construct constraint positions: every 3rd y-position at fixed z
    constraint_positions = [(yy, zpos) for yy in y[::3]]

    # Generate time-dependent constraints
    constraints_list = []
    for ypos, zpos in constraint_positions:
        Uconstraints = u_turb.reshape(-1, 1)
        Vconstraints = mtf.sel(uvw='v', y=ypos, z=zpos).isel(x=slice(x_start_index, x_end_index)).values.reshape(-1, 1)
        Wconstraints = mtf.sel(uvw='w', y=ypos, z=zpos).isel(x=slice(x_start_index, x_end_index)).values.reshape(-1, 1)
        Xconstraints = (t[:, None] * U)
        Yconstraints = np.full_like(Xconstraints, ypos)
        Zconstraints = np.full_like(Xconstraints, zpos)
        constraints_data = np.hstack((Xconstraints, Yconstraints, Zconstraints, Uconstraints, Vconstraints, Wconstraints))
        constraints_list.append(constraints_data)

    constraints_array = np.vstack(constraints_list)

    # Subsample constraint array
    constraints_array = constraints_array[::n_con_sample]

    return constraints_array

def CoherentWaveConstraint3(mtf, U, zhub, TurbNxyz, Turbdxyz, Amp, N, n_con_sample=6):
    """
    Constructs a 6D spatiotemporal constraint array representing a coherent wave-like
    velocity disturbance within a turbulent flow field, superimposed on a pre-existing 
    Mann turbulence data cube.

    This version imposes a longitudinal sinusoidal wave (`u'`) that evolves over time
    at the top of the turb box and samples multiple lateral positions across the `y`
    direction. The disturbance is coherent across these lateral positions.

    Parameters
    ----------
    mtf : xr.DataArray
        4D Mann turbulence field with dimensions ('uvw', 'x', 'y', 'z').
        Must contain 'u', 'v', and 'w' components.
        The x-dimension is assumed to represent time, scaled by mean wind speed U.
    U : float
        Mean longitudinal wind speed (m/s).
    zhub : float
        Not used in this function. Included for compatibility.
    TurbNxyz : tuple of int
        Number of grid points in (x, y, z) directions: (Nx, Ny, Nz).
    Turbdxyz : tuple of float
        Grid spacing in (x, y, z) directions: (dx, dy, dz) in meters.
    Amp : float
        Amplitude of the imposed sinusoidal wave disturbance (m/s).
    N : float
        Buoyancy (Brunt-Väisälä) frequency of the wave (Hz).
    n_con_sample : int, optional (default=18)
        Initial subsampling factor for the constraint in time (x-direction).
        Automatically adjusted downward to ensure sufficient wave resolution.

    Returns
    -------
    constraints_array : np.ndarray
        Array of shape (n_points, 6), where columns are:
        [x, y, z, u', v, w], with:
            x : streamwise coordinate (m)
            y : lateral coordinate (m)
            z : vertical coordinate (m)
            u' : imposed longitudinal velocity disturbance (m/s)
            v : transverse component from Mann field (m/s)
            w : vertical component from Mann field (m/s)

    Notes
    -----
    - The imposed wave is coherent across multiple y-locations and fixed at the z-level
      closest to the specified Height.
    - Only the last 90% of the x/time domain is used to avoid initial spin-up effects.
    - The constraint is sampled every 3rd y-position to reduce density and redundancy.
    - The wave resolution is automatically validated:
        * The constraint spacing dx_con must satisfy: k * dx_con < 0.785
        * If not, the function iteratively reduces `n_con_sample` until this condition is met.
    - A ValueError is raised if no valid sampling factor is found.

    """
    # Define Parameters
    Nx, Ny, Nz = TurbNxyz
    dx, dy, dz = Turbdxyz

    # Brunt-Väisälä frequency -> wave number
    k = 2 * np.pi / (U / N)
    print(f"Wave frequency fw {np.round(N, 3)} [Hz]; Wave number k {np.round(k, 3)} [m^-1]")

    # Define Temporal Grid
    dt = dx / U
    Nt = Nx
    t_start = 0.1 * Nt * dt
    t_end = Nt * dt
    t = np.arange(t_start, t_end, dt)
    x_start_index = int(t_start / dt)
    x_end_index = int(t_end / dt)   


    # Define Spatial Grid
    y = np.arange(0, Ny * dy, dy)
    z = np.arange(0, Nz * dz, dz)
    Y, Z = np.meshgrid(y, z, indexing='ij')

    # Determine suitable n_con_sample by checking wave resolution
    if not isinstance(n_con_sample, int):
        raise ValueError("n_con_sample must be an integer.")

    while n_con_sample >= 1:
        dtCon = n_con_sample * dx / U
        dxCon = U * dtCon
        fCon = 1 / dtCon
        kdx = k * dxCon

        if kdx < 0.785:
            break
        else:
            print(f"Warning: k*dxCon = {kdx:.3f} too large, decreasing n_con_sample to improve resolution.")
            n_con_sample -= 1

    if n_con_sample < 1:
        raise ValueError("Unable to find suitable n_con_sample to resolve the wave. Try reducing N or increasing resolution.")

    # Print final constraint sampling info
    print(f"Final n_con_sample: {n_con_sample}")
    print(f"Frequency of constraint {np.round(fCon, 3)} [Hz]")
    print(f"dt of constraint {np.round(dtCon, 3)} [s]")
    print(f"dx of constraint {np.round(dxCon, 3)} [m]; k*dx {np.round(kdx, 3)}")

    # Use the highest z position in the grid
    z_idx = -1
    z_top = z[z_idx]

    # Find index of z closest to hub height
    zhub_idx = np.argmin(np.abs(z - zhub))
    zhub_pos = z[zhub_idx]

    # Construct constraint positions:
    # - every 3rd y-position at top of box with full amplitude
    # - every 3rd y-position at hub height with half amplitude
    constraint_positions = [(yy, z_top, Amp) for yy in y[::3]]
    constraint_positions += [(yy, zhub_pos, 0.5 * Amp) for yy in y[::3]]

    # Generate time-dependent constraints
    constraints_list = []
    for ypos, zpos, amp in constraint_positions:
        Uconstraints = amp * np.sin(2 * np.pi * N * t).reshape(-1, 1)
        Vconstraints = mtf.sel(uvw='v', y=ypos, z=zpos).isel(x=slice(x_start_index, x_end_index)).values.reshape(-1, 1)
        Wconstraints = mtf.sel(uvw='w', y=ypos, z=zpos).isel(x=slice(x_start_index, x_end_index)).values.reshape(-1, 1)
        Xconstraints = (t[:, None] * U)
        Yconstraints = np.full_like(Xconstraints, ypos)
        Zconstraints = np.full_like(Xconstraints, zpos)
        constraints_data = np.hstack((Xconstraints, Yconstraints, Zconstraints, Uconstraints, Vconstraints, Wconstraints))
        constraints_list.append(constraints_data)

    constraints_array = np.vstack(constraints_list)

    # Subsample constraint array
    constraints_array = constraints_array[::n_con_sample]

    return constraints_array