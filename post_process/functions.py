import numpy as np
import matplotlib.pyplot as plt

"""def compute_psd(time_series, fs):
    # Perform FFT and calculate power spectral density (PSD), taken from 46100 Assignment 3
    N = len(time_series)
    freqs = np.fft.fftfreq(N, 1/fs) #find frequency bins(Number of samples, time step)
    fft_vals = np.fft.fft(time_series) #fft on time_series
    psd = (np.abs(fft_vals)**2) / (N*fs) #find  power spectral density
    return freqs[:N//2], psd[:N//2]  # Return only positive frequencies and corresponding PSD since FFT is symmetric for real time series
"""
"""def turb_constr_kaimal(U_turb, TI_turb, L_turb): # OLD Function!
    # Could add different sigmas and Ls for u, v, w according to IEC-61400-1:2019 Annex C Table C.1
    # Keep simple becasue it will be replaced by some timeseries (soon sinewaves?)

    # --- Define Parameters ---
    U = U_turb          # Mean wind speed (m/s)
    TI = TI_turb        # Turbulence intensity (%)
    L = L_turb          # Turbulence length scale (m)
    sigma_1 = TI * U    # Standard deviation of wind fluctuations

    # --- Define Spatial & Temporal Grid ---
    Nx, Ny, Nz = 500, 20, 20   # Grid resolution (20x20 points, 5000 time steps)
    dx, dy, dz = 0.1, 5, 5     # Spatial (m) and temporal (s) resolution
    x = np.arange(0, Nx * dx, dx)
    y = np.arange(0, Ny * dy, dy)
    z = np.arange(0, Nz * dz, dz)

    # --- Kaimal Spectrum (IEC 61400) ---
    def kaimal_spectrum(f, U, L):
        return (4 * sigma_1**2 * L / U) / ((1 + 6 * f * L / U)**(5/3))

    # Frequency domain
    f = np.linspace(0.01, 5, 100)  # Frequency range (Hz)
    
    # Calculate Spectra
    S_u = kaimal_spectrum(f, U, L)      # Turbulence spectrum u
    S_v = kaimal_spectrum(f, U, 0.5*L)  # Smaller L for v
    S_w = kaimal_spectrum(f, U, 0.3*L)  # Even Smaller L for w

    # Generate random phases for turbulence realization
    np.random.seed(1)
    phi = np.random.uniform(0, 2*np.pi, len(f))

    # --- Generate Time-Series Turbulence ---
    wind_fluctuations = np.zeros((Nx, Ny, Nz, 3))  # Now includes u, v, w components

    # Generate three independent turbulent components
    for i in range(Ny):
        for j in range(Nz):
            amp_u = np.sqrt(2 * S_u * (f[1] - f[0]))  # Amplitude scaling (df is constant see linspace)
            amp_v = np.sqrt(2 * S_v * (f[1] - f[0]))  # Amplitude scaling (df is constant see linspace)
            amp_w = np.sqrt(2 * S_w * (f[1] - f[0]))  # Amplitude scaling (df is constant see linspace)
            
            u_turb = np.sum(amp_u[:, None] * np.cos(2 * np.pi * f[:, None] * x + phi[:, None]), axis=0) # Cosine for only real part
            v_turb = np.sum(amp_v[:, None] * np.cos(2 * np.pi * f[:, None] * x + phi[:, None]), axis=0) # Cosine for only real part
            w_turb = np.sum(amp_w[:, None] * np.cos(2 * np.pi * f[:, None] * x + phi[:, None]), axis=0) # Cosine for only real part

            wind_fluctuations[:, i, j, 0] = u_turb  # u-component
            wind_fluctuations[:, i, j, 1] = v_turb  # v-component
            wind_fluctuations[:, i, j, 2] = w_turb  # w-component



    # --- Add Mean Wind to Turbulence ---
    wind_field = U + wind_fluctuations  # Total wind field

    # --- Y/Z Positions for constrained turbulence ---
    # X Positions is represented as timeseries (see Taylors hypothesis on frozen turbulence)
    ypos = 10
    zpos = 10

    # --- Visualization ---
    plt.figure()
    plt.plot(x, wind_field[:, ypos, zpos, 0], label=f"u at (y:{ypos},z:{zpos})")
    plt.plot(x, wind_field[:, ypos, zpos, 1], label=f"v at (y:{ypos},z:{zpos})")
    plt.plot(x, wind_field[:, ypos, zpos, 2], label=f"w at (y:{ypos},z:{zpos})")
    plt.xlabel("Time (s)")
    plt.ylabel("Wind Speed (m/s)")
    plt.title("Wind Speed Time Series at One Grid Point")
    plt.legend()
    plt.grid()

    f_psd_u, psd_u = compute_psd(wind_field[:, ypos, zpos, 0], np.max(f))
    f_psd_v, psd_v = compute_psd(wind_field[:, ypos, zpos, 1], np.max(f))
    f_psd_w, psd_w = compute_psd(wind_field[:, ypos, zpos, 2], np.max(f))
    
    plt.figure()
    plt.loglog(f, f*S_u, label="Theoretical Kaimal Spectrum")
    plt.loglog(f_psd_u, f_psd_u*psd_u, label=f"Generated uu Spectrum at (y:{ypos},z:{zpos})")
    plt.loglog(f_psd_v, f_psd_v*psd_v, label=f"Generated vv Spectrum at (y:{ypos},z:{zpos})")
    plt.loglog(f_psd_w, f_psd_w*psd_w, label=f"Generated ww Spectrum at (y:{ypos},z:{zpos})")
    plt.title("Comparison of theoretical Kaimal Spectrum and spectra of generated time series")
    plt.grid()
    plt.legend()

    #plt.show()

    # --- Generate Constraint Data ---
    Xconstraints = (x[:, None] * U)  # Time-based x position following x = U * t
    Yconstraints = np.full_like(Xconstraints, ypos)  # Fixed y position
    Zconstraints = np.full_like(Xconstraints, zpos)  # Fixed z position
    Uconstraints = wind_field[:, ypos, zpos, 0].reshape(-1, 1)  # u-component
    Vconstraints = wind_field[:, ypos, zpos, 1].reshape(-1, 1)  # v-component
    Wconstraints = wind_field[:, ypos, zpos, 2].reshape(-1, 1)  # w-component

    # Create the full constraint data structure
    Constraints = np.hstack([Xconstraints, Yconstraints, Zconstraints, Uconstraints, Vconstraints, Wconstraints])

    return Constraints"""

"""def constraint_gen(positions, U_turb, TI_turb, Nxyz_turb, dxyz_turb, L_turb, n_con_sample=int(4)):
    # --- Define Parameters ---
    U = U_turb          # Mean wind speed (m/s)
    TI = TI_turb        # Turbulence intensity (%)
    L = L_turb          # Turbulence length scale (m)
    sigma_1 = TI * U    # Standard deviation of wind fluctuations
    Nx, Ny, Nz = Nxyz_turb  # X coordinate of Hipersim turbulence box
    dx, dy, dz = dxyz_turb  # Spacing of Hipersim turbulence box

    # --- Define Temporal Grid ---
    dt = dx / U
    Nt = Nx
    t_start = 0.1 * Nt * dt
    t = np.arange(t_start, Nt * dt, dt)
    
    # --- Kaimal Spectrum (IEC 61400) ---
    def kaimal_spectrum(f, U, L):
        return (4 * sigma_1**2 * L / U) / ((1 + 6 * f * L / U)**(5/3))

    # Frequency domain
    f = np.linspace(0.01, 5, 100)  # Frequency range (Hz)
    
    # Calculate Spectra
    S_u = kaimal_spectrum(f, U, L)      # Turbulence spectrum u
    S_v = kaimal_spectrum(f, U, 0.5*L)  # Smaller L for v
    S_w = kaimal_spectrum(f, U, 0.3*L)  # Even Smaller L for w

    # Generate random phases for turbulence realization
    np.random.seed(1)
    phi = np.random.uniform(0, 2*np.pi, len(f))

    # --- Generate Time-Series Turbulence ---
    wind_fluctuations = np.zeros((len(t), 3))  # Array of length Nt for three components (u, v, w)

    # --- Generate three independent turbulent components ---
    amp_u = np.sqrt(2 * S_u * (f[1] - f[0]))  # Amplitude scaling (df is constant see linspace)
    amp_v = np.sqrt(2 * S_v * (f[1] - f[0]))  # Amplitude scaling (df is constant see linspace)
    amp_w = np.sqrt(2 * S_w * (f[1] - f[0]))  # Amplitude scaling (df is constant see linspace)
    
    u_turb = np.sum(amp_u[:, None] * np.cos(2 * np.pi * f[:, None] * t + phi[:, None]), axis=0) # Cosine for only real part
    v_turb = np.sum(amp_v[:, None] * np.cos(2 * np.pi * f[:, None] * t + phi[:, None]), axis=0) # Cosine for only real part
    w_turb = np.sum(amp_w[:, None] * np.cos(2 * np.pi * f[:, None] * t + phi[:, None]), axis=0) # Cosine for only real part

    wind_fluctuations[:, 0] = u_turb  # u-component
    wind_fluctuations[:, 1] = v_turb  # v-component
    wind_fluctuations[:, 2] = w_turb  # w-component

    # --- Add Mean Wind to Turbulence ---
    # wind_field = U + wind_fluctuations  # Total wind field
    wind_field = wind_fluctuations + 5  # Only fluctuations

    # --- Y/Z Positions for constrained turbulence ---
    # constraint_positions = [(y1, z1), (y2, z2)]
    constraint_positions = positions

    # --- Loop overe Y/Z Positions for constrained turbulence ---
    constraints_list = []

    for ypos, zpos in constraint_positions:
        if ypos >= Ny or zpos >= Nz:
            raise ValueError(f"Constraint position (y={ypos}, z={zpos}) is out of grid bounds Ny={Ny}, Nz={Nz}")

        # --- Generate Constraint Data ---
        Xconstraints = (t[:, None] * U)  # Time-based x position following x = U * t
        Yconstraints = np.full_like(Xconstraints, ypos)  # Fixed y position
        Zconstraints = np.full_like(Xconstraints, zpos)  # Fixed z position
        Uconstraints = wind_field[:, 0].reshape(-1, 1)  # u-component
        Vconstraints = wind_field[:, 1].reshape(-1, 1)  # v-component
        Wconstraints = wind_field[:, 2].reshape(-1, 1)  # w-component

    # Stack data into single array [x, y, z, u, v, w]
        constraints_data = np.hstack((Xconstraints, Yconstraints, Zconstraints, Uconstraints, Vconstraints, Wconstraints))
        constraints_list.append(constraints_data)

    # Merge all constraints into one array
    constraints_array = np.vstack(constraints_list)

    # Sample from array to make it more coarse/granular
    if not isinstance(n_con_sample, int):
        raise ValueError("n_con_sample needs to be an integer for constraint slicing!")
    
    constraints_array = constraints_array[::n_con_sample]
    dtCon = n_con_sample * dx / U
    fCon = 1/dtCon
    dxCon = U * dtCon

    print(f"Frequency of constraint {np.round(fCon, 3)} [Hz]")
    print(f"dt of constraint {np.round(dtCon, 3)} [s]")
    print(f"dx of constraint {np.round(dxCon, 3)} [m]")

    return constraints_array"""
"""def constraint_gen_flat(vanillaturb, positions, Vave, Uadd, Nxyz_turb, dxyz_turb, n_con_sample=int(4)):
    # --- Define Parameters ---
    U = Vave
    Nx, Ny, Nz = Nxyz_turb  # X coordinate of Hipersim turbulence box
    dx, dy, dz = dxyz_turb  # Spacing of Hipersim turbulence box
    y1, z1 = positions[0]

    # --- Define Temporal Grid ---
    dt = dx / U
    Nt = Nx
    t_start = 0.1 * Nt * dt
    t = np.arange(t_start, Nt * dt, dt)

    # --- Generate Time-Series Turbulence ---
    wind_fluctuations = np.zeros((len(t), 3))  # Array of length Nt for three components (u, v, w)

    wind_fluctuations[:, 0] = Uadd  # u-component
    wind_fluctuations[:, 1] = vanillaturb.sel(uvw='v', y=y1, z=z1)[int(len(vanillaturb.sel(uvw='v', y=y1, z=z1)) * 0.1):]  # v-component
    wind_fluctuations[:, 2] = vanillaturb.sel(uvw='w', y=y1, z=z1)[int(len(vanillaturb.sel(uvw='w', y=y1, z=z1)) * 0.1):]  # w-component

    # --- Add Mean Wind to Turbulence ---
    wind_field = wind_fluctuations        # Only fluctuations

    # --- Y/Z Positions for constrained turbulence ---
    # constraint_positions = [(y1, z1), (y2, z2)]
    constraint_positions = positions

    # --- Loop overe Y/Z Positions for constrained turbulence ---
    constraints_list = []

    for ypos, zpos in constraint_positions:
        if ypos >= Ny or zpos >= Nz:
            raise ValueError(f"Constraint position (y={ypos}, z={zpos}) is out of grid bounds Ny={Ny}, Nz={Nz}")

        # --- Generate Constraint Data ---
        Xconstraints = (t[:, None] * U)  # Time-based x position following x = U * t
        Yconstraints = np.full_like(Xconstraints, ypos)  # Fixed y position
        Zconstraints = np.full_like(Xconstraints, zpos)  # Fixed z position
        Uconstraints = wind_field[:, 0].reshape(-1, 1)  # u-component
        Vconstraints = wind_field[:, 1].reshape(-1, 1)  # v-component
        Wconstraints = wind_field[:, 2].reshape(-1, 1)  # w-component

    # Stack data into single array [x, y, z, u, v, w]
        constraints_data = np.hstack((Xconstraints, Yconstraints, Zconstraints, Uconstraints, Vconstraints, Wconstraints))
        constraints_list.append(constraints_data)

    # Merge all constraints into one array
    constraints_array = np.vstack(constraints_list)

    # Sample from array to make it more coarse/granular
    if not isinstance(n_con_sample, int):
        raise ValueError("n_con_sample needs to be an integer for constraint slicing!")
    
    constraints_array = constraints_array[::n_con_sample]
    dtCon = n_con_sample * dx / U
    fCon = 1/dtCon
    dxCon = U * dtCon

    print(f"Frequency of constraint {np.round(fCon, 3)} [Hz]")
    print(f"dt of constraint {np.round(dtCon, 3)} [s]")
    print(f"dx of constraint {np.round(dxCon, 3)} [m]")

    return constraints_array"""

def Charnock(ustar, Ac=0.0162, g=9.81):
    """
    Computes the roughness length (z0) using the Charnock relation.
    
    Parameters:
    ustar (float): Friction velocity (m/s)
    Ac (float, optional): Charnock constant (default: 0.0162)
    g (float, optional): Acceleration due to gravity (m/s², default: 9.81)
    
    Returns:
    float: Roughness length (m)
    
    Source: ESDU International (1982)
    """
    return Ac * ustar**2 / g

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

def UniCNBL(ustar, z, z0, f, he, N, kappa=0.4, aB=1/290):
    """
    Computes wind speed in a conventionally neutral boundary layer (CNBL) including inversion effects.
    
    Parameters:
    ustar (float): Friction velocity (m/s)
    z (float): Height above ground (m)
    z0 (float): Roughness length (m)
    f (float): Coriolis parameter (s⁻¹)
    he (float): Boundary layer height (m)
    N (float): Brunt–Väisälä frequency (s⁻¹)
    kappa (float, optional): von Kármán constant (default: 0.4)
    aB (float, optional): Stability correction factor (default: 1/290)
    
    Returns:
    float: Wind speed U (m/s)
    
    Source: KELLY et al. (2019)
    """
    Ro = ustar / (f * he)
    correctionTerm = aB * Ro**0.3 * N**2 * z**2 / (kappa * ustar)
    return (ustar / kappa) * np.log(z / z0) + correctionTerm

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
    # Calculation of log-law for Reference wind speed at hub height
    kappa = 0.4                         # von-Karman constant
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

    # Making CNBL profile
    CNBL = UniCNBL(ustar, z, z0, f, he, N)

    # Sample from hub height - radius + radius
    lower, upper = int(zhub - 1 - 90), int(zhub - 1 + 90)
    shearBox = Jet[lower:upper] - U

    # Plot the profile and references
    plt.figure()
    plt.plot(CNBL, z, linestyle="--", linewidth=1, label="Universal CNBL fit", color="g")
    plt.plot(LogLaw, z, linestyle="--", linewidth=1, label="Log-Law", color="black")
    plt.axvline(G, linestyle="--", linewidth=0.7, label="G (based on GDL)", color="grey")
    plt.plot(Jet, z, label="Windspeed Profile", color="r")
    plt.plot(shearBox, z[lower:upper], label="Output to Mann Box", color="b")
    plt.xlabel("Wind Speed U [m/s]")
    plt.ylabel("Height z [m]")
    plt.grid()
    plt.legend()

    print("Height:", he)
    print("Strength:", UJet)
    print("Width:", Width)

    """To add to the uvw components of the mtf object use:
    - Vertical profile given in array of (Nz,)
    - Reshape to (1, 1, Nz) so it can be broadcasted over (Nx, Ny, Nz)
    - Thus: mtf.uvw[0, :, :, :] += vertical_profile[None, None, :]
    """

    return shearBox

def ConstraintWave(positions, Vave, Nxyz_turb, dxyz_turb, theta0, dthetadz, n_con_sample=int(4)):
    # --- Define Parameters ---
    U = Vave          # Mean wind speed (m/s)
    Nx, Ny, Nz = Nxyz_turb  # X coordinate of Hipersim turbulence box
    dx, dy, dz = dxyz_turb  # Spacing of Hipersim turbulence box

    # --- Brunt-Väisäla Frequency ---
    g = 9.81
    Nc = np.sqrt((g/theta0) * dthetadz)
    k = 2*np.pi/(U/Nc)
    print(f"Wave frequency fw {Nc} [Hz]; Wave number k {k} [m^-1]")

    # --- Define Temporal Grid ---
    dt = dx / U
    Nt = Nx
    t_start = 0.1 * Nt * dt
    t = np.arange(t_start, Nt * dt, dt)

    # --- Generate Time-Series Turbulence ---
    wind_fluctuations = np.zeros((len(t), 3))  # Array of length Nt for three components (u, v, w)

    # --- Generate three independent turbulent components ---
    amp_u = 2
    amp_v = 1
    amp_w = 1
    
    u_turb = amp_u * np.sin(2 * np.pi * Nc * t)
    v_turb = amp_v * np.sin(2 * np.pi * Nc * t)
    w_turb = amp_w * np.sin(2 * np.pi * Nc * t)

    wind_fluctuations[:, 0] = u_turb  # u-component
    wind_fluctuations[:, 1] = v_turb  # v-component
    wind_fluctuations[:, 2] = w_turb  # w-component

    wind_field = wind_fluctuations  # Only fluctuations

    # --- Y/Z Positions for constrained turbulence ---
    # constraint_positions = [(y1, z1), (y2, z2)]
    constraint_positions = positions

    # --- Loop overe Y/Z Positions for constrained turbulence ---
    constraints_list = []

    for (ypos, zpos) in constraint_positions:
        if ypos >= Ny or zpos >= Nz:
            raise ValueError(f"Constraint position (y={ypos}, z={zpos}) is out of grid bounds Ny={Ny}, Nz={Nz}")

        # --- Generate Constraint Data ---
        Xconstraints = (t[:, None] * U)  # Time-based x position following x = U * t
        Yconstraints = np.full_like(Xconstraints, ypos)  # Fixed y position
        Zconstraints = np.full_like(Xconstraints, zpos)  # Fixed z position
        Uconstraints = wind_field[:, 0].reshape(-1, 1)  # u-component
        Vconstraints = wind_field[:, 1].reshape(-1, 1)  # v-component
        Wconstraints = wind_field[:, 2].reshape(-1, 1)  # w-component

    # Stack data into single array [x, y, z, u, v, w]
        constraints_data = np.hstack((Xconstraints, Yconstraints, Zconstraints, Uconstraints, Vconstraints, Wconstraints))
        constraints_list.append(constraints_data)

    # Merge all constraints into one array
    constraints_array = np.vstack(constraints_list)

    # Sample from array to make it more coarse/granular
    if not isinstance(n_con_sample, int):
        raise ValueError("n_con_sample needs to be an integer for constraint slicing!")
    
    constraints_array = constraints_array[::n_con_sample]
    dtCon = n_con_sample * dx / U
    fCon = 1/dtCon
    dxCon = U * dtCon

    if k*dxCon >= 0.4:
        raise ValueError("k*dxCon should be << 1 to ensure a wave signal is apparent!")

    print(f"Frequency of constraint {np.round(fCon, 3)} [Hz]")
    print(f"dt of constraint {np.round(dtCon, 3)} [s]")
    print(f"dx of constraint {np.round(dxCon, 3)} [m]; k*dx {np.round(k*dxCon, 3)}")
    

    return constraints_array

def ConstraintWaveCirc(center, radius, Vave, Nxyz_turb, dxyz_turb, theta0, dthetadz, decay_factor=0.5, n_con_sample=int(4)):
    # --- Define Parameters ---
    U = Vave                # Mean wind speed (m/s)
    Nx, Ny, Nz = Nxyz_turb  # X coordinate of Hipersim turbulence box
    dx, dy, dz = dxyz_turb  # Spacing of Hipersim turbulence box

    # --- Brunt-Väisäla Frequency ---
    g = 9.81
    Nc = np.sqrt((g/theta0) * dthetadz)*10
    k = 2*np.pi/(U/Nc)
    print(f"Wave frequency fw {np.round(Nc, 3)} [Hz]; Wave number k {np.round(k, 3)} [m^-1]")

    # --- Define Temporal Grid ---
    dt = dx / U
    Nt = Nx
    t_start = 0.1 * Nt * dt
    t = np.arange(t_start, Nt * dt, dt)

    # --- Define Spatial Grid ---
    y = np.arange(0, Ny * dy, dy)
    z = np.arange(0, Nz * dz, dz)
    Y, Z = np.meshgrid(y, z)

    # --- Generate Time-Series Turbulence ---
    wind_fluctuations = np.zeros((len(t), 3))  # Array of length Nt for three components (u, v, w)

    # --- Generate three independent turbulent components ---
    amp_u = 2
    amp_v = 1
    amp_w = 1
    
    u_turb = amp_u * np.sin(2 * np.pi * Nc * t)
    v_turb = amp_v * np.sin(2 * np.pi * Nc * t)
    w_turb = amp_w * np.sin(2 * np.pi * Nc * t)

    wind_fluctuations[:, 0] = u_turb  # u-component
    wind_fluctuations[:, 1] = v_turb  # v-component
    wind_fluctuations[:, 2] = w_turb  # w-component

    # --- Find Y/Z Positions for constrained turbulence in the circle ---
    yc, zc = center         # Center of the circular constraint area
    rc = radius             # Radius of the circurlar constraint area
    circle_indices = (Y - yc) ** 2 + (Z - zc) ** 2 <= rc ** 2  # constraint_positions = [(y1, z1), (y2, z2)]
    # Extract (y, z) coordinates
    constraint_positions = list(zip(Y[circle_indices], Z[circle_indices]))
    # Extract (y, z) coordinates
    coordinates_inside = list(zip(Y[circle_indices], Z[circle_indices]))
    coordinates_outside = list(zip(Y[~circle_indices], Z[~circle_indices]))

    # Plot the results
    plt.figure()
    plt.scatter(*zip(*coordinates_outside), color='lightgray', s=10, label='Outside Circle')
    plt.scatter(*zip(*coordinates_inside), color='blue', s=10, label='Inside Circle')
    plt.scatter(yc, zc, color='red', marker='x', label='Center')
    plt.gca().set_aspect('equal')  # Keep the aspect ratio 1:1
    plt.xlabel('y')
    plt.ylabel('z')
    plt.legend()
    plt.title('Points Inside and Outside the Circle')

    # --- Loop overe Y/Z Positions for constrained turbulence ---
    constraints_list = []

    for (ypos, zpos) in constraint_positions:
        if ypos >= Ny or zpos >= Nz:
            raise ValueError(f"Constraint position (y={ypos}, z={zpos}) is out of grid bounds Ny={Ny}, Nz={Nz}")

        # Compute radial distance and decay factor
        r = np.sqrt((ypos - yc)**2 + (zpos - zc)**2)    # Possibly have to add int() so that it fits on meshgrid
        decay = np.exp(-decay_factor * (r / radius)) if r <= radius else 0 # Sets outside of circle to zero, but shouldnt occur
        
        # Apply decay
        Uconstraints = (wind_fluctuations[:, 0] * decay).reshape(-1, 1)
        Vconstraints = (wind_fluctuations[:, 1] * decay).reshape(-1, 1)
        Wconstraints = (wind_fluctuations[:, 2] * decay).reshape(-1, 1)

        # Define constraint positions in time and space
        Xconstraints = (t[:, None] * U)
        Yconstraints = np.full_like(Xconstraints, ypos)
        Zconstraints = np.full_like(Xconstraints, zpos)
        
        # Stack into final constraint format
        constraints_data = np.hstack((Xconstraints, Yconstraints, Zconstraints, Uconstraints, Vconstraints, Wconstraints))
        constraints_list.append(constraints_data)

    # Merge all constraints into one array
    constraints_array = np.vstack(constraints_list)

    # Sample from array to make it more coarse/granular
    if not isinstance(n_con_sample, int):
        raise ValueError("n_con_sample needs to be an integer for constraint slicing!")
    
    constraints_array = constraints_array[::n_con_sample]
    dtCon = n_con_sample * dx / U
    fCon = 1/dtCon
    dxCon = U * dtCon

    print(f"Frequency of constraint {np.round(fCon, 3)} [Hz]")
    print(f"dt of constraint {np.round(dtCon, 3)} [s]")
    print(f"dx of constraint {np.round(dxCon, 3)} [m]; k*dx {np.round(k*dxCon, 3)}")

    if k*dxCon >= 0.4:
        raise ValueError("k*dxCon should be << 1 to ensure a wave signal is apparent!")
    
    return constraints_array

