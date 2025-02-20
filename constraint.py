import numpy as np
import matplotlib.pyplot as plt

def compute_psd(time_series, fs):
    # Perform FFT and calculate power spectral density (PSD), taken from 46100 Assignment 3
    N = len(time_series)
    freqs = np.fft.fftfreq(N, 1/fs) #find frequency bins(Number of samples, time step)
    fft_vals = np.fft.fft(time_series) #fft on time_series
    psd = (np.abs(fft_vals)**2) / (N*fs) #find  power spectral density
    return freqs[:N//2], psd[:N//2]  # Return only positive frequencies and corresponding PSD since FFT is symmetric for real time series

def turb_constr_kaimal(U_turb, TI_turb, L_turb): 
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

    return Constraints

def constraint_gen(positions, U_turb, TI_turb, Nxyz_turb, dxyz_turb, L_turb):
    # --- Define Parameters ---
    U = U_turb          # Mean wind speed (m/s)
    TI = TI_turb        # Turbulence intensity (%)
    L = L_turb          # Turbulence length scale (m)
    sigma_1 = TI * U    # Standard deviation of wind fluctuations
    Nx, Ny, Nz = Nxyz_turb  # X coordinate of Hipersim turbulence box
    dx, dy, dz = dxyz_turb  # Spacing of Hipersim turbulence box

    # --- Define Temporal Grid ---
    dt = dx / U
    Nt = int((Nx * dx) / (U * dt))
    t = np.arange(0, Nt * dt, dt)
    
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
    wind_fluctuations = np.zeros((Nt, 3))  # Array of length Nt for three components (u, v, w)

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
    wind_field = wind_fluctuations  # Only fluctuations

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


    return constraints_array

def constraint_wave(positions, U_turb, TI_turb, Nxyz_turb, dxyz_turb, L_turb):
    # --- Define Parameters ---
    U = U_turb          # Mean wind speed (m/s)
    TI = TI_turb        # Turbulence intensity (%)
    L = L_turb          # Turbulence length scale (m)
    sigma_1 = TI * U    # Standard deviation of wind fluctuations
    Nx, Ny, Nz = Nxyz_turb  # X coordinate of Hipersim turbulence box
    dx, dy, dz = dxyz_turb  # Spacing of Hipersim turbulence box

    # --- Define Temporal Grid ---
    dt = dx / U
    Nt = int((Nx * dx) / (U * dt)) // 2
    t = np.arange(0, Nt * dt, dt)
    f = 2 / (Nt * dt)

    # --- Generate Time-Series Turbulence ---
    wind_fluctuations = np.zeros((Nt, 3))  # Array of length Nt for three components (u, v, w)

    # --- Generate three independent turbulent components ---
    amp_u = 2   # Amplitude scaling (df is constant see linspace)
    amp_v = 1   # Amplitude scaling (df is constant see linspace)
    amp_w = 1   # Amplitude scaling (df is constant see linspace)
    
    u_turb = amp_u * np.sin(2 * np.pi * f * t)
    v_turb = amp_v * np.sin(2 * np.pi * f * t)
    w_turb = amp_w * np.sin(2 * np.pi * f * t)

    wind_fluctuations[:, 0] = u_turb  # u-component
    wind_fluctuations[:, 1] = v_turb  # v-component
    wind_fluctuations[:, 2] = w_turb  # w-component

    # --- Add Mean Wind to Turbulence ---
    # wind_field = U + wind_fluctuations  # Total wind field
    wind_field = wind_fluctuations  # Only fluctuations

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


    return constraints_array