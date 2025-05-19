"""Library of common .py functions for thesis -- Leonard Riemer -- 19/05/2025"""
import numpy as np

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

def JetShear(U, zhub, TurbNz, Turbdz, JetHeight, JetWidth, JetStrength):
    """
    Computes the vertical wind shear profile influenced by a supergeostrophic jet,
    using a log-law base and a Gaussian jet centered at specified height.

    Parameters:
    U (float): Reference wind speed at hub height (m/s)
    zhub (float): Hub height (m)
    TurbNz (int): Number of vertical points in the turbulence box
    Turbdz (float): Vertical spacing in the turbulence box (m)
    JetHeight (float): Height of the jet center (m)
    JetWidth (float): Characteristic width of the jet (m)
    JetStrength (float): Maximum wind speed contribution from the jet (m/s)

    Returns:
    np.ndarray: Wind shear box (array of wind speeds) over the turbine rotor span
    """
    # Calculation of log-law parameters for Reference wind speed at hub height
    z0 = 0.0001                         # Roughness length over water, also possible through Charnock relation.
    ustar = LogLawustar(U, zhub, z0)    # Friction velocity defined at the surface (For now just fitted to U=10 m/s at hub height, Else possibly average from Sprog 0.2580)

    # Define vertical space
    Nz = 1000                           # 1000 grid points
    dz = Turbdz                         # Spacing of dz 1 [m]
    z = np.arange(dz, Nz*dz, dz)        # vertical space array

    # Make vertical wind speed profile and add gaussian jet
    LogLaw = LogLawU(ustar, z, z0)
    Jet = LogLaw + JetStrength * np.exp(-((z - JetHeight) / JetWidth)**2)

    # Sample from hub height ± half of the turbulence box
    TurbRadius = TurbNz // 2
    hub_index = int(zhub / dz)
    lower, upper = max(0, int(hub_index - TurbRadius)), min(len(Jet), int(hub_index + TurbRadius))
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