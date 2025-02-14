"""
Useful functions for synthesizing turbulence

Leonard Riemer - 12/02/2025
"""

# Functions for turbulence parameters according to IEC-61400-1:2019

def get_length(zhub):          # According to IEC-61400-1:2019 6.3.1 Eq.(5)
    if zhub <= 60:
        Lambda = 0.7 * zhub
    else:
        Lambda = 42
    return 0.8 * Lambda         # According to IEC-61400-1:2019 Annex C.2 Eq.(C.12)

def get_strength(sigma_1):      # According to IEC-61400-1:2019 Annex C.2 Eq.(C.12)
    return 0.55 * sigma_1

def get_anisotropy():           # According to IEC-61400-1:2019 Annex C.2 Eq.(C.12)
    return 3.9

def get_sigma_1(U, TI):         # According to IEC-61400-1:2019 6.3.2.3 Eq.(10)
    return TI * (0.75 * U + 5.6)

def get_sigma_k(component, sigma_1):    #According to IEC-61400-1:2019 Annex C Table C.1, not applicable to Mann Box
    match component:
        case "u":
            return sigma_1
        case "v":
            return sigma_1 * 0.8
        case "w":
            return sigma_1 * 0.5
        case _:
            raise ValueError(f"Invalid component: {component}")

def get_L_k(component, zhub):           #According to IEC-61400-1:2019 Annex C Table C.1, not applicable to Mann Box
    if zhub <= 60:
        Lambda = 0.7 * zhub
    else:
        Lambda = 42

    match component:
        case "u":
            return 8.10 * Lambda
        case "v":
            return 2.70 * Lambda
        case "w":
            return 0.66 * Lambda
        case _:
            raise ValueError(f"Invalid component: {component}")