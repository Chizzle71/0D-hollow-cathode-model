import numpy as np
from scipy.optimize import fsolve, brentq
import constants as c



def solve_T_eV(n_n, AoV, m, a, E_i, q, T, D, prop, bohm_factor=0.61, bracket=(0.1, 10)):
    '''T_eV is in eV, theyre in like per electron'''
    def excess(T_eV):
        ions_produced = n_n * reaction_rate(T_eV, a, E_i/c.e, q) #pass E_i in in eV not J anymore :)
        #ions_lost_bohm = bohm_loss(T_eV, m, AoV, bohm_factor)
        ions_lost_ambipolar = ambipolar_loss(T_eV, n_n, T, D, m, prop)
        #ions_lost = min(ions_lost_bohm, ions_lost_ambipolar)#min as both functions are limits so whichever is smaller is active and is correct here
        ions_lost = ions_lost_ambipolar
        return ions_produced - ions_lost
    T_eV = brentq(excess, *bracket)
    ions_produced = n_n * reaction_rate(T_eV, a, E_i/c.e, q)
    return T_eV, ions_produced



def ambipolar_loss(T_eV, n_n, T, D, m, prop): #yes this is general but ive only conceptually checked this for insert region so hold off
    T_e = T_eV*c.e/c.k
    
    nu_i_n = coll_freq_i_n(n_n, T, prop) #collisions
    nu_e_n = coll_freq_e_n(n_n, T_e, prop)

    mu_i = mobility(c.e, m, nu_i_n) #mobility coefficients
    mu_e = mobility(c.e, c.m_e, nu_e_n)

    D_i = diffusion_coefficient(T, m, nu_i_n) #diffusion coefficients
    D_e = diffusion_coefficient(T_e, c.m_e, nu_e_n)

    D_a = ambipolar_diffusion_coefficient(mu_e, mu_i, D_e, D_i) #ambipolar diffusion coefficient - electron moves makes E then ion follows and balance yay

    ambipolar_loss = ambipolar_loss_walls(D/2, D_a)
    return ambipolar_loss



def ambipolar_diffusion_coefficient(mu_e, mu_i, D_e, D_i):
    Diff_a = (mu_i*D_e + mu_e*D_i)/(mu_e + mu_i)
    return Diff_a



def ambipolar_loss_walls(r, D_a):
    '''lambda is diffusion length using the first 0 of the 0th order bessel function for it to be 0 at insert wall and max at centre'''
    '''this wants radius and D_a generic for the region in question'''
    Lambda = r/c.j_0_0
    return D_a/Lambda**2



def bohm_loss(T_eV, m, area_volume, bohm_factor):
    return bohm_factor*np.sqrt(c.e*T_eV/m)*area_volume



def mobility(q, m, nu):
    mu = np.abs(q)/(m*nu)
    return mu



def coll_freq_i_n(n_n, T_gas, prop):
    '''T_i roughly T_gas'''
    if prop == "Xe":
        sigma = c.Xe_coll_i_n
        m = c.Xe_m_kg
    elif prop == "Bi":
        sigma = c.Bi_coll_i_n
        m = c.Bi_m_kg
    nu_i_n = n_n*sigma*np.sqrt(8*c.k*T_gas/(np.pi*m))
    return nu_i_n



def coll_freq_e_n(n_n, T_e, prop):
    if prop == "Xe":
        sigma = c.Xe_coll_e_n
    elif prop == "Bi":
        sigma = c.Bi_coll_e_n
    nu_e_n = n_n*sigma*np.sqrt(8*c.k*T_e/(np.pi*c.m_e))
    return nu_e_n



def diffusion_coefficient(T, m, nu):
    '''temperature in kelvin, m in kg, nu in s-1 cause it is collision frequency'''
    Diff = (c.k*T)/(m*nu)
    return Diff



def flux_a(mu_i, mu_e, D_i, D_e, n_grad):
    '''flux assuming ambipolar diffusion, reduced to 1D version from chens 3D version'''
    flux_a = -(mu_i*D_e + mu_e*D_i)/(mu_i + mu_e)*n_grad
    return flux_a



def reaction_rate(T_eV, a, P, q):
    """Maxwellian-averaged <sigma_i v> in m^3 s-1 for a single T_eV in eV, P is first ionisation energy in eV"""
    E_max = 50*T_eV + 10*P
    v_th = np.sqrt(2*P*c.e/c.m_e) #threshold speed
    v = np.linspace(v_th, np.sqrt(2*E_max*c.e/c.m_e), 1000)
    Ek = 0.5*c.m_e*v**2/c.e
    integrand = lotz(Ek, a, P, q) * v * speed_distribution(v, T_eV)
    return np.trapezoid(integrand, v)



def lotz(T_eV, a, P, q):
    sigma = a*1e-4*q*np.log(T_eV/P)/(T_eV*P)
    return np.where(T_eV > P, sigma, 0)



def speed_distribution(v, T_eV):
    return 4*np.pi * (c.m_e/(2*np.pi*c.e*T_eV))**1.5 * v**2 * np.exp(-c.m_e*v**2/(2*c.e*T_eV))



def poiseuille(mu, L, Q_downstream, p_downstream, r):
    '''standard units, returns pressure upstream in Pa, compressible ideal gas variant'''
    return ((16*mu*L*Q_downstream*p_downstream)/(np.pi*r**4) + p_downstream**2)**0.5



def ideal_gas(p, T, prop):
    '''standard units, returns density in kg m-3, yes molar mass in kg mol-1'''
    if prop == "Xe":
        mlm = c.Xe_m
    elif prop == "Bi":
        mlm = c.Bi_m
    return (p*mlm)/(c.R_u*T)



def viscosity(A, M, T):
    '''returns in Pa s, works for Bi, and metals apparently'''
    return T*A*M**0.5



def paschen(prop, T, d, p):
    if prop == "Xe":
        sigma = 8e-20 #NOTE take deeper with cross section file
        E = c.Xe_E_i
        gamma_see = 0.0076967005076142144 #NOTE taken from argon then scaled for Xe
        #gamma_see = 1
    elif prop == "Bi":
        sigma = 1.12e-19 #NOTE take deeper with cross section file
        E = c.Bi_E_i
        gamma_see = 0.004625 #NOTE taken from argon then scaled for Bi
        #gamma_see = 1
    A = (sigma)/(c.k*T) * c.torr*c.cm #in V cm-1 torr-1
    B = (sigma*E)/(c.e*c.k*T) * c.torr*c.cm #in V cm-1 torr-1
    p = p/c.torr #in torr
    d = d/c.cm #in cm
    denom = np.log(A*p*d) - np.log(np.log(1 + 1/gamma_see))
    V = (B*p*d)/denom
    return np.where((denom > 0) & (V < 1e3), V, np.nan)



def emission(emitter, T, E):
    '''only got lab6 and bao values for now'''
    if emitter == "lab6":
        A = c.A_lab6
    elif emitter == "bao":
        A = c.A_bao
    phi_wf = work_function(emitter, T)
    phi_eff = phi_wf - schottky(E)
    return A*T**2*np.e**(-(phi_eff)/(c.k*T))



def work_function(emitter, T):
    '''return work function in J'''
    if emitter == "lab6":
        phi_0 = c.phi_0_lab6
        alpha = c.alpha_lab6
    elif emitter == "bao":
        phi_0 = c.phi_0_bao
        alpha = c.alpha_bao
    return phi_0 + alpha*T



def schottky(E):
    '''E is at the emitter surface, returns in J)''' #NOTE check units here (J)
    return c.e*((c.e*E)/(4*np.pi*c.eps0))**0.5



def find_T_required(emitter, I_d_target, area, E, T_guess, dT=0.5, tol=0.01, T_max=3000, max_iter=10000):
    T = T_guess
    for _ in range(max_iter):
        I = area * emission(emitter, T, E)
        if abs(I - I_d_target) <= tol * I_d_target:
            return T, I
        if T > T_max:
            raise RuntimeError("T exceeded T_max without converging")
        T += dT if I < I_d_target else -dT
    raise RuntimeError("did not converge")



def thermal_speed(T_s, M):
    '''Temperature of species is in Kelvin, M is mass of the species in kg'''
    return np.sqrt((c.k*T_s)/(M))



def sheath_field(n_e, T_eV, phi_s, m):
    '''field at the emitter surface using child langmuir ion sheath and in V m-1'''
    J_i = 0.61*c.e*n_e*np.sqrt(c.e*T_eV/m)
    s = np.sqrt(4*c.eps0/(9*J_i)*np.sqrt(2*c.e/m))*phi_s**0.75
    return 4*phi_s/(3*s)



def insert_currents(n_e, phi_s, T_eV, T_ins, loss_rate, V, A_wall, A_emit, emitter, m):
    '''returns emitted, ion and back flux electron currents in A'''
    I_i = c.e*n_e*V*loss_rate #all ion loss lands on the insert walls, tweak can be made
    v_e = np.sqrt(8*c.e*T_eV/(np.pi*c.m_e))
    I_b = 0.25*c.e*n_e*v_e*np.exp(-phi_s/T_eV)*A_wall
    E_s = sheath_field(n_e, T_eV, phi_s, m)
    I_e = emission(emitter, T_ins, E_s)*A_emit
    return I_e, I_i, I_b



def insert_balance(I_total, T_eV, T_ins, loss_rate, V, A_wall, A_emit, emitter, m, E_i, cost_factor=1, n_n=None, L=None, D=None, prop=None):
    '''solves n_e in m-3 and phi_s in V, E_i in J then usees insert currents again to get value'''
    def balance(x):
        n_e = 10**x[0]
        phi_s = abs(x[1])
        I_e, I_i, I_b = insert_currents(n_e, phi_s, T_eV, T_ins, loss_rate, V, A_wall, A_emit, emitter, m)
        current = I_e + I_i - I_b - I_total
        energy = I_e*phi_s - I_i*(cost_factor*E_i/c.e + phi_s + T_eV/2) - 2*T_eV*I_b #P=IV and that thing from chen, cost factor wants better estimate
        energy += ohmic_heating(I_total, n_e, n_n, T_eV, L, D, prop) - 2.5*T_eV*I_total
        return [current/I_total, energy/(10*I_total)]
    sol, info, ier, msg = fsolve(balance, [20.5, 10], full_output=True) #pass in n_e and phi_s, minimise current difference and power difference
    if ier != 1 or np.max(np.abs(info["fvec"])) > 1e-6:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    n_e, phi_s = 10**sol[0], abs(sol[1])
    I_e, I_i, I_b = insert_currents(n_e, phi_s, T_eV, T_ins, loss_rate, V, A_wall, A_emit, emitter, m)
    return n_e, phi_s, I_e, I_i, I_b



def insert_heat(T_insert, T_eV, phi_s, I_e, I_i, I_b, emitter, E_i):
    '''heat into the insert from the plasma and heat taken by emission cooling in W'''
    '''p_emit_cool - p_plasma is the richardson cooling minus the plasma heating to the insert so the difference between
    these is the extra temperature required, this extra temperature required in watts is for the thermal sim
    and is not just power from the heater but there is radiation and conduction losses as well, convection nah tho'''
    phi_wf = work_function(emitter, T_insert)/c.e #in J
    P_plasma = I_i*(E_i/c.e + phi_s + T_eV/2 - phi_wf) + I_b*(2*T_eV + phi_wf)
    P_emit_cool = I_e*(phi_wf + 2*c.k*T_insert/c.e)
    return P_plasma, P_emit_cool



def coulomb_log(n_e, T_eV):
    '''Naval paper source, n_e in m-3''' #i believe this is applicable here
    return max(2.0, 23 - 0.5*np.log(n_e*1e-6) + 1.5*np.log(T_eV))



def coll_freq_e_i(n_e, T_eV):
    '''electron ion collision frequency in s-1'''
    return 2.91e-12*n_e*coulomb_log(n_e, T_eV)*T_eV**-1.5



def plasma_resistance(n_e, n_n, T_eV, L, D, prop):
    '''resistance (ohms) for a plasma cylinder of length L and diameter D, standard units'''
    T_e = T_eV*c.e/c.k
    nu_m = coll_freq_e_i(n_e, T_eV) + coll_freq_e_n(n_n, T_e, prop)
    eta = c.m_e*nu_m/(n_e*c.e**2)
    return eta*L/(np.pi*0.25*D**2)



def ohmic_heating(I_d, n_e, n_n, T_eV, L, D, prop):
    '''i think the same concept from gk'''
    return plasma_resistance(n_e, n_n, T_eV, L, D, prop)*I_d**2



def orifice_n_e(I_d, T_eV_orifice, T_eV_insert, n_n, ion_rate, L, D, E_i, prop, cost_factor=1):
    '''balances power from ohmic heating in and convection out and ion reactions out'''
    '''average orifice plasma density in m-3 and ion_rate is ionisation rate per electron in s-1, from solve_T_eV, E_i in J and cost_factor=1 matches gk'''
    V_1 = np.pi*0.25*D**2*L
    P_convection = 2.5*I_d*(T_eV_orifice - T_eV_insert)
    def resid(log_n):
        n_e = 10**log_n
        P_ohm = ohmic_heating(I_d, n_e, n_n, T_eV_orifice, L, D, prop)
        P_ion = n_e*V_1*ion_rate*cost_factor*E_i
        return P_ohm - P_convection - P_ion
    return 10**brentq(resid, 16, 25)



def m(n_e, T_eV, phi_s, A_wall, m, I_i_cap=np.inf, h=0.61):
    I_i = min(h*c.e*n_e*np.sqrt(c.e*T_eV/m)*A_wall, I_i_cap)
    v_e = np.sqrt(8*c.e*T_eV/(np.pi*c.m_e))
    I_b = h*0.25*c.e*n_e*v_e*np.exp(-phi_s/T_eV)*A_wall
    return I_i, I_b



def floating_potential(T_eV, m):
    '''wall potential in V to balance Bohm ion and random electron flux'''
    return 0.5*T_eV*np.log(m/(2*np.pi*c.m_e))



def orifice_wall_currents(n_e, T_eV, phi_s, A_wall, m, I_i_cap=np.inf, h=0.61):
    '''ion (Bohm, capped) and back-flux electron currents to the orifice wall in A'''
    I_i = min(h*c.e*n_e*np.sqrt(c.e*T_eV/m)*A_wall, I_i_cap)
    v_e = np.sqrt(8*c.e*T_eV/(np.pi*c.m_e))
    I_b = h*0.25*c.e*n_e*v_e*np.exp(-phi_s/T_eV)*A_wall
    return I_i, I_b
