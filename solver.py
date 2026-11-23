import numpy as np
import math
from dataclasses import dataclass
from scipy.optimize import fsolve, brentq
import constants as c
import physics as p



#initial values
@dataclass
class initialise:
    #elements
    prop = "Xe"
    emitter = "lab6" #nstar is ba-o in w, lab6 works
    tube = "Mo"
    keeper = "Mo"
    heater = "Ta"

    #initial values - currently nstar ish
    L_insert = 25.4*c.mm
    Di_insert = 3.8*c.mm
    Do_insert = 6.35*c.mm #these dont match - was 5.5, doesnt effect model

    L_orifice = 0.74*c.mm
    D_orifice = 1.02*c.mm #1.02 was nstar

    Di_tube = 6.35*c.mm
    Do_tube = 8.5*c.mm #guess but not used
    L_tube = 200*c.mm - L_insert - L_orifice #not actual length, this is just interior length so minus insert and orifice overlap

    L_gap = 4*c.mm #testing, 4 is standard

    L_keeper = 1*c.mm #guess
    D_keeper = 3*c.mm #guess
    Di_keeper = 10*c.mm #guess

    geometry_name = "1" #NOTE

    #operating conditions
    I_d_target = 7.33 #A target
    mdot = 0.24*c.mg #kg s-1 this is currently a guess but i want to vary this till it gives me the best operating envelope
    
    #measured quantities
    p_plume = 10 #Pa
    if emitter == "lab6":
        T_universal_guess = 1700 #K
    elif emitter == "bao":
        T_universal_guess = 1600 #K
    T_tube = T_universal_guess
    T_insert = T_universal_guess
    T_orifice = T_universal_guess
    T_gap = T_universal_guess
    T_keeper = T_universal_guess
    T_plume = T_universal_guess



def solve(L_tube, Di_tube, Do_tube,
          L_insert, Di_insert, Do_insert,
          L_orifice, D_orifice,
          L_gap,
          L_keeper, D_keeper, Di_keeper,
          emitter,
          prop, mdot,
          p_plume, I_d_target,
          T_universal_guess, T_tube, T_insert, T_orifice, T_gap, T_keeper, T_plume):


    #setup NOTE make it more universal
    if prop == "Xe":
        m = c.Xe_m_kg
        mu = c.Xe_mu
        E_i = c.Xe_E_i
        a = c.Xe_a
        q = c.Xe_q
    elif prop == "Bi":
        m = c.Bi_m_kg #weighted average for assumed partial pressures at 1700K 
        mu = c.Bi_mu
        E_i = c.Bi_E_i
        a = c.Bi_a
        q = c.Bi_q


    #pressure calcs
    rho_keeper_downstream = p.ideal_gas(p_plume, T_keeper, prop)
    Q_keeper_downstream = mdot/rho_keeper_downstream
    p_keeper_upstream = p.poiseuille(mu, L_keeper, Q_keeper_downstream, p_plume, D_keeper/2)
    p_keeper = 0.5*(p_keeper_upstream + p_plume)

    rho_gap_downstream = p.ideal_gas(p_keeper_upstream, T_gap, prop)
    Q_gap_downstream = mdot/rho_gap_downstream
    p_gap_upstream = p.poiseuille(mu, L_gap, Q_gap_downstream, p_keeper_upstream, Di_keeper/2)
    p_gap = 0.5*(p_keeper_upstream + p_gap_upstream)
    #print("gap pressure (torr)", p_gap/c.torr)

    rho_orifice_downstream = p.ideal_gas(p_gap_upstream, T_orifice, prop)
    Q_orifice_downstream = mdot/rho_orifice_downstream
    p_orifice_upstream = p.poiseuille(mu, L_orifice, Q_orifice_downstream, p_gap_upstream, D_orifice/2)
    p_orifice = 0.5*(p_gap_upstream + p_orifice_upstream)

    rho_insert_downstream = p.ideal_gas(p_orifice_upstream, T_insert, prop)
    Q_insert_downstream = mdot/rho_insert_downstream
    p_insert_upstream = p.poiseuille(mu, L_insert, Q_insert_downstream, p_orifice_upstream, Di_insert/2)
    p_insert = 0.5*(p_orifice_upstream + p_insert_upstream)

    rho_tube_downstream = p.ideal_gas(p_insert_upstream, T_tube, prop)
    Q_tube_downstream = mdot/rho_tube_downstream
    p_tube_upstream = p.poiseuille(mu, L_tube, Q_tube_downstream, p_insert_upstream, Di_tube/2)
    p_tube = 0.5*(p_insert_upstream + p_tube_upstream)


    #breakdown stuff - loop so that plot gives x axis of mass flow rate, y axis of breakdown voltage, and y axis 2 for pressure
    mdot_range_ignition = np.geomspace(mdot, 5000*mdot, 1000)
    p_range_plume_ignition = mdot_range_ignition/mdot * p_plume #NOTE assumes pressure linear with mfr in plume

    rho_range_keeper_downstream_ignition = p.ideal_gas(p_range_plume_ignition, T_keeper, prop)
    Q_range_keeper_downstream_ignition = mdot_range_ignition/rho_range_keeper_downstream_ignition
    p_range_keeper_upstream_ignition = p.poiseuille(mu, L_keeper, Q_range_keeper_downstream_ignition, p_range_plume_ignition, D_keeper/2)

    rho_range_gap_downstream_ignition = p.ideal_gas(p_range_keeper_upstream_ignition, T_gap, prop)
    Q_range_gap_downstream_ignition = mdot_range_ignition/rho_range_gap_downstream_ignition
    p_range_gap_upstream_ignition = p.poiseuille(mu, L_gap, Q_range_gap_downstream_ignition, p_range_keeper_upstream_ignition, Di_keeper/2)
    p_range_gap_ignition = 0.5*(p_range_keeper_upstream_ignition + p_range_gap_upstream_ignition)

    rho_range_orifice_downstream_ignition = p.ideal_gas(p_range_gap_upstream_ignition, T_orifice, prop)
    Q_range_orifice_downstream_ignition = mdot_range_ignition/rho_range_orifice_downstream_ignition
    p_range_orifice_upstream_ignition = p.poiseuille(mu, L_orifice, Q_range_orifice_downstream_ignition, p_range_gap_upstream_ignition, D_orifice/2)

    rho_range_insert_downstream_ignition = p.ideal_gas(p_range_orifice_upstream_ignition, T_insert, prop)
    Q_range_insert_downstream_ignition = mdot_range_ignition/rho_range_insert_downstream_ignition
    p_range_insert_upstream_ignition = p.poiseuille(mu, L_insert, Q_range_insert_downstream_ignition, p_range_orifice_upstream_ignition, Di_insert/2)

    rho_range_tube_downstream_ignition = p.ideal_gas(p_range_insert_upstream_ignition, T_tube, prop)
    Q_range_tube_downstream_ignition = mdot_range_ignition/rho_range_tube_downstream_ignition
    p_range_tube_upstream_ignition = p.poiseuille(mu, L_tube, Q_range_tube_downstream_ignition, p_range_insert_upstream_ignition, Di_tube/2)

    voltage_range_ignition = p.paschen(prop, T_gap, L_gap, p_range_gap_ignition)


    #n_n is number density of neutrals
    n_n_tube = (p_tube)/(T_tube*c.k)
    
    n_n_insert = (p_insert)/(T_insert*c.k)
    print("n_n insert =", n_n_insert)

    n_n_orifice = (p_orifice)/(T_orifice*c.k)
    print("n_n orifice =", n_n_orifice)

    #n_n_gap = (p_gap)/(T_gap*c.k) #NOTE invalid really

    n_n_keeper = (p_keeper)/(T_keeper*c.k)

    #n_n_plume = (p_plume)/(T_plume*c.k) #NOTE invalid really


    #geometry
    area_tube_radial = L_tube*np.pi*Di_tube
    area_tube_axial = np.pi*0.25*(Di_tube**2 - Di_insert**2)
    volume_tube = L_tube*np.pi*0.25*Di_tube**2
    area_volume_tube = (area_tube_radial + area_tube_axial)/(volume_tube) #area for just losses over total volume

    area_insert_radial = L_insert*np.pi*Di_insert
    area_insert_axial = np.pi*0.25*(Di_insert**2 - D_orifice**2)
    volume_insert = L_insert*np.pi*0.25*Di_insert**2
    area_volume_insert = (area_insert_radial + area_insert_axial)/(volume_insert)

    area_orifice_radial = L_orifice*np.pi*D_orifice
    volume_orifice = L_orifice*np.pi*0.25*D_orifice**2
    area_volume_orifice = (area_orifice_radial)/(volume_orifice)


    T_eV_orifice, ion_rate_orifice = p.solve_T_eV(n_n_orifice, area_volume_orifice, m, a, E_i, q, T_orifice, D_orifice, prop)

    #insert solving
    if emitter == "lab6":
        T_insert_range = np.linspace(1500, 1950, 1000)
    elif emitter == "bao":
        T_insert_range = np.linspace(1300, 1950, 1000)
    n_e_range = np.full(len(T_insert_range), np.nan) #nan is to remove 0 values and just emit the point
    phi_s_range = np.full(len(T_insert_range), np.nan)
    I_e_range = np.full(len(T_insert_range), np.nan)
    I_i_range = np.full(len(T_insert_range), np.nan)
    I_b_range = np.full(len(T_insert_range), np.nan)
    P_net_range = np.full(len(T_insert_range), np.nan) #P_emit_cool - P_plasma
    T_eV_range = np.full(len(T_insert_range), np.nan) #23
    n_n_range = np.full(len(T_insert_range), np.nan)
    ions_produced_range = np.full(len(T_insert_range), np.nan)

    n_e_orifice_range = np.full(len(T_insert_range), np.nan)
    I_i_orifice_range = np.full(len(T_insert_range), np.nan)
    I_b_orifice_range = np.full(len(T_insert_range), np.nan)
    I_insert_range = np.full(len(T_insert_range), np.nan)

    for i, T_insert in enumerate(T_insert_range):
        n_n_insert = p_insert/(T_insert*c.k) #same pressure, gas now at T_insert
        T_eV_insert, ions_produced_insert = p.solve_T_eV(n_n_insert, area_volume_insert, m, a, E_i, q, T_insert, Di_insert, prop) #ions produced is per electron here
        I_k = 0 #A keeper current and 0 is for left at floating potential
        I_total = I_d_target + I_k #I_k in this sense also the efficiency loss term
        A_wall_insert = area_volume_insert*volume_insert
        emitter_area = L_insert*np.pi*Di_insert #NOTE should be L_emit but not important cause 0D will give average anyway
        #loss_rate_insert = min(p.bohm_loss(T_eV_insert, m, area_volume_insert, bohm_factor=0.61), p.ambipolar_loss(T_eV_insert, n_n_insert, T_insert, Di_insert, m, prop))
        n_e_orifice_i = p.orifice_n_e(I_d_target, T_eV_orifice, T_eV_insert, n_n_orifice, ion_rate_orifice, L_orifice, D_orifice, E_i, prop)
        R_orifice = p.plasma_resistance(n_e_orifice_i, n_n_orifice, T_eV_orifice, L_orifice, D_orifice, prop)
        I_insert = I_total
        converged = False
        phi_s_orifice = I_i_or = I_b_or = np.nan
        for _ in range(1000):
            n_e, phi_s, I_e, I_i, I_b = p.insert_balance(I_insert, T_eV_insert, T_insert, ions_produced_insert, volume_insert, A_wall_insert, emitter_area, emitter, m, E_i, n_n=n_n_insert, L=L_insert, D=Di_insert, prop=prop)
            if np.isnan(n_e):
                break
            phi_s_orifice = max(phi_s + 0.5*I_d_target*R_orifice, p.floating_potential(T_eV_orifice, m))
            I_i_cap = c.e*n_e_orifice_i*volume_orifice*ion_rate_orifice
            I_i_or, I_b_or = p.orifice_wall_currents(n_e_orifice_i, T_eV_orifice, phi_s_orifice, area_orifice_radial, m, I_i_cap)
            I_insert_new = I_total - (I_i_or - I_b_or)
            if abs(I_insert_new - I_insert) < 1e-6*I_total:
                converged = True
                break
            I_insert = 0.5*I_insert + 0.5*I_insert_new
        if not converged:
            n_e = phi_s = I_e = I_i = I_b = I_i_or = I_b_or = np.nan
        P_plasma, P_emit_cool = p.insert_heat(T_insert, T_eV_insert, phi_s, I_e, I_i, I_b, emitter, E_i)
        n_n_range[i] = n_n_insert
        T_eV_range[i] = T_eV_insert
        n_e_range[i] = n_e
        phi_s_range[i] = phi_s
        I_e_range[i] = I_e
        I_i_range[i] = I_i
        I_b_range[i] = I_b
        P_net_range[i] = P_plasma - P_emit_cool #this is the power into the emitter minues the power out
        ions_produced_range[i] = ions_produced_insert * (n_e*volume_insert) #(converted form per electron to total) this refers to the balance between the ones created from n_e at T_eV and the loss to the insert material and orifice inside wall through either bohm loss or ambipolar diffusion (net = 0)
        n_e_orifice_range[i] = n_e_orifice_i
        I_i_orifice_range[i] = I_i_or
        I_b_orifice_range[i] = I_b_or
        I_insert_range[i] = I_insert
        #print(T_ins, n_e)


    #operating point - just a nice to have
    P_loss = 13 #guess of the watts lost in conduction and radiation minus the heat input from heater
    f = P_net_range - P_loss
    ok = ~np.isnan(f)
    cross = np.where(ok[:-1] & ok[1:] & (np.sign(f[:-1]) != np.sign(f[1:])))[0]
    print("crosings of P_net = P_loss:", len(cross))
    if len(cross) == 0:
        print("no valid P_net range (W) =", np.nanmin(P_net_range), np.nanmax(P_net_range))
    else:
        j = cross[0]
        frac = f[j]/(f[j] - f[j+1])
        interp = lambda arr: arr[j] + frac*(arr[j+1] - arr[j])
        print("T_insert (K) =", interp(T_insert_range))
        print("T_eV insert (eV) =", interp(T_eV_range))
        print("n_e insert (m-3) =", interp(n_e_range))
        print("phi_s insert (V) =", interp(phi_s_range))
        print("I_e, I_i, I_b (A) =", interp(I_e_range), interp(I_i_range), interp(I_b_range))
        print("T_eV orifice (eV) =", T_eV_orifice)
        print("n_e orifice (m-3) =", interp(n_e_orifice_range))
        print("insert pressure (Torr) =", p_insert/c.torr)


    #particle flux balance - mdot is important but the number densities and diffusion paramters also need to be included, so start solve from insert again
    # i have n_n, n_e and n_i (assume quasineutral still) in ranges as function of T_insert for insert and n_n can be estimated from temperature estimate if needed
    # (these are not net terms) (these will all be ranges depending on T_insert)

    #values for fluxes later
    T_eV_tube = T_eV_range
    n_i_tube = n_e_tube = 0 #an assumptions goebel and katz make, i belive theres negligible charged particles in tube anyway
    Gamma_tube_walls_i = n_i_tube * volume_tube * p.ambipolar_loss(T_eV_tube, n_n_tube, T_tube, Di_tube, m, prop)
    Gamma_tube_walls_e = n_e_tube * volume_tube * p.ambipolar_loss(T_eV_tube, n_n_tube, T_tube, Di_tube, m, prop)
    Gamma_walls_insert_e = I_e_range/c.e
    Gamma_insert_walls_i = I_i_range/c.e
    Gamma_insert_walls_e = I_b_range/c.e
    Gamma_walls_insert_n = Gamma_insert_walls_i
    creation_rate_insert_i = ions_produced_range #number of ions produced but taken from the insert current solver where net ion flux from neighbouring regions is assumed negligible to this
    Gamma_gap_orifice_n = 0 #fair assumptions
    Gamma_gap_orifice_i = 0
    Gamma_gap_orifice_e = 0
    n_e_orifice = n_e_orifice_range
    creation_rate_orifice_i = n_e_orifice*volume_orifice*ion_rate_orifice
    Gamma_orifice_walls_i = I_i_orifice_range/c.e
    Gamma_orifice_walls_e = I_b_orifice_range/c.e


    #feed
    Gamma_feed_tube_n = mdot/m #in particles per second, presume no backflow for this boundary (update equations if changed), about e18
    Gamma_feed_tube_i = 0
    Gamma_feed_tube_e = 0

    #tube
    Gamma_net_tube_insert_n = Gamma_feed_tube_n #from tube to orifice (net charge build up is 0 but net charge flux at boundary does not have to be 0 obviously)
    Gamma_net_tube_insert_i = Gamma_feed_tube_i - Gamma_tube_walls_i #assuming no creation of ions here or the like
    Gamma_net_tube_insert_e = Gamma_feed_tube_e - Gamma_tube_walls_e

    #insert
    Gamma_net_insert_orifice_n = Gamma_net_tube_insert_n + Gamma_walls_insert_n - creation_rate_insert_i
    Gamma_net_insert_orifice_i = Gamma_net_tube_insert_i - Gamma_insert_walls_i + creation_rate_insert_i
    Gamma_net_insert_orifice_e = Gamma_net_tube_insert_e + Gamma_walls_insert_e - Gamma_insert_walls_e + creation_rate_insert_i #assumes singly charge reactions only

    #orifice
    Gamma_net_orifice_gap_n = Gamma_net_insert_orifice_n + Gamma_orifice_walls_i - creation_rate_orifice_i
    Gamma_net_orifice_gap_i = Gamma_net_insert_orifice_i - Gamma_orifice_walls_i + creation_rate_orifice_i
    Gamma_net_orifice_gap_e = Gamma_net_insert_orifice_e - Gamma_orifice_walls_e + creation_rate_orifice_i
    '''
    #gap
    Gamma_gap_keeper_n = Gamma_orifice_gap_n + Gamma_keeper_gap_n - Gamma_gap_orifice_n - Gamma_gap_walls_n - creation_rate_gap_i
    Gamma_gap_keeper_i = Gamma_orifice_gap_i + Gamma_keeper_gap_i - Gamma_gap_orifice_i - Gamma_gap_walls_i + creation_rate_gap_i
    Gamma_gap_keeper_e = Gamma_orifice_gap_e + Gamma_keeper_gap_e - Gamma_gap_orifice_e - Gamma_gap_walls_e + creation_rate_gap_i

    #keeper
    Gamma_keeper_plume_n = Gamma_gap_keeper_n + Gamma_plume_keeper_n - Gamma_keeper_gap_n - Gamma_keeper_walls_n - creation_rate_keeper_i
    Gamma_keeper_plume_i = Gamma_gap_keeper_i + Gamma_plume_keeper_i - Gamma_keeper_gap_i - Gamma_keeper_walls_i + creation_rate_keeper_i
    Gamma_keeper_plume_e = Gamma_gap_keeper_e + Gamma_plume_keeper_e - Gamma_keeper_gap_e - Gamma_keeper_walls_e + creation_rate_keeper_i
    '''


    I_d_check = c.e*(Gamma_net_orifice_gap_e - Gamma_net_orifice_gap_i)
    print("max I_d at gap - target (A) =", np.nanmax(np.abs(I_d_check - I_d_target)))
    print("min neutral flux at gap (s-1) =", np.nanmin(Gamma_net_orifice_gap_n))
    print("orifice wall net current range (A):", np.nanmin(I_i_orifice_range - I_b_orifice_range), np.nanmax(I_i_orifice_range - I_b_orifice_range))
    print("converged points:", np.sum(~np.isnan(I_i_orifice_range)), "of", len(T_insert_range))


    return (p_keeper_upstream, p_gap_upstream, p_orifice_upstream, p_insert_upstream, p_tube_upstream, #pressure throughout the cathode data
            mdot_range_ignition, p_range_tube_upstream_ignition, voltage_range_ignition, #ignition data
            n_n_range, T_eV_range, T_insert_range, n_e_range, phi_s_range, I_e_range, I_i_range, I_b_range, P_net_range, ions_produced_range, #first insert batch data
            Gamma_net_orifice_gap_e, Gamma_net_orifice_gap_i, Gamma_net_insert_orifice_e, Gamma_net_insert_orifice_i) #particle flux measures



def main():
    ini = initialise()


    (p_keeper_upstream, p_gap_upstream, p_orifice_upstream, p_insert_upstream, p_tube_upstream,
     mdot_range_ignition, p_range_tube_upstream_ignition, voltage_range_ignition,
     n_n, T_eV, T_insert, n_e, phi_s, I_e, I_i, I_b, P_net, ions_produced,
     Gamma_net_orifice_gap_e, Gamma_net_orifice_gap_i, Gamma_net_insert_orifice_e, Gamma_net_insert_orifice_i) = solve(
        ini.L_tube, ini.Di_tube, ini.Do_tube,
        ini.L_insert, ini.Di_insert, ini.Do_insert,
        ini.L_orifice, ini.D_orifice,
        ini.L_gap,
        ini.L_keeper, ini.D_keeper, ini.Di_keeper,
        ini.emitter,
        ini.prop, ini.mdot,
        ini.p_plume, ini.I_d_target,
        ini.T_universal_guess, ini.T_tube, ini.T_insert, ini.T_orifice, ini.T_gap, ini.T_keeper, ini.T_plume)


    #temp diagnostics
    print("pressure into cathode (Torr) =", p_tube_upstream/c.torr)


    #save the data i want
    save_ignition = "False"
    if save_ignition:
        np.save(f"Data/breakdown_mfr_{ini.prop}_{ini.L_gap/c.mm}mm.npy", mdot_range_ignition)
        np.save(f"Data/breakdown_pressure_{ini.prop}_{ini.L_gap/c.mm}mm.npy", p_range_tube_upstream_ignition)
        np.save(f"Data/breakdown_voltage_{ini.prop}_{ini.L_gap/c.mm}mm.npy", voltage_range_ignition)

    save_insert = "True"
    if save_insert:
        np.save(f"Data/insert_n_n_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", n_n)
        np.save(f"Data/insert_T_eV_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", T_eV)
        np.save(f"Data/insert_T_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", T_insert) #insert temperature
        np.save(f"Data/insert_electrondensity_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", n_e) #electron number dnesity in insert region
        np.save(f"Data/insert_sheathpotential_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", phi_s) #sheath potential over emitter
        np.save(f"Data/insert_emissioncurrent_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", I_e) #electron current from emitter
        np.save(f"Data/insert_ionbombardment_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", I_i) #ion current into emitter
        np.save(f"Data/insert_electronbombardment_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", I_b) #electron current into emitter
        np.save(f"Data/insert_Pnet_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", P_net) #additional net power produced to insert wihtout conduction and radiation of thermal sim
        np.save(f"Data/insert_ionproductionrate_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", ions_produced)

    save_flux = "True"
    if save_flux:
        np.save(f"Data/insert_fluxinsertorificee_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", Gamma_net_insert_orifice_e)
        np.save(f"Data/insert_fluxinsertorificei_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", Gamma_net_insert_orifice_i)
        np.save(f"Data/insert_fluxorificegape_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", Gamma_net_orifice_gap_e)
        np.save(f"Data/insert_fluxorificegapi_{ini.prop}_{ini.mdot}_{ini.geometry_name}.npy", Gamma_net_orifice_gap_i)


if __name__ == "__main__":
    main()