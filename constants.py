cm = 0.01
mm = 0.001
mg = 0.000001
torr = 133.322368
j_0_0 = 2.405 #first 0 of the 0th order bessel function

e = 1.602176634e-19
k = 1.380649e-23
R_u = 8.31446261815324
eps0 = 8.85418782e-12 #permittivity of vacuum
m_e = 9.10938e-31

Xe_m = 0.13129 #kg mol-1
Xe_m_kg = 2.18e-25
Xe_mu = 8.8e-5 #Pa s
Xe_E_i = 12.13*e #J for first ionisation
Xe_a = 4e-14
Xe_q = 6
Xe_coll_i_n = 1.6e-18 #NOTE this is cross section for collision between said things in m^2
Xe_coll_e_n = 2.5e-19 #NOTE could be a source of inacuracy

Bi_p1 = 0.6746 #NOTE also can be equation
Bi_p2 = 1 - Bi_p1
Bi_m = 0.2089804*Bi_p1 + 2*0.2089804*Bi_p2
Bi_m_kg = 3.4702e-25*Bi_p1 + 2*3.4702e-25*Bi_p2
Bi_mu = 1.106855e-4 #NOTE Pa s should depend on T, this is at 1700K
Bi_E_i = 7.289*e #J for first ionisation
Bi_a = 4e-14
Bi_q = 5
Bi_coll_i_n = Xe_coll_i_n * Bi_m/Xe_m #NOTE finding values would be much better than scaling
Bi_coll_e_n = Xe_coll_e_n * Bi_m/Xe_m #NOTE 

ignition_voltage = 100

phi_0_lab6 = 2.66*e #in J
alpha_lab6 = 1.23e-4 *e # temperature term
A_lab6 = 120/cm**2

phi_0_bao = 1.67*e #barium oxide in porous tungsten
alpha_bao = 2.82e-4 *e
A_bao = 120/cm**2
