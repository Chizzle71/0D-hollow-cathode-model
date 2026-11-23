import solver
import constants as c

# Parameters to sweep
props = ["Xe", "Bi"]

geometries = {
    "1": 1.02*c.mm,
    "2": 0.5*1.02*c.mm
}

mdots = [0.24*c.mg]#, 0.48*c.mg]


# Run parameter sweep
for prop in props:
    for geometry_name, D_orifice in geometries.items():
        for mdot in mdots:

            # Change simulation parameters
            solver.initialise.prop = prop
            solver.initialise.geometry_name = geometry_name
            solver.initialise.D_orifice = D_orifice
            solver.initialise.mdot = mdot

            # Run simulation
            solver.main()
