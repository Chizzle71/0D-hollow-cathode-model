import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import constants as c
import physics as p



def set_font_sizes(legend_size=14):
    # Set LaTeX for text rendering
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    # Set font sizes based on legend size
    plt.rc("legend", fontsize=legend_size)  # legend fontsize (base size)
    plt.rc(
        "axes", titlesize=legend_size * 1.2
    )  # fontsize of the axes title (20% larger)
    plt.rc(
        "axes", labelsize=legend_size * 1.0
    )  # fontsize of the x and y labels (same as legend)
    plt.rc(
        "xtick", labelsize=legend_size * 0.8
    )  # fontsize of the tick labels (20% smaller)
    plt.rc(
        "ytick", labelsize=legend_size * 0.8
    )  # fontsize of the tick labels (20% smaller)
    plt.rc(
        "figure", titlesize=legend_size * 1.4
    )  # fontsize of the figure title (40% larger)
    annotation_size = (
        legend_size * 0.9
    )  # annotation text size (10% smaller than legend)
    plt.rc("font", size=annotation_size)  # default font size for annotations

    # Return the settings dict for reference (optional)
    settings = {
        "legend": legend_size,
        "axes.title": legend_size * 1.2,
        "axes.label": legend_size * 1.0,
        "xtick.label": legend_size * 0.8,
        "ytick.label": legend_size * 0.8,
        "figure.title": legend_size * 1.4,
        "annotation": annotation_size,
    }

    return settings
set_font_sizes()



points_max = 100
def load_insert(path):
    arr = np.load(path)
    if points_max is None or arr.ndim == 0:
        return arr
    step = max(1, len(arr)//points_max)
    return arr[::step]



n_colours = 3
shades = LinearSegmentedColormap.from_list("purple_cyan", ["#5a00a4", "#00b7ff", "#00c407", "#d5d500"])(np.linspace(0, 1, n_colours))
shades2 = LinearSegmentedColormap.from_list("red_pink", ["#8b0000", "#ff1a1a", "#ff7f00", "#ff00aa"])(np.linspace(0, 1, n_colours))



def find_gaps_ignition():
    def gaps_for(prop):
        return {p.stem.split("_")[-1].removesuffix("mm") for p in Path("Data").glob(f"breakdown_mfr_{prop}_*mm.npy")}
    return sorted(gaps_for("Xe") & gaps_for("Bi"), key=float)



def breakdown_voltage():
    fig, ax = plt.subplots(constrained_layout=True)
    for i, gap in enumerate(find_gaps_ignition()):
        mdot_range_ignition_Xe = np.load(f"Data/breakdown_mfr_Xe_{gap}mm.npy")
        voltage_range_ignition_Xe = np.load(f"Data/breakdown_voltage_Xe_{gap}mm.npy")
        mdot_range_ignition_Bi = np.load(f"Data/breakdown_mfr_Bi_{gap}mm.npy")
        voltage_range_ignition_Bi = np.load(f"Data/breakdown_voltage_Bi_{gap}mm.npy")
        ax.loglog(mdot_range_ignition_Xe/c.mg, voltage_range_ignition_Xe, color=shades[i % n_colours], linestyle="--", label=f"Xe, {gap} mm")
        ax.loglog(mdot_range_ignition_Bi/c.mg, voltage_range_ignition_Bi, color=shades[i % n_colours], linestyle=":", label=f"Bi, {gap} mm")
    ax.set_title("Ignition")
    ax.set_xlabel("mfr (mgs$^{-1}$)")
    ax.set_ylabel("V")
    ax.legend()
    fig.savefig("Figures/Ignition_voltage.png", dpi=300)



def breakdown_pressure():
    fig, ax = plt.subplots(constrained_layout=True)
    for i, gap in enumerate(find_gaps_ignition()):
        mdot_range_ignition_Xe = np.load(f"Data/breakdown_mfr_Xe_{gap}mm.npy")
        p_range_tube_upstream_ignition_Xe = np.load(f"Data/breakdown_pressure_Xe_{gap}mm.npy")
        mdot_range_ignition_Bi = np.load(f"Data/breakdown_mfr_Bi_{gap}mm.npy")
        p_range_tube_upstream_ignition_Bi = np.load(f"Data/breakdown_pressure_Bi_{gap}mm.npy")
        ax.loglog(mdot_range_ignition_Xe/c.mg, p_range_tube_upstream_ignition_Xe/c.torr, color=shades[i % n_colours], linestyle="--", label=f"Xe, {gap} mm")
        ax.loglog(mdot_range_ignition_Bi/c.mg, p_range_tube_upstream_ignition_Bi/c.torr, color=shades[i % n_colours], linestyle=":", label=f"Bi, {gap} mm")
    ax.set_title("Ignition")
    ax.set_xlabel("mfr (mgs$^{-1}$)")
    ax.set_ylabel("Pressure (Torr)")
    ax.legend()
    fig.savefig("Figures/Ignition_pressure.png", dpi=300)



def find_gaps_insert(prop):
    keys = {file.stem.removeprefix(f"insert_T_{prop}_") for file in Path("Data").glob(f"insert_T_{prop}_*.npy")}
    return sorted(keys, key=lambda k: (float(k.split("_")[0]), int(k.split("_")[1])))



def insert_electrondensity_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            n_e = load_insert(f"Data/insert_electrondensity_{prop}_{key}.npy")
            ax.semilogy(T_ins, n_e, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("n$_e$ (m$^{-3}$)")
    ax.legend()
    fig.savefig("Figures/Insert_electrondensity.png", dpi=300)



def insert_emissioncurrent_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            I_e = load_insert(f"Data/insert_emissioncurrent_{prop}_{key}.npy")
            ax.plot(T_ins, I_e, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("I$_e$ (A)")
    ax.legend()
    fig.savefig("Figures/Insert_emissioncurrent.png", dpi=300)



def insert_currents_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, colours in (("Xe", shades), ("Bi", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            I_e = load_insert(f"Data/insert_emissioncurrent_{prop}_{key}.npy")
            I_i = load_insert(f"Data/insert_ionbombardment_{prop}_{key}.npy")
            I_b = load_insert(f"Data/insert_electronbombardment_{prop}_{key}.npy")
            ax.plot(T_ins, I_e, color=colours[i % n_colours], marker="x", linestyle="none", label=f"{prop} I$_e$")
            ax.plot(T_ins, I_i, color=colours[i % n_colours], marker="+", linestyle="none", label=f"{prop} I$_i$")
            ax.plot(T_ins, I_b, color=colours[i % n_colours], marker="o", markerfacecolor="none", linestyle="none", label=f"{prop} I$_b$")
            ax.axhline(np.nanmedian(I_e + I_i - I_b), color="black", linewidth=0.8)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("I (A)")
    ax.legend()
    fig.savefig("Figures/Insert_currents.png", dpi=300)



def insert_pnet_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            Pnet = load_insert(f"Data/insert_Pnet_{prop}_{key}.npy")
            ax.plot(T_ins, Pnet, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
            ax.axhline(13, color="black", linewidth=0.8)
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"P$_{net}$ (W)")
    ax.legend()
    fig.savefig("Figures/Insert_power_surplus.png", dpi=300)



def insert_sheathpotential_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            phi_s = load_insert(f"Data/insert_sheathpotential_{prop}_{key}.npy")
            ax.semilogy(T_ins, phi_s, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"$\phi_s$ (V)")
    ax.legend()
    fig.savefig("Figures/Insert_sheathpotential.png", dpi=300)



def insert_T_eV_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            T_eV = load_insert(f"Data/insert_T_eV_{prop}_{key}.npy")
            ax.plot(T_ins, T_eV, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"T$_{eV}$ (eV)")
    ax.legend()
    fig.savefig("Figures/Insert_T_eV.png", dpi=300)



def insert_n_n_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            n_n = load_insert(f"Data/insert_n_n_{prop}_{key}.npy")
            ax.plot(T_ins, n_n, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"n$_n$ (m$^{-3}$)")
    ax.legend()
    fig.savefig("Figures/Insert_n_n.png", dpi=300)



def insert_ionproductionrate_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            ionproductionrate = load_insert(f"Data/insert_ionproductionrate_{prop}_{key}.npy")
            ax.plot(T_ins, ionproductionrate, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"ion production rate (s$^{-1}$)")
    ax.legend()
    fig.savefig("Figures/Insert_ion_production_rate.png", dpi=300)



def insert_fluxinsertorificei_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            fluxinsertorificei = load_insert(f"Data/insert_fluxinsertorificei_{prop}_{key}.npy")
            ax.plot(T_ins, fluxinsertorificei, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"net ion flux from insert to orifice (s$^{-1}$)")
    ax.legend()
    fig.savefig("Figures/Flux_insert_orifice_i.png", dpi=300)



def insert_fluxinsertorificee_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            fluxinsertorificee = load_insert(f"Data/insert_fluxinsertorificee_{prop}_{key}.npy")
            ax.plot(T_ins, fluxinsertorificee, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"net electron flux from insert to orifice (s$^{-1}$)")
    ax.legend()
    fig.savefig("Figures/Flux_insert_orifice_e.png", dpi=300)



def insert_fluxorificegape_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            fluxorificegape = load_insert(f"Data/insert_fluxorificegape_{prop}_{key}.npy")
            ax.plot(T_ins, fluxorificegape, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"net electron flux from orifice to gap (s$^{-1}$)")
    ax.legend()
    fig.savefig("Figures/Flux_orifice_gap_e.png", dpi=300)



def insert_fluxorificegapi_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            fluxorificegapi = load_insert(f"Data/insert_fluxorificegapi_{prop}_{key}.npy")
            ax.plot(T_ins, fluxorificegapi, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"net ion flux from orifice to gap (s$^{-1}$)")
    ax.legend()
    fig.savefig("Figures/Flux_orifice_gap_i.png", dpi=300)



def insert_fluxorificegapi_Tinsert():
    fig, ax = plt.subplots(constrained_layout=True)
    for prop, marker, colours in (("Xe", "x", shades), ("Bi", "+", shades2)):
        for i, key in enumerate(find_gaps_insert(prop)):
            mdot = float(key.split("_")[0])
            geometry = key.split("_")[1]
            T_ins = load_insert(f"Data/insert_T_{prop}_{key}.npy")
            fluxorificegapi = load_insert(f"Data/insert_fluxorificegapi_{prop}_{key}.npy")
            ax.plot(T_ins, fluxorificegapi, color=colours[i % n_colours], marker=marker, linestyle="none", label=f"{prop}, {mdot/c.mg:.2f} mgs$^{{-1}}$, geometry {geometry}")
    ax.set_title("Insert")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"net ion flux from orifice to gap (s$^{-1}$)")
    ax.legend()
    fig.savefig("Figures/Flux_orifice_gap_i.png", dpi=300)



def main():
    #breakdown_pressure()
    #breakdown_voltage()

    insert_electrondensity_Tinsert()
    insert_emissioncurrent_Tinsert()
    insert_currents_Tinsert()
    insert_pnet_Tinsert()
    insert_sheathpotential_Tinsert()
    insert_T_eV_Tinsert()
    insert_n_n_Tinsert()
    insert_ionproductionrate_Tinsert()

    insert_fluxinsertorificee_Tinsert()
    insert_fluxinsertorificei_Tinsert()
    insert_fluxorificegape_Tinsert()
    insert_fluxorificegapi_Tinsert()



if __name__ == "__main__":
    main()