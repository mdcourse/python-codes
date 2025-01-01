import numpy as np

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Create a custom logger
logger = logging.getLogger('simulation_logger')
logger.setLevel(logging.INFO)
# Disable propagation to prevent double logging
logger.propagate = False

console_handler = logging.StreamHandler()  # To log to the terminal
file_handler = logging.FileHandler('simulation.log')  # To log to a file
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def update_dump_file(code, filename, velocity=False):
    if code.dumping_period is not None:
        if code.step % code.dumping_period == 0:
            # Convert units to the *real* dimensions
            box_boundaries = code.box_boundaries\
                * code.reference_distance # Angstrom
            atoms_positions = code.atoms_positions\
                * code.reference_distance # Angstrom
            atoms_types = code.atoms_type
            if velocity:
                atoms_velocities = code.atoms_velocities \
                    * code.reference_distance/code.reference_time # Angstrom/femtosecond
            # Start writting the file
            if code.step == 0: # Create new file
                f = open(code.data_folder + filename, "w")
            else: # Append to excisting file
                f = open(code.data_folder + filename, "a")
            f.write("ITEM: TIMESTEP\n")
            f.write(str(code.step) + "\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(str(code.total_number_atoms) + "\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for dim in np.arange(code.dimensions):
                f.write(str(box_boundaries[dim][0]) + " "
                        + str(box_boundaries[dim][1]) + "\n")
            cpt = 1
            if velocity:
                f.write("ITEM: ATOMS id type x y z vx vy vz\n")
                characters = "%d %d %.3f %.3f %.3f %.3f %.3f %.3f %s"
                for type, xyz, vxyz in zip(atoms_types,
                                        atoms_positions,
                                        atoms_velocities):
                    v = [cpt, type, xyz[0], xyz[1], xyz[2],
                            vxyz[0], vxyz[1], vxyz[2]]
                    f.write(characters % (v[0], v[1], v[2], v[3], v[4],
                                        v[5], v[6], v[7], '\n'))
                    cpt += 1
            else:
                f.write("ITEM: ATOMS id type x y z\n")
                characters = "%d %d %.3f %.3f %.3f %s"
                for type, xyz in zip(atoms_types,
                                    atoms_positions):
                    v = [cpt, type, xyz[0], xyz[1], xyz[2]]
                    f.write(characters % (v[0], v[1], v[2],
                                        v[3], v[4], '\n'))
                    cpt += 1
            f.close()

def log_simulation_data(code):
    if code.thermo_period is not None:
        if code.step % code.thermo_period == 0:
            if code.step == 0:
                Epot = code.compute_potential(output="potential") \
                    * code.reference_energy  # kcal/mol
            else:
                Epot = code.Epot * code.reference_energy  # kcal/mol
            if code.step == 0:
                if code.thermo_outputs == "Epot":
                    logger.info(f"step Epot")
                elif code.thermo_outputs == "Epot-MaxF":
                    logger.info(f"step Epot MaxF")
                elif code.thermo_outputs == "Epot-press":
                    logger.info(f"step Epot press")
            if code.thermo_outputs == "Epot":
                logger.info(f"{code.step} {Epot:.2f}")
            elif code.thermo_outputs == "Epot-MaxF":
                logger.info(f"{code.step} {Epot:.2f} {code.MaxF:.2f}")
            elif code.thermo_outputs == "Epot-press":
                code.calculate_pressure()
                press = code.pressure \
                    * code.reference_pressure  # Atm
                logger.info(f"{code.step} {Epot:.2f} {press:.2f}")    

