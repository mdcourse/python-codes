

import numpy as np

def update_dump_file(code, filename, velocity=False):
    if code.dumping_period is not None:
        if code.step % code.dumping_period == 0:
            # Convert units to the *real* dimensions
            box_boundaries = code.box_boundaries*code.ref_length # Angstrom
            atoms_positions = code.atoms_positions*code.ref_length # Angstrom
            atoms_types = code.atoms_type
            if velocity:
                atoms_velocities = code.atoms_velocities \
                    * code.ref_length/code.ref_time # Angstrom/femtosecond
            # Start writting the file
            if code.step == 0: # Create new file
                f = open(code.data_folder + filename, "w")
            else: # Append to excisting file
                f = open(code.data_folder + filename, "a")
            f.write("ITEM: TIMESTEP\n")
            f.write(str(code.step) + "\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(str(np.sum(code.number_atoms)) + "\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for dim in np.arange(3):
                f.write(str(box_boundaries[dim][0].magnitude) + " "
                        + str(box_boundaries[dim][1].magnitude) + "\n")
            cpt = 1
            if velocity:
                f.write("ITEM: ATOMS id type x y z vx vy vz\n")
                characters = "%d %d %.3f %.3f %.3f %.3f %.3f %.3f %s"
                for type, xyz, vxyz in zip(atoms_types,
                                        atoms_positions.magnitude,
                                        atoms_velocities.magnitude):
                    v = [cpt, type, xyz[0], xyz[1], xyz[2],
                            vxyz[0], vxyz[1], vxyz[2]]
                    f.write(characters % (v[0], v[1], v[2], v[3], v[4],
                                        v[5], v[6], v[7], '\n'))
                    cpt += 1
            else:
                f.write("ITEM: ATOMS id type x y z\n")
                characters = "%d %d %.3f %.3f %.3f %s"
                for type, xyz in zip(atoms_types,
                                    atoms_positions.magnitude):
                    v = [cpt, type, xyz[0], xyz[1], xyz[2]]
                    f.write(characters % (v[0], v[1], v[2],
                                        v[3], v[4], '\n'))
                    cpt += 1
            f.close()  

