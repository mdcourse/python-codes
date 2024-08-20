

class Prepare:
    def __init__(self,
                number_atoms=[10],  # List
                epsilon=[0.1],  # List - Kcal/mol
                sigma=[1],  # List - Angstrom
                atom_mass=[1],  # List - g/mol
                *args,
                **kwargs):
        self.number_atoms = number_atoms
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass
        super().__init__(*args, **kwargs)

