

from InitializeSimulation import InitializeSimulation

self = InitializeSimulation(number_atoms=[2, 3],
    epsilon=[0.1, 1.0], # kcal/mol
    sigma=[3, 6], # A
    atom_mass=[1, 1], # g/mol
    box_dimensions=[20, 20, 20], # A
    )
print("Atom positions:")
print(self.atoms_positions)

