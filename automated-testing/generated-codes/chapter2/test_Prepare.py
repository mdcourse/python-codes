

from Prepare import Prepare

self = Prepare(number_atoms=[2, 3],
    epsilon=[0.1, 1.0], # kcal/mol
    sigma=[3, 6], # A
    atom_mass=[1, 1], # g/mol
    )
print("Reference energy:")
print(self.reference_energy)
print("Reference distance:")
print(self.reference_distance)
print("array_epsilon_ij:")
print(self.array_epsilon_ij)
print("array_sigma_ij:")
print(self.array_sigma_ij)

