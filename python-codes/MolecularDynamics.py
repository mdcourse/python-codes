import numpy as np
from InitializeSimulation import InitializeSimulation


class MolecularDynamics(InitializeSimulation):
    def __init__(self,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)

