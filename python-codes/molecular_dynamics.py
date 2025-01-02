import numpy as np
from initialize_simulation import InitializeSimulation


class MolecularDynamics(InitializeSimulation):
    def __init__(self,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)

