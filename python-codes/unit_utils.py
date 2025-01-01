import pint
import numpy as np

def validate_units(value: pint.Quantity, expected_unit: pint.Unit, name: str):
    """Validate the units of a given quantity."""
    if not isinstance(value, pint.Quantity):
        raise TypeError(f"Invalid type for {name}: expected a 'pint.Quantity', got '{type(value).__name__}'.")
    if value.units != expected_unit:
        raise ValueError(f"Invalid units for {name}: expected {expected_unit}, got {value.units}")

def nondimensionalize_single(value: pint.Quantity, ref_value: pint.Quantity) -> float:
    """Nondimensionalize a single quantity."""
    return (value / ref_value).magnitude

def nondimensionalize_array(array: np.ndarray, ref_value: pint.Quantity) -> np.ndarray:
    """Nondimensionalize a NumPy array."""
    return (array / ref_value).magnitude
