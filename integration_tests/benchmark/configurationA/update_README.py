import numpy as np
import re

# Function to calculate the average and standard error of the second column in a data file
def calculate_statistics(filename):
    try:
        data = np.loadtxt(filename, usecols=1)  # Load the second column
        mean = np.mean(data)
        std_error = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard error of the mean
        return mean, std_error
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

# Filenames
config_file = "README.md"  # Your configuration file
energy_file = "Epot.dat"
pressure_file = "pressure.dat"

# Calculate statistics
average_epot, error_epot = calculate_statistics(energy_file)
average_press, error_press = calculate_statistics(pressure_file)

if average_epot is None or average_press is None:
    print("Failed to calculate averages or errors. Exiting.")
    exit(1)

# Update the configuration file
try:
    with open(config_file, 'r') as file:
        config_lines = file.readlines()

    with open(config_file, 'w') as file:
        for line in config_lines:
            if re.match(r"^Epot\s*=", line):
                file.write(f"Epot = {average_epot:.3f} ± {error_epot:.3f} kcal/mol\n")
            elif re.match(r"^press\s*=", line):
                file.write(f"press = {average_press:.3f} ± {error_press:.3f} atm\n")
            else:
                file.write(line)

    print(f"Updated {config_file} with Epot = {average_epot:.3f} ± {error_epot:.3f} kcal/mol and press = {average_press:.3f} ± {error_press:.3f} atm")
except Exception as e:
    print(f"Error updating {config_file}: {e}")
