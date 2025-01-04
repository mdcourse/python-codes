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
density_file = "density.dat"

# Calculate statistics
average_density, error_density = calculate_statistics(density_file)

if average_density is None or error_density is None:
    print("Failed to calculate averages or errors. Exiting.")
    exit(1)

# Update the configuration file
try:
    with open(config_file, 'r') as file:
        config_lines = file.readlines()

    with open(config_file, 'w') as file:
        for line in config_lines:
            if re.match(r"^Epot\s*=", line):
                file.write(f"density = {average_density:.3f} ± {error_density:.3f} g/cm3\n")
            else:
                file.write(line)

    print(f"Updated {config_file} with density = {average_density:.3f} ± {error_density:.3f} g/cm3")
except Exception as e:
    print(f"Error updating {config_file}: {e}")
