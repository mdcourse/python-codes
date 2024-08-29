

import os
import logging

# Function to set up the logger
def setup_logger(folder_name, overwrite=False):
    # Create a custom logger
    logger = logging.getLogger('simulation_logger')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Disable propagation to prevent double logging

    # Clear any existing handlers if this function is called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers for console and file
    console_handler = logging.StreamHandler()  # To log to the terminal
    log_file_path = os.path.join(folder_name, 'simulation.log')

    # Use 'w' mode to overwrite the log file if overwrite is True, else use 'a' mode to append
    file_mode = 'w' if overwrite else 'a'
    file_handler = logging.FileHandler(log_file_path, mode=file_mode)  # To log to a file

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def log_simulation_data(code):

    # Setup the logger with the folder name, overwriting the log if code.step is 0
    logger = setup_logger(code.data_folder, overwrite=(code.step == 0))

    # Logging the simulation data
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
                press = code.pressure * code.reference_pressure  # Atm
                logger.info(f"{code.step} {Epot:.2f} {press:.2f}")

