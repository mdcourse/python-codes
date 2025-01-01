import csv

def import_data(file_path):
    """
    Imports a data file with a variable number of columns into a list
    of numerical arrays. The first line (header) is read as a string.

    Parameters:
    - file_path (str): Path to the data file.

    Returns:
    - header (str): The header line as a string.
    - data (list of lists): List where each sublist contains the numeric values of a row.
    """
    data = []
    header = ""
    with open(file_path, mode='r') as file:
        # Read the header as a string
        header = file.readline().strip()
        # Use csv.reader to process the remaining lines
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            # Filter out empty fields resulting from multiple spaces
            filtered_row = [float(value) for value in row if value]
            data.append(filtered_row)
    return header, data

