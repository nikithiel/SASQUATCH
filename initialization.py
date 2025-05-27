from preprocessing import read_ansys_output_to_dfs, mean_of_timesteps, normalize_data, scale_data
import pandas as pd
import os

def read_user_defined_parameters(filename):
    """Reading user defined parameter from config.txt file.

    Args:
        - filename (str): filename with parameters
    Returns:
        dict: dictionary of parameters and their values
    """

    config_data = {}

    # Read the data and populate the dictionary
    with open(filename, 'r') as file:
        for line in file:
            # Skip empty lines and lines starting with '#' (comments)
            if not (not line.strip() or line.strip().startswith('#')):
                splitted_line = line.split()
                key = splitted_line[0]

                values_end = next((i for i, x in enumerate(splitted_line) if x.startswith('#')), None)
                if len(splitted_line) > 1:
                    values = splitted_line[1:values_end]

                    # Store the values as a list
                    config_data_values = []
                    for value in values:
                        if value.lower() == 'true':
                            config_data_values.append(True)
                        elif value.lower() == 'false':
                            config_data_values.append(False)
                        elif value.isdigit():
                            config_data_values.append(int(value))
                        else:
                            config_data_values.append(value)
                if len(config_data_values) == 1:
                    config_data[key] = config_data_values[0]
                else:
                    config_data[key] = config_data_values

    return config_data

def get_data_bounds(df):
    """Get bounds of data.

    Args:
        df: dataFrame -> contains data
    Returns:
        dict_of_bounds: dict -> dictionary of bounds: {column:{min:min_value, max:max_value}
    """
    # calculate bounds of each column in df
    dict_of_bounds = df.describe().loc[['min', 'max']].to_dict()
    return dict_of_bounds