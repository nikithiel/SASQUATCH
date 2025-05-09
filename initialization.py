from preprocessing import read_ansys_output_to_dfs, mean_of_timesteps, normalize_data, scale_data, mv_uq_procect_preprocessing
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

def read_data(data_path, output_path, da=False, save_data='data.cvs'): #deprecated, now it is combined in preprocessing() in preprocessing.py
    """Reads the .xlsx, .csv, or .out files with data.

    Args:
        - data_path: int -> path to data
        - output_path: str -> path to save data
        - da: bool -> True if you perform Data Analysis
        - save_data: bool -> whether data is saved to .csv
    Returns:
        - X_df_all: dataFrame -> contains all data
    """
    # ----- DATA Preprocessing ----- #
    # Reading data from Ansys output and calcuate mean over timesteps
    
    data_path = f"input_data/{data_path}"
    output_path = './output_data/' + output_path + '/'

    if ".xlsx" in data_path: data_df_all = pd.read_excel(data_path) # xlsx files
    if ".csv" in data_path: data_df_all = pd.read_csv(data_path) # csv files
    else: data_df_all = read_ansys_output_to_dfs(data_path, da=da) # Ansys Output Files
    
    # Saving data
    if save_data.lower() != 'false':
        # Check directory and creating directory if necessary
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data_df_all.to_csv(output_path + "/" + save_data, index=False)
    
    return data_df_all

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