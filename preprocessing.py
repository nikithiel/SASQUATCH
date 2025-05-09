import os
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from plotting import plot_boxplots

def read_ansys_output_to_dfs(data_path, da=False):
    """Reading the .out files from ansys
    
    Args:
        - data_path : str -> path to input data / folder (ansys)
        - da : bool -> whether data analysis is running
        - normalize : bool -> whether normalize the data. The default is True.
        - scaler : str -> which scaler is used. The default is 'none'.
    Returns:
        final_combined_df : dataFrame -> all data in one dataframe

    """
    # Define the root folder path
    root_folder_path = data_path 

    # Initialize an empty list to store DataFrames for each subfolder
    dfs_per_subfolder = []

    # Get the list of all subfolders in the root folder
    subfolders = [subfolder for subfolder in os.listdir(root_folder_path) if os.path.isdir(os.path.join(root_folder_path, subfolder))]
    
    # Iterate over each subfolder
    for subfolder in subfolders:
        if not subfolder[0].isdigit() : pass# and subfolder[0] == '1': pass# or subfolder[0] == '7' or subfolder[0] == '5': pass
        else:
            folder_path = os.path.join(root_folder_path, subfolder)
            print("   Reading Data From: ", folder_path)
            
            # Get dictionary of input parameters from folder title
            try:
                input_parameters = read_input_parameter(subfolder)
            
                # Get the list of all .out files in the folder
                file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".out")]
                
                # Initialize an empty DataFrame to store the combined data for this folder
                combined_df = pd.DataFrame()
                    
                # Iterate over each .out file in the folder
                for file_path in file_paths:
                    # Read the data from the file into a DataFrame
                    with open(file_path, 'r') as file:
                        lines = file.readlines()[2:]  # Skip the first two lines
                        column_names_line = lines[0].strip().strip('()')
                        column_names = [name.strip('"') for name in column_names_line.split('"') if name.strip()]
                        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=3, header=None, names=column_names)    
                        for param, value in input_parameters.items():
                            combined_df[param] = value
                        df = df.apply(pd.to_numeric, errors='coerce')
                        combined_df = pd.concat([combined_df, df], axis=1)
                combined_df = combined_df.T.drop_duplicates().T
                
                # Plotting boxplots of data of each folder
                if da: plot_boxplots(combined_df, combined_df, "Boxplots of Dataframe in Subfolder "+subfolder)
                
                # Append data of subfolder to list of all data
                combined_df.reset_index(drop=True, inplace=True)
                dfs_per_subfolder.append(combined_df)
            except:
                print("  !!! Error while reading data: ", folder_path)
    
    final_combined_df = pd.concat(dfs_per_subfolder, axis=0, ignore_index=True)
    final_combined_df.reset_index(drop=True, inplace=True)
    final_combined_df = final_combined_df.drop_duplicates()
    
    return final_combined_df

def mean_of_timesteps(df, input_parameter):
    """Returns the reduced df by calculating means for each input_parameter_set.
    
    Args:
        - df: dataFrame -> contains data
        - input_parameter: list -> of strings with input parameters
    Returns:
        dataFrame -> reduced dataFrame
    """
    return df.groupby(input_parameter).mean().reset_index()

def normalize_data(df):
    """Normalizing the dataframe.
    
    Args:
        df: dataFrame -> to normalize
    Returns:
        dataFrame -> normalized dataFrame
    """
    normalized_data = normalize(df, norm='l2', axis=0)
    return pd.DataFrame(normalized_data, columns=df.columns)

def scale_data(df, scaler='none'):
    """Scaling the data:

    Args:
        - df: dataFrame -> contains data
        - scaler: str -> scaler type
    Returns:
        dataFrame -> scaled dataFrame
    """
    # Scaling data:
    if scaler == 'minmax':
        return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    elif scaler == 'standard':
        return pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    else:
        print("!!! Error: Unknown scaler: ",scaler," !!!")

def read_input_parameter(subfolder_string):
    """Extracting input parameter values from folder names.

    Args:
        subfolder_string : str -> subfolder name

    Returns:
        dict : dict -> dictionary of inputparameters and their values
    """
    dict = {}
    parts = subfolder_string.split('_')
    for part in parts[1:]:
        key_end = next(i for i, char in enumerate(part) if char.isdigit() or char == '-')
        key = part[:key_end]
        value = float(part[key_end:])
        dict[key] = value
    return dict

'''def preprocessing(**args):
    """Wrapper for project specific preprocessing.
    """
    return mv_uq_procect_preprocessing(**args)'''
    
def mv_uq_procect_preprocessing(df, input_parameter, output_parameter, output_path, \ #deprecated, now preprocessing() is the main preprocessing function
                                normalize=False, scaler='none', get_mean=False,
                                is_transient=False, lower_bound=721, upper_bound=1200):
    """Preprocessing for mitral valve uncertainty quantification. Cutting first 720 Time Steps.
    
    Args:
        - df: dataFrame -> contains data
        - input_parameter: list -> list of input parameter
        - output_parameter: list -> list of output parameter
        - normalize: bool -> whether data is normalized
        - scaler: str -> defines scaler
        - get_mean: bool -> whether calculate mean of e.g. timesteps
        - is_transient: bool -> whether data is transient (and filtering is required)
        - lower_bound: int -> lower bound of time steps to keep
        - upper_bound: int -> upper bound of time steps to keep
        
    Returns:
        X_df: dataFrame -> preprocessed input data
        y_df: dataFrame -> preprocessed output data
    """

    # Keeps only instances with values between lower and upper bound
    if is_transient:
        # Filter for transient data based on time steps.
        column_name = "Time Step"
        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

        df.to_csv(output_path + "/test_after_prep.csv", index=False)
    
    # Normalizing and scaling data:
    if normalize: df = normalize_data(df)
    if scaler != 'none': df = scale_data(df, scaler)
    
    if get_mean: 
        data_df = mean_of_timesteps(df, input_parameter)
    else:
        data_df = df
        
    data_df.to_csv(output_path + "/reduced_data.csv")
    
    # split df into input and output
    X_df = data_df[input_parameter]
    y_df = data_df[output_parameter]

    return X_df, y_df

def preprocessing(da=False, **kwargs):
    '''
    New ,refactored preprocessing function combined with the read_data().
    Reads the data and returns both input and output data as dataframes.
    Args:
        - df: dataFrame -> contains data
        - da: bool -> whether data analysis is running
        - kwargs: dict -> additional parameters for preprocessing
    Returns:
        - X_df: dataFrame -> preprocessed input data
        - y_df: dataFrame -> preprocessed output data
    '''
    
    data_path = f"input_data/" + kwargs['data_path']
    output_path = './output_data/' + kwargs['output_name'] + '/'

    if ".xlsx" in data_path: data_df_all = pd.read_excel(data_path) # xlsx files
    if ".csv" in data_path: data_df_all = pd.read_csv(data_path) # csv files
    else: data_df_all = read_ansys_output_to_dfs(data_path, da=da) # Ansys Output Files
    
    # Saving data
    if kwargs['save_data'].lower() != 'false':
        # Check directory and creating directory if necessary
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data_df_all.to_csv(output_path + "/" + kwargs['save_data'], index=False)
    
    df = data_df_all.copy()
    
    lower_bound = kwargs.get('lower_bound', None)
    upper_bound = kwargs.get('upper_bound', None)

    if kwargs['is_transient']:
        # Filter for transient data based on time steps.
        column_name = "Time Step"
        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
        df.to_csv(kwargs['output_name'] + "/test_after_prep.csv", index=False)
        
    if kwargs['normalize']:
        df = normalize_data(df)
        
    if kwargs['scaler'] != 'none':
        df = scale_data(df, kwargs['scaler'])
    
    if kwargs['get_mean']:
        data_df = mean_of_timesteps(df, kwargs['input_parameter'])
    else:
        data_df = df
        
    data_df.to_csv(kwargs['output_name'] + "/reduced_data.csv")
    
    X_df = data_df[kwargs['input_parameter']]
    y_df = data_df[kwargs['output_parameter']]

    return X_df, y_df