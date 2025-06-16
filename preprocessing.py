import os
import collections.abc
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler
from plotting import plot_boxplots

def read_ansys_output_to_dfs(data_path, da=False):
    """Reading the .out files from ansys
    
    Args:
        - data_path : str -> path to input data / folder (ansys).
        - da : bool -> whether data analysis is running.
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

def preprocessing(da=False, **kwargs):
    '''
    New ,refactored preprocessing function combined with the read_data().
    Reads the data and returns both input and output data as dataframes.
    Args:
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
    if kwargs['save_data']:
        # Check directory and creating directory if necessary
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data_df_all.to_csv( output_path + "/saved_data.csv", index=False)
    
    df = data_df_all.copy()
    
    # Set the upper and lower bound based on the parameter data.
    lower_bound = kwargs.get('lower_bound', None)
    upper_bound = kwargs.get('upper_bound', None)

    if kwargs['is_transient']:
        # Filter for transient data based on time steps.
        column_name = "Time Step"
        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
        
        if kwargs['normalize']:
            df = normalize_data(df)
            
        if kwargs['scaler'] != 'none':
            df = scale_data(df, kwargs['scaler'])
        
        df = mean_of_timesteps(df, kwargs['input_parameter'])
    
    df.to_csv(output_path + "/reduced_data.csv")

    X_df = df[kwargs['input_parameter']]
    y_df = df[kwargs['output_parameter']]

    return X_df, y_df

def get_bounded_data(**kwargs):
    """ Function that gets the bounded data based on given start point and perturbation.
    
    Args:
        kwargs: dict -> contains the following keys from the config file:
            - input_start: str -> type of start point ('avg', 'median', 'start')
            - input_start_perturbation: float -> perturbation percentage
            - input_start_point: list -> custom start point values (optional)
            - output_name: str -> name of the output folder
            - input_parameter: list -> list of input parameters to filter by
            - sa_output_parameter: list -> list of sensitivity analysis output parameters to filter by

    Returns:
        X_df, y_df: dataFrame, dataFrame -> filtered input and output data based on bounds
    """
    # Read all the parameters for the bounded data
    startType = kwargs.get('input_start', 'avg')
    start = kwargs.get('input_start_point', None)
    
    # Reads the perturbation percentage for the bounds
    perturbation = kwargs.get('input_start_perturbation', 50)
    if not isinstance(perturbation, collections.abc.Sequence): # If only 1 pertrubation value is given, it will be used for all values
        perturbation = [perturbation] * len(kwargs['input_parameter'])
        perturbation = dict(zip(kwargs['input_parameter'], perturbation))
    elif len(perturbation) == len(kwargs['input_parameter']): # If the number of perturbation is not the exact same, return an error
        perturbation = dict(zip(kwargs['input_parameter'], perturbation))
    else:
        print(' Incorrect perturbation value given. Either input only 1 value or the exact number as input value')
        exit(0)
    
    # Reads the reduced data acquired from preprocessing and gets the bounds
    output_path = './output_data/' + kwargs['output_name'] + '/'
    dfpath = output_path + "/reduced_data.csv"
    df = pd.read_csv(dfpath)
    minmax = df.agg(['min', 'max'])
    
    if startType not in ['avg', 'median', 'start']:
        print(f" Invalid starting point type found! Will default to average starting point.")
        startType = 'avg' # Will always default to average starting point
    
    if startType == 'avg':
        # Calculates the average and set bounds based on perturbation
        print( "   Preparing filtered data using average starting point and perturbation values of: " , perturbation)
        tempdf = df.copy()
        tempdf.loc['start'] = df.mean(axis=0)
        tempdf.loc['upper'] = df.mean(axis=0)
        tempdf.loc['lower'] = df.mean(axis=0)
        for a in kwargs['input_parameter']:
            tempdf.loc['upper',a] = ((minmax.iloc[1][a]-tempdf.loc['start'][a]) * (1 + (perturbation[a] / 100)) ) + tempdf.loc['start'][a]
            tempdf.loc['lower',a] =  tempdf.loc['start'][a] - ((tempdf.loc['start'][a]-minmax.iloc[0][a]) * (1 - (perturbation[a] / 100)))
    
    if startType == 'median':
        # Calculates the median and set bounds based on perturbation
        print( f"   Preparing filtered data using median starting point and perturbation values of: " , perturbation)
        tempdf = df.copy()
        tempdf.loc['start'] = df.median(axis=0)
        for a in kwargs['input_parameter']:
            tempdf.loc['upper',a] = (minmax.iloc[1][a] - tempdf.loc['start'][a])*(1 + (perturbation[a] / 100)) + tempdf.loc['start'][a]
            tempdf.loc['lower',a] = tempdf.loc['start'][a] - (tempdf.loc['start'][a]-minmax.iloc[0][a])*(1 - (perturbation[a] / 100))
        
    if startType == 'start':
        # Uses a custom start point and sets bounds based on perturbation
        if start is None:
            print("   No start point(s) provided. Please provide a valid start point(s).")
            exit(0)
        print( f"   Preparing filtered data using custom starting point of" , start , " , and perturbation values of: " , perturbation)
        tempdf = df.copy()
        tempdf.loc['start'] = start
        for a in kwargs['input_parameter']:
            tempdf.loc['upper',a] = (minmax.iloc[1][a] - tempdf.loc['start'][a])*(1 + (perturbation[a] / 100)) + tempdf.loc['start'][a]
            tempdf.loc['lower',a] = tempdf.loc['start'][a] - (tempdf.loc['start'][a]-minmax.iloc[0][a])*(1 - (perturbation[a] / 100))

    finaldf = df.copy() # The dataframe to be returned
    for a in kwargs['input_parameter']:
        # Filter the dataset for every output parameter bounds
        finaldf = df[df[a].between(tempdf[a]['lower'], tempdf[a]['upper'])]
        if finaldf.empty:
            print(f"   No matching data found for the selected start point. Consider reselecting the start point or increase perturbation value.")
            exit(0)
    
    print(   f" Data filtering complete with remaining data containing" , finaldf.shape[0] , "points from" , df.shape[0] , "data points")
    finaldf.to_csv(output_path + '/reduced_filtered_data.csv') # Saves the filtered reduced data for potential future use
    # returns the filtered input, filtered output. Returns alongside that the starting point, upper bound and lower bounds of each dataset after perturbation.
    return finaldf[kwargs['input_parameter']], finaldf[kwargs['sa_output_parameter']], tempdf