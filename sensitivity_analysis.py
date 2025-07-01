from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np
from plotting import print_dict_stats

def sensitivity_analysis(X_df, y_df, models, input_bounds, sample_size):
    """Performs sensitivity analysis.
    
    Args:
        - X_df: dataFrame -> contains input data
        - y_df: dataFrame -> contains ouput data
        - models: dict -> contains models and their names
        - input_bounds: dict -> contains min and max values for parameter bounds
        - sample_size: int -> sample size (e.g. 2^9)
    Returns:
        - sobolindices_dict: dict -> contains sobol indices information
        - input_bounds: list -> contains input parameter names
        - X_dict: dict -> contains input parameter sets for all models
        - Y_dict: dict -> contains output sets for all models
    """
    sobol_indices_dict = {}
    X_dict = {}
    Y_dict = {}
    for model_name, model in models.items():
        trained_model = MultiOutputRegressor(model).fit(X_df,y_df)

        # From input_bounds get num_vars, names, bounds
        num_vars = len(input_bounds)
        names = list(input_bounds.keys())
        bounds = [(input_bounds[key]['min'], input_bounds[key]['max']) for key in input_bounds]
        problem = {
            'num_vars': num_vars,
            'names': names,
            'bounds': bounds
        }
        X_sa = saltelli.sample(problem, sample_size)
        Y = trained_model.predict(X_sa)
        X_dict[model_name] = X_sa
        Y_dict[model_name] = Y
        #plot_feature_distribution_ndarray(Y,num_bins=20)
        sobol_indices_dict_by_output = {}
        for output_param, output_name in enumerate(y_df.columns.tolist()):
            sobol_indices_dict_by_output[output_name] = sobol.analyze(problem, Y[:,output_param], calc_second_order=True, print_to_console=False)
        sobol_indices_dict[model_name] = sobol_indices_dict_by_output
        print("    SA for ",model_name,": Done")

    return sobol_indices_dict, input_bounds.keys(), X_dict, Y_dict

def sensitivity_analysis_perturbation(X_df, y_df, filtered_df, models, model, sample_size, uncertainty_metrics = np.linspace(10,100,10)):
    """
    Performs sensitivity analysis on bounded data using different bounds variations.

    Args:
        - X_df: Bounded input data
        - y_df: Bounded output data
        - filtered_df: The starting point, upper perturbed bound and lower perturbed bound of the dataset
        - models: Models used for the training
        - model: The model for plotting?
        - sample_size: The sample size during sampling for sensitivity analysis
        - uncertainty_metrics: How would the perturbations be scaled. e.g: 10 20 30 40 ... 100.

    Returns:
        - uncertainty_Y_dict: Dictionary containing the output data uncertainty based on each perturbation
        - uncertainty_sobol_dict: Dictionary containing the sobol outputs for each perturbation
        - sa_Y_variation_dict: Dictionary containing the output variation during sensitivity analysis
    """
    perturbed_bounds = {}
    uncertainty_Y_dict = {}
    uncertainty_sobol_dict = {}
    sa_Y_variation_dict = {}
    
    for uncertainty in uncertainty_metrics:
        perturbed_bounds[uncertainty] = {}
        for param in X_df.keys():
            perturbed_bounds[uncertainty][param] = {}
            perturbed_bounds[uncertainty][param]['max'] = filtered_df.loc['start'][param] + ((filtered_df.loc['upper'][param] - filtered_df.loc['start'][param]) * (uncertainty/100))
            perturbed_bounds[uncertainty][param]['min'] = filtered_df.loc['start'][param] - ((filtered_df.loc['start'][param] - filtered_df.loc['lower'][param]) * (uncertainty/100))
        
        print(" Starting SA with scaled perturbation at " + str(uncertainty) + " percent")
        sa_results, _, _, Y_dict = sensitivity_analysis(X_df, y_df, models, perturbed_bounds[uncertainty], sample_size= sample_size)
        
        uncertainty_Y_dict[uncertainty] = Y_dict
        uncertainty_sobol_dict[uncertainty] = sa_results
        sa_Y_variation_dict[uncertainty] = print_dict_stats(Y_dict, model=model)
        
    return uncertainty_Y_dict, uncertainty_sobol_dict, sa_Y_variation_dict