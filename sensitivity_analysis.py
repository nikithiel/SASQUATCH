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

        # from input_bounds get num_vars, names, bounds
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

def sensitivity_analysis_bounds(X_df, y_df, models, sa_input_bounds, sample_size, input_metric, sa_input_initial_dict, model):
    """Performs sensitivity analysis with varying the bounds.
    
    Args:
        - X_df: dataFrame -> contains input data
        - y_df: dataFrame -> contains ouput data
        - models: dict -> contains models and their names
        - input_bounds: dict -> contains min and max values for parameter bounds
        - sample_size: int -> sample size (e.g. 2^9)
        - input-metric: string -> defines type of bound-variation
        - sa_input_initial_dict: dict -> contains the initial parameter configuration
        - model: string -> defines the model for the output
    Returns:
        - uncertainty_Y_dict: dict -> contains uncertainty for all uncertainty values
        - uncertainty_sobol_dict: dict -> contains the sobol indices results for all uncertainty values
        - sa_Y_variation_dict: dict -> contains the sa results for all uncertainty values
    """
    sa_input_bound_dict = {}
    sa_Y_variation_dict = {}
    uncertainty_Y_dict = {}
    uncertainty_sobol_dict = {}
    uncertainty_metric = [np.arange(0.005, 0.04, 0.00684), np.arange(0.1, 1.26, 1*0.23), np.arange(0.07, 1, 0.225)] #
    metric = 0
    if input_metric == 'percentages': metric = 0
    elif input_metric == 'millimeter': metric = 1
    elif input_metric == 'maximum': metric = 2

    for uncertainty in uncertainty_metric[metric]:
        print("   Uncertainty:", uncertainty)
        sa_input_bound_dict[uncertainty] = {}
        for param_name, param_val in sa_input_initial_dict.items():
            sa_input_bound_dict[uncertainty][param_name] = {}
            if input_metric == 'millimeter':
                if param_name == 'alpha':
                    min_value = -np.arcsin(uncertainty/np.sqrt((2*3.6)**2+uncertainty**2))*180/np.pi
                    max_value = -min_value
                elif param_name == 'R':
                    min_value = param_val - uncertainty/2
                    max_value = param_val + uncertainty/2
                else:
                    min_value = param_val - uncertainty
                    max_value = param_val + uncertainty
            elif input_metric == 'percentages':
                if param_name == 'alpha':
                    min_value = 0
                    max_value = 0
                else:
                    min_value = param_val - (param_val * uncertainty)
                    max_value = param_val + (param_val * uncertainty)
            elif input_metric == 'maximum':
                if param_name == 'alpha':
                    min_value = -10*uncertainty
                    max_value = -min_value
                elif param_name == 'R':
                    min_value = param_val - uncertainty
                    max_value = param_val + uncertainty
                else:
                    min_value = param_val - 2*uncertainty
                    max_value = param_val + 2*uncertainty

            sa_input_bound_dict[uncertainty][param_name]['min'] = min(min_value, max_value)
            sa_input_bound_dict[uncertainty][param_name]['max'] = max(min_value, max_value)
            
        sa_input_bounds = sa_input_bound_dict[uncertainty]
        sa_results, input_parameter_list, X_dict , Y_dict = sensitivity_analysis(X_df, y_df, models, sa_input_bounds, sample_size=sample_size)
        if input_metric=='percentages' or input_metric=='maximum': uncertainty=uncertainty*100
        uncertainty_Y_dict[uncertainty] = Y_dict
        uncertainty_sobol_dict[uncertainty] = sa_results
        sa_Y_variation_dict[uncertainty] = print_dict_stats(Y_dict, model=model)

    return uncertainty_Y_dict, uncertainty_sobol_dict, sa_Y_variation_dict