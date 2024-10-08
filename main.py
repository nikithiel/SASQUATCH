"""
DASUSA
------

This program can be used to perform sensitiviy analysis and uncertainty quantification.

For small datasets you can predict unknown data with the help of surrogate models.
To choose a suitable surrogate model you can use 'surrogate model comparison'.
To check whether your data is being used correctly you can analyze your data with 'data analysis'.

Author: Joel Gestrich
Mail: joel.gestrich@rwth-aachen.de
Date: 23-12-20 to 24-02-
Project: Bachelor Thesis
"""

# ----- Imports ----- #
import warnings
import pandas as pd
from initialization import read_user_defined_parameters, read_data, get_data_bounds
from preprocessing import preprocessing
from models import creatingModels
from surrogate_model_comparision import kFold_Evaluation
from plotting import plot_smc_timings, plot_smc_r2score_and_errors, plot_data
from plotting import surrogate_model_predicted_vs_actual, show_plots, plot_sa_results_heatmap
from plotting import plot_sa_results_17_segments, plot_boxplots, plot_correlation
from plotting import plot_feature_distribution, plot_feature_scatterplot, plot_densitys
from plotting import bounds_variation_plot, bounds_mean_std, bounds_sobol
from sensitivity_analysis import sensitivity_analysis, sensitivity_analysis_bounds

# ----- Program Information ----- #
print("\n===============================================")
print("| Surrogate Modeling and Sensitivity Analysis |")
print("===============================================\n")
print("! Check Parameters in config.txt file !\n")
user_input = input("Enter one of the following:\n  [da] Data Analysis\n  [su] Surrogate Model Comparison\n  [sa] Sensitivity Analysis\n  [uq] Uncertainty Quantification\n\n [ps] Project Specific\n\n ")
print()

# Reading user defined hyper parameter
parameter = read_user_defined_parameters('config.txt')
warnings.filterwarnings("ignore") # ignores warnings

if user_input == 'da':
    print("Analyze Data\n------------")
    # ----- Initialize Hyperparameter ----- #
    try:
        data_path = parameter['data_path']
        input_parameter = parameter['input_parameter']
        output_parameter = parameter['output_parameter']
        save_data = parameter['save_data']
        normalize = parameter['normalize']
        scaler = parameter['scaler']
        get_mean = parameter['get_mean_of_each_input_pair']
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")
    
    df = read_data(data_path, da=True, save_data=save_data)
    X_df, y_df = preprocessing(df=df, input_parameter=input_parameter, output_parameter=output_parameter, normalize=normalize, scaler=scaler, get_mean=get_mean)
    plot_correlation(X_df, y_df)
    show_plots()
    plot_boxplots(X_df, y_df, title="Boxplots of Ouput Parameter")
    show_plots()
    plot_feature_distribution(y_df, num_bins=20, title="Distribution of Output Parameter")
    show_plots()
    plot_data(X_df, y_df)
    show_plots()
    print("Analyze Data: Done")

elif user_input == 'su' or user_input == 'sc':
    print("Surrogate Model Comparison")

    # ----- Initialize Hyperparameter ----- #
    try:
        data_path = parameter['data_path']
        input_parameter = parameter['input_parameter']
        output_parameter = parameter['output_parameter']
        model_names = parameter['models']
        n_splits = parameter['n_splits']
        shuffle = parameter['shuffle']
        random_state = parameter['random_state']
        is_plot_data = parameter['plot_data']
        save_data = parameter['save_data']
        normalize = parameter['normalize']
        scaler = parameter['scaler']
        get_mean = parameter['get_mean_of_each_input_pair']
        metrics = parameter['metrics']
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")
    
    # ----- DATA Preprocessing ----- #
    df = read_data(data_path, da=True, save_data=save_data)
    X_df, y_df = preprocessing(df=df, input_parameter=input_parameter, output_parameter=output_parameter, normalize=normalize, scaler=scaler, get_mean=get_mean)
    print("  Data Preprocessing: Done")
    
    # ----- Creating models ----- #
    input_bounds = get_data_bounds(X_df)
    models, model_names = creatingModels(model_names, input_bounds)
    print("  Creating Models: Done")

    # ----- Training and Testing of Surrogate Models ----- #
    smc_results = kFold_Evaluation(X=X_df, y=y_df, n_splits=n_splits, shuffle=shuffle, random_state=random_state, models=models)
    print("  Training and Testing: Done")

    # ----- Saving Surrogate Model Comparison Results ----- #
    smc_results.to_csv('smc_results.csv', index=False)
    print("  Saving of smc results: Done")

    # ----- Plotting Results ----- #
    surrogate_model_predicted_vs_actual(models, X_df, y_df)
    plot_smc_timings(smc_results, is_title=False)
    plot_smc_r2score_and_errors(smc_results, output_parameter, metrics=metrics, average_output=True, is_title=False, title="Model Errors and Scores avrg r2 mape")
    plot_smc_r2score_and_errors(smc_results, output_parameter, metrics=metrics, average_output=False, is_title=False, title="Model Errors and Scores", rmse_log_scale=True)
    show_plots()
    
    print("  Plotting of smc results: Done")
    print("Surrogate Model Comparison: Done")

elif user_input == 'sa':
    print("Sensitivity Analysis")

    # ----- Initialize Hyperparameter ----- #
    try:
        data_path = parameter['data_path']
        input_parameter = parameter['input_parameter']
        output_parameter = parameter['sa_output_parameter']
        model_names_input = parameter['sa_models']
        output_parameter_sa_plot = parameter['output_parameter_sa_plot']
        sa_17_segment_model = parameter['sa_17_segment_model']
        sa_sobol_indice = parameter['sa_sobol_indice']
        save_data = parameter['save_data']
        normalize = parameter['normalize']
        scaler = parameter['scaler']
        get_mean = parameter['get_mean_of_each_input_pair']
        sample_size = parameter['sa_sample_size']
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")
    

    # ----- DATA Preprocessing ----- #
    df = read_data(data_path, da=True, save_data=save_data)
    X_df, y_df = preprocessing(df=df, input_parameter=input_parameter, output_parameter=output_parameter, normalize=normalize, scaler=scaler, get_mean=get_mean)
    print("  Data Preprocessing: Done")

    # ----- Creating models ----- #
    input_bounds = get_data_bounds(X_df)
    models, model_names = creatingModels(model_names_input, input_bounds)
    print("  Creating Models: Done : ",model_names)
    
    # ----- Sensitivity Analysis ----- #
    #sa_input_bounds = {'y': {'min': -7, 'max': -3}, 'z': {'min': 49., 'max': 53.}, 'alpha': {'min': -10, 'max': 10}, 'R': {'min': 2.6, 'max': 4.6}}
    sa_results, input_parameter_list, X_dict , Y_dict = sensitivity_analysis(X_df, y_df, models, input_bounds, sample_size=sample_size)
    print("  Perform SA: Done")

    # ----- Saving SA Results ----- #
    with open("sa_results.txt", 'w') as sa_out_file:
        sa_out_file.write(str(sa_results))
    print("  Saving of sa results: Done")

    # ----- Plotting Results ----- #
    Y_dict['Training Data'] = y_df.values
    plot_densitys(X_dict, Y_dict, lables=output_parameter, is_title=False, title="Density plot")
    values_list = []
    model_list = []
    for (model, values) in Y_dict.items():
        values_list.append(values)
        model_list.append(model)
    plot_sa_results_heatmap(sa_results, model_names, input_parameter_list, output_parameter_sa_plot, sa_sobol_indice)
    plot_sa_results_17_segments(sa_results, input_parameter_list, "NIPCE", sa_sobol_indice)
    show_plots()
    print("  Plotting of sa results: Done")

    print("Sensitivity Analysis: Done")

elif user_input == 'ps':
    print("Project Specific")
    try:
        data_path = parameter['data_path']
        input_parameter = parameter['input_parameter']
        output_parameter = parameter['output_parameter']
        save_data = parameter['save_data']
        normalize = parameter['normalize']
        scaler = parameter['scaler']
        get_mean = parameter['get_mean_of_each_input_pair']
        model_names = parameter['models']
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")
    
    df = read_data(data_path, da=True, save_data=save_data)
    X_df, y_df = preprocessing(df=df, input_parameter=input_parameter, output_parameter=output_parameter, normalize=normalize, scaler=scaler, get_mean=get_mean)
        
    input_bounds = get_data_bounds(X_df)
    models, model_names = creatingModels(model_names, input_bounds)
    print("  Creating Models: Done : ",model_names)
    print(y_df.columns)
    plot_feature_distribution(y_df[['WSS', 'Eloss', 'Ekin', 'TVPG']], num_bins=10, is_title=False, title="Output_Distribution", num_subplots_in_row=4, figure_size='small')
    show_plots()
    surrogate_model_predicted_vs_actual(models, X_df, y_df, is_title=False, title="Actual_vs_Predicted")
    show_plots()
    plot_feature_scatterplot(pd.concat([X_df, y_df], axis=1), ['y','z','alpha','R'], ['WSS', 'Eloss', 'Ekin', 'TVPG'], is_title=False, title="Scatterplot Input Output")
    show_plots()

# Sensitivity analysisy bounds or uncertainty quantification#
elif user_input == 'uq':
    print("Uncertainty Quantification")
    # ----- Initialize Hyperparameter ----- #
    try:
        data_path = parameter['data_path']
        input_parameter = parameter['input_parameter']
        output_parameter = parameter['sa_output_parameter']
        model_names_input = parameter['sa_models']
        output_parameter_sa_plot = parameter['output_parameter_sa_plot']
        sa_17_segment_model = parameter['sa_17_segment_model']
        sa_sobol_indice = parameter['sa_sobol_indice']
        save_data = parameter['save_data']
        normalize = parameter['normalize']
        scaler = parameter['scaler']
        get_mean = parameter['get_mean_of_each_input_pair']
        sample_size = parameter['sa_sample_size']
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")
    
    # ----- DATA Preprocessing ----- #
    df = read_data(data_path, da=True, save_data=save_data)
    X_df, y_df = preprocessing(df=df, input_parameter=input_parameter, output_parameter=output_parameter, normalize=normalize, scaler=scaler, get_mean=get_mean)
    sa_input_initial_dict = {'y': -5, 'z': 51, 'alpha': 0, 'R': 3.6} #'alpha': 0,
    metric = 'maximum' # millimeter, percentages, maximum 
    if metric == 'percentages':  
        sa_input_initial_dict.pop('alpha')
        X_df.pop('alpha')
    print("  Data Preprocessing: Done")

    # ----- Creating models ----- #
    input_bounds = get_data_bounds(X_df)
    models, model_names = creatingModels(model_names_input, input_bounds)
    print("  Creating Models: Done : ",model_names)
    
    # ----- Sensitivity Analysis ----- #  
    the_model = 'NIPCE'      
    uncertainty_Y_dict, uncertainty_sobol_dict, sa_Y_variation_dict = sensitivity_analysis_bounds(X_df=X_df, y_df=y_df, models=models, sa_input_bounds=input_bounds, sample_size=sample_size, input_metric=metric, sa_input_initial_dict=sa_input_initial_dict, model=the_model)
    print("  Perform SA: Done")

    # ----- Plotting Results ----- #
    if metric == 'percentages':
        x_annot="Input Variation in %"
        y_annot="Output Variation in %"
        title="Input Variation percentage"
    elif metric == 'millimeter':
        x_annot="Input Variation in mm"
        y_annot="Output Variation in %"
        title="Input Variation mm"
    elif metric == 'maximum':
        x_annot="Input Variation in %"
        y_annot="Output Variation in %"
        title="Input Variation perc of input param range"

    bounds_variation_plot(sa_Y_variation_dict, output_parameter, is_title=False, title=title+"_small_"+the_model, x_annot=x_annot, y_annot=y_annot, legend=False)

    to_plot = 'some_segments'
    to_plot_dict = {}
    to_plot_dict['all_segments'] = {}
    to_plot_dict['some_segments'] = {}
    to_plot_dict['tvpg'] = {}
    to_plot_dict['E'] = {}
    to_plot_dict['Eloss'] = {}
    to_plot_dict['Ekin'] = {}
    
    to_plot_dict['all_segments']['numbers'] = [0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    to_plot_dict['all_segments']['names'] = ['WSS','wss-1',	'wss-2', 'wss-3','wss-4','wss-5','wss-6','wss-7','wss-8','wss-9','wss-10','wss-11','wss-12','wss-13','wss-14','wss-15','wss-16','wss-17']

    to_plot_dict['some_segments']['numbers'] = [4,5,6,8,10,11,17,19,20]
    to_plot_dict['some_segments']['names'] = ['wss-1', 'wss-2', 'wss-3','wss-5','wss-7','wss-8','wss-14','wss-16','wss-17']

    to_plot_dict['tvpg']['numbers'] = [3]
    to_plot_dict['tvpg']['names'] = ['TVPG']

    to_plot_dict['E']['numbers'] = [1,2]
    to_plot_dict['E']['names'] = ['Eloss', 'Ekin']
    
    to_plot_dict['Eloss']['numbers'] = [1]
    to_plot_dict['Eloss']['names'] = ['Eloss']
    
    to_plot_dict['Ekin']['numbers'] = [2]
    to_plot_dict['Ekin']['names'] = ['Ekin']

    bounds_sobol(uncertainty_sobol_dict, model_name = the_model, sobol_index='ST', is_title=False, title=title+"_Bounds_sobol_allinone", x_annot=x_annot, y_annot="Sensitivity")
    
    bounds_mean_std(uncertainty_Y_dict, output_parameters=to_plot_dict[to_plot]['numbers'], output_names=to_plot_dict[to_plot]['names'], model=the_model, is_title=False, title=title+"_bounds_std_region_allinone_nolegend_"+str(metric)+"_"+to_plot, x_annot=x_annot, y_annot="Output Value", all_in_one=True, annotation='legend', figsize=(3.2,9))
    for configuration in to_plot_dict.keys():
        bounds_mean_std(uncertainty_Y_dict, output_parameters=to_plot_dict[configuration]['numbers'], output_names=to_plot_dict[configuration]['names'], model=the_model, is_title=False, title=title+"_bounds_std_region_allinone_"+str(metric)+"_"+configuration, x_annot=x_annot, y_annot="Output Value", all_in_one=True, annotation='pstd', figsize=(3.2,3))
    show_plots()
    
    print("  Plotting of sa results: Done")

    print("Sensitivity Analysis Bounds: Done")

else:
    print('Unknown Input: ', user_input)
    
