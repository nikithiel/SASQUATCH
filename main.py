"""
SASQUATCH
------

This program can be used to perform sensitiviy analysis and uncertainty quantification.

For small datasets you can predict unknown data with the help of surrogate models.
To choose a suitable surrogate model you can use 'surrogate model comparison'.
To check whether your data is being used correctly you can analyze your data with 'data analysis'.

Contributors: Joel Gestrich, Niklas Thiel, Michael Neidlin
Mail: joel.gestrich@rwth-aachen.de, niklas.thiel@rwth-aachen.de

"""

# ----- Imports ----- #
import warnings
import pandas as pd
from initialization import read_user_defined_parameters, get_data_bounds
from preprocessing import preprocessing, get_bounded_data
from models import creatingModels
from surrogate_model_comparison import kFold_Evaluation, train_and_save_models
from plotting import plot_smc_timings, plot_smc_r2score_and_errors, plot_data, plot_densities
from plotting import surrogate_model_predicted_vs_actual, show_plots, plot_sa_results_heatmap
from plotting import plot_sa_results_17_segments, plot_boxplots, plot_correlation
from plotting import plot_feature_distribution, plot_feature_scatterplot, actual_scatter_plot
from plotting import bounds_variation_plot, bounds_mean_std, bounds_sobol
from sensitivity_analysis import sensitivity_analysis, sensitivity_analysis_perturbation

import os
import pickle
import time
import collections.abc

# ----- Program Information ----- #
print("\n===============================================")
print("| Surrogate Modeling and Sensitivity Analysis |")
print("===============================================\n")
print("! Check Parameters in config.txt file !\n")
print()

warnings.filterwarnings("ignore") # ignores warnings

# Reading user defined hyper parameter
parameter = read_user_defined_parameters('configMVUQ.txt')
showplot = parameter['plot_data']

run_type = parameter['run_type']

if run_type == 'da':
    print("Analyze Data\n------------")
    X_df, y_df = preprocessing(da=True, **parameter)
    output_plots = './output_data/' + parameter['output_name'] + '/' +'data_analysis' + '/'

    # Plotting default plots
    print("Plotting Data\n------------")
    plot_correlation(X_df, y_df, output_plots)
    plot_feature_distribution(y_df, output_plots, dict(zip(parameter['output_parameter'], parameter['output_units'])), num_bins=20, title="Distribution of Output Parameter")
    actual_scatter_plot(X_df, y_df, output_plots)
    #show_plots() if input(" Display plots?: (y/n) ") == 'y' else None

    # Optional plots
    plot_boxplots(X_df, y_df, output_plots, title="Boxplots of Ouput Parameter") if input(" Would you like to plot the boxplot as well?: (y/n)") == 'y' else None
    plot_data(X_df, y_df, output_plots) if input(" Would you like to plot the detailed scatterplot as well?: (y/n)") == 'y' else None
    show_plots() if showplot else None
    print("Analyze Data: Done")

elif run_type == 'su' or run_type == 'sc':
    print("Surrogate Model Training and Comparison")

    # ----- DATA Preprocessing ----- #
    X_df, y_df = preprocessing(da=False, **parameter)
    print("  Data Preprocessing: Done")

    # ----- Creating models ----- #
    input_bounds = get_data_bounds(X_df)
    models, model_names = creatingModels(input_bounds, parameter)
    print("  Creating Models: Done")

    #determine the path for the specific file and the folder to save the file
    output_data = './output_data/' + parameter['output_name'] + '/' + 'surrogate_model' + '/'
    output_plots = output_data + 'Plots/'

    # ----- Training and Testing of Surrogate Models ----- #
    smc_results = kFold_Evaluation(X=X_df, y=y_df, models=models, parameter = parameter)
    train_and_save_models(X_df, y_df, models, output_data + '/trainedModels/')
    print("  Training and Testing: Done")
    
    #checkpoint here to ask if user wants to save the models or not
    print("Do you want to compare the models? (y/n): ")
    if input().lower() != 'y':
        print("  Surrogate Model Comparison: Skipped")
        exit()
    else:
        print("  Performing Surrogate Model Comparison")

    # ----- Saving Surrogate Model Comparison Results ----- #
    smc_results.to_csv(os.path.join(output_data, 'smc_results.csv'), index=False)
    print("  Saving of smc results: Done")
    print("Surrogate Model Training: Done")

    if parameter['plot_data']:
        # ----- Plotting Results ----- #
        # Plotting default plot
        plot_smc_r2score_and_errors(smc_results, parameter['output_parameter'], output_plots, metrics=parameter['metrics'], average_output=True, is_title=False, title="Model Errors and Scores avrg r2 mape")

        # Optional plots
        plot_smc_r2score_and_errors(smc_results, parameter['output_parameter'], output_plots, metrics=parameter['metrics'], average_output=False, is_title=False, title="Model Errors and Scores", rmse_log_scale=True) if input(" Would you like to plot detailed r2 score?: (y/n) ") == 'y' else None
        surrogate_model_predicted_vs_actual(models, X_df, y_df, output_plots, parameter['output_parameter'], dict(zip(parameter['output_parameter'], parameter['output_units']))) if input(" Would you like to plot the model comparison as a scatter plot?: (y/n) ") == 'y' else None
        plot_smc_timings(smc_results, output_plots, is_title=False) if input(" Would you like to plot the surrogate model comparison timing plot?: (y/n) ") == 'y' else None
        show_plots() if showplot else None
        
        print("  Plotting of smc results: Done")
        print("Surrogate Model Comparison: Done")

elif run_type == 'sa':  
    print("Sensitivity Analysis")

    # ----- Initialize path to save the results ----- #
    try:
        # Creating the path to save the file
        output_data = './output_data/' + parameter['output_name'] + '/' + 'sensitivity_analysis' + '/'
        output_plots = output_data

    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")

    # ----- DATA Preprocessing ----- #
    X_df, y_df = preprocessing(da=False, **parameter)
    print("  Data Preprocessing: Done")

    # ----- Creating models ----- #
    input_bounds = get_data_bounds(X_df)
    models, model_names = creatingModels(input_bounds, parameter)
    print("  Creating Models: Done : ",model_names)
    
    # ----- Sensitivity Analysis ----- #
    X_df, y_df, _ = get_bounded_data(**parameter) # Acquiring the filtered dataset from the reduced dataset.
    sa_results, input_parameter_list, X_dict , Y_dict = sensitivity_analysis(X_df, y_df, models, input_bounds, parameter['sa_sample_size'])
    print("  Perform SA: Done")

    # ----- Saving SA Results ----- #
    if input('   Do you want to save the sensitivity analysis results(y/n):' ) == 'y':
        with open("sa_results.txt", 'a') as sa_out_file:
            sa_out_file.write(str(sa_results))
        print("  Saving sensitivity analysis results: Done")
    else:
        print("   Saving sensitivity analysis results: Skipped")

    # ----- Plotting Results ----- #
    Y_dict['Training Data'] = y_df.values
    plot_densities(X_dict, Y_dict, output_plots, labels=parameter['sa_output_parameter'], is_title=False, title="Density plot")
    values_list = []
    model_list = []
    for (model, values) in Y_dict.items():
        values_list.append(values)
        model_list.append(model)
    
    plot_17_segment = True if input (" Plot the 17 segment model as well? (y/n)") == 'y' else False
    plot_sa_results_heatmap(sa_results, model_names, input_parameter_list, parameter['output_parameter_sa_plot'], output_plots, parameter['sa_sobol_indice'])
    if plot_17_segment:
        plot_sa_results_17_segments(sa_results, input_parameter_list, output_plots, parameter['sa_17_segment_model'], parameter['sa_sobol_indice'])
    show_plots() if showplot else None
    print("  Plotting of sa results: Done")

    print("Sensitivity Analysis: Done")

# Sensitivity analysis bounds or uncertainty quantification
elif run_type == 'uq':
    print("Uncertainty Quantification")
    # ----- Initialize Hyperparameter ----- #
    try:
        
        input_parameter_label = parameter.get('input_parameter_label', parameter['input_parameter'])
        output_parameter = parameter['sa_output_parameter']
        output_parameter_label = parameter.get('output_parameter_label', parameter['sa_output_parameter'])

        output_data = './output_data/' + parameter['output_name'] + '/' + 'uncertainty_quantification/'
        output_plots = output_data
            
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")
    
    # ----- DATA Preprocessing ----- #
    X_df, y_df = preprocessing(da=False, **parameter)
    X_df, y_df, sa_input_dict = get_bounded_data(**parameter)
    print("  Data Preprocessing: Done")

    # ----- Creating models ----- #
    input_bounds = get_data_bounds(X_df)
    models, model_names = creatingModels(input_bounds, parameter)
    print("  Creating Models: Done : ",model_names)
    
    # ----- Checking Metrics ----- #
    output_parameter_list = [a[1:-1].split(",") for a in parameter["uq_output_parameter"]] # Acquires the grouped parameters
    output_parameter_labels = [a[1:-1].split(",") for a in parameter["uq_output_parameter_label"]] # Acquires the grouped parameters labels matching the parameters
    
    # ----- Making the labels function compatible ----- #
    output_parameter_units = []
    n = 0
    for n in range(len(output_parameter_list)):
        for i in range((len(output_parameter_list[n]))):
            output_parameter_units.append(parameter['uq_output_units'][n])
    output_parameter_list = [a for b in output_parameter_list for a in b]
    output_parameter_labels = [a for b in output_parameter_labels for a in b]
    
    # ----- Sensitivity Analysis ----- #    
    the_model = parameter['sa_17_segment_model'].replace("_"," ") 
    uncertainty_Y_dict, uncertainty_sobol_dict, sa_Y_variation_dict = sensitivity_analysis_perturbation(X_df=X_df, y_df=y_df, filtered_df = sa_input_dict, models=models, model=parameter['sa_17_segment_model'].replace('_',' '), sample_size=parameter['sa_sample_size'])
    print("  Perform SA: Done")

    # ----- Plotting Results ----- #
    x_annot="Scaled Input Variation in %"
    y_annot="Output Variation in %"
    title="Input Variation percentage"

    to_plot = 'some_segments'
    to_plot_dict = {}
    to_plot_dict['all_segments'] = {}
    to_plot_dict['some_segments'] = {}
    to_plot_dict['tvpg'] = {}
    to_plot_dict['E'] = {}
    to_plot_dict['Eloss'] = {}
    to_plot_dict['Ekin'] = {}
    to_plot_dict['custom'] = {}
    
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
    
    to_plot_dict['custom']['names'] = output_parameter_list
    to_plot_dict['custom']['numbers'] = [parameter['sa_output_parameter'].index(a) for a in output_parameter_list]
    
    bounds_variation_plot(sa_Y_variation_dict, output_parameter_list, output_plots, is_title=False, title=title+"_small_"+the_model, x_annot=x_annot, y_annot=y_annot, legend=True)

    bounds_sobol(uncertainty_sobol_dict, output_plots, input_parameter_label, dict(zip(output_parameter_list, output_parameter_labels)), model_name = the_model, sobol_index='ST', 
                 fig_size=(16.2/2.54, 21.5/2.54), font_size=10, is_title=False, title=title+"_Bounds_sobol_allinone", x_annot=x_annot, y_annot="Sensitivity")
    
    bounds_mean_std(uncertainty_Y_dict, output_plots, output_parameters=to_plot_dict[to_plot]['numbers'], output_names=to_plot_dict[to_plot]['names'], \
                    model=the_model, is_title=False, title=title+"_bounds std region allinone nolegend "+str(parameter['input_start'])+" "+to_plot, x_annot=x_annot, y_annot="Output Value", \
                        all_in_one=True, annotation='legend', figsize=(3.2,9))
    
    for configuration in to_plot_dict.keys():
        bounds_mean_std(uncertainty_Y_dict, output_plots, output_parameters=to_plot_dict[configuration]['numbers'], output_names=to_plot_dict[configuration]['names'], \
                        model=the_model, is_title=False, title=title+" bounds std region allinone_"+str(parameter['input_start'])+" "+configuration, x_annot=x_annot, y_annot="Output Value", \
                            all_in_one=True, annotation='pstd', figsize=(3.2,3))
    show_plots() if showplot else None 
    
    print("  Plotting of sa results: Done")

    print("Sensitivity Analysis Bounds: Done")

else:
    print('Unknown Input: ', run_type)