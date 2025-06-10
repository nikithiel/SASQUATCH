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
import matplotlib.pyplot as plt
from surrogate_model_comparison import kFold_Evaluation, train_and_save_models
from plotting import plot_densities
from plotting import show_plots, plot_sa_results_heatmap
from plotting import plot_sa_results_17_segments
from plotting import plot_feature_scatterplot
from plotting import bounds_variation_plot, bounds_mean_std, bounds_sobol
from plotting import plot_data_analysis, plot_surrogate_model_comparison, plot_sensitivity_analysis
from sensitivity_analysis import sensitivity_analysis, sensitivity_analysis_bounds

import os
import pickle
import time
import json
from copy import deepcopy

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
    print("Analyzing Data\n------------")
    X_df, y_df = preprocessing(da=True, **parameter)
    
    try: # Path where the file is saved
        output_path = './output_data/' + parameter['output_name'] + '/' +'data_analysis' + '/'
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")

    # Plotting default plots
    print("Plotting Data\n------------")
    plot_data_analysis(X_df, y_df, output_path, **parameter)
    plt.show() if showplot else None
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

    # Determine the path for the specific file and the folder to save the file
    try: # Path where the plots are saved
        output_data = './output_data/' + parameter['output_name'] + '/' + 'surrogate_model' + '/'
        output_path = output_data + 'plots/'
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")

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
        print("  Plotting surrogate model comparison result")
        plot_surrogate_model_comparison(smc_results=smc_results, X_df=X_df, y_df=y_df, output_path=output_path, models=models, kwargs=parameter)
        show_plots() if showplot else None
        
        print("  Plotting of smc results: Done")
        print("Surrogate Model Comparison: Done")

elif run_type == 'sa':
    print("Sensitivity Analysis")

    # ----- Initialize path to save the results ----- #
    try: # Path where the plot and output is saved
        output_file = './output_data/' + parameter['output_name'] + '/sensitivity_analysis/ST_results/'
        output_path = './output_data/' + parameter['output_name'] + '/sensitivity_analysis/plots/'

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
    X_df, y_df = get_bounded_data(**parameter) # Acquiring the filtered dataset from the reduced dataset.
    sa_results, input_parameter_list, X_dict , Y_dict = sensitivity_analysis(X_df, y_df, models, input_bounds, parameter['sa_sample_size'])
    sa_results_for_plot = deepcopy(sa_results)
    print("  Perform SA: Done")

    # ----- Saving SA Results ----- #
    if input('   Do you want to save the sensitivity analysis results(y/n):' ) == 'y':
        # Create the path if it doesn't exist yet
        os.makedirs(os.path.dirname(output_file), exist_ok = True)

        # Save only the ST sensitivity matrix
        for key in sa_results.keys():
            for key2 in sa_results[key].keys():
                sa_results[key][key2] = sa_results[key][key2]['ST']

        # Save the file as csv
        for key in sa_results.keys():
            filenamecsv = output_file + "/sa_array_" + key + ".csv"
            pd.DataFrame(sa_results[key]).to_csv(filenamecsv)
            for key2 in sa_results[key].keys():
                sa_results[key][key2] = sa_results[key][key2].tolist() #  Convert the results to a list for dumping the json file      

        # Saving the results in a json file
        for key in sa_results.keys():
            filename = output_file + "sa_results_" + key + ".json"
            with open(filename, "w") as sa_file:
                json.dump(sa_results[key], sa_file, separators=(',',':'), indent=4)
        print("  Saving sensitivity analysis results: Done")
    else:
        print("   Saving sensitivity analysis results: Skipped")

    # ----- Plotting Results ----- #
    Y_dict['Training Data'] = y_df.values
    plot_sensitivity_analysis(X_dict, Y_dict, output_path, sa_results_for_plot, model_names, input_parameter_list, parameter)
    plt.show() if showplot else None
    print("  Plotting of sa results: Done")

    print("Sensitivity Analysis: Done")

# Sensitivity analysis bounds or uncertainty quantification
elif run_type == 'uq':
    print("Uncertainty Quantification")
    # ----- Initialize Hyperparameter ----- #
    try:
        # Path where the file is saved
        output_data = './output_data/' + parameter['output_name'] + '/' + 'uncertainty_quantification/'
        output_plots = output_data           
    except Exception:
        print("!!! Error: Parameter in config.txt file missing !!!")
    
    # ----- DATA Preprocessing ----- #
    X_df, y_df = preprocessing(da=False, **parameter)
    sa_input_initial_dict = {'y': -5, 'z': 51, 'alpha': 0, 'R': 3.6} # Starting point
    metric = 'maximum' # millimeter, percentages, maximum 
    if metric == 'percentages':  
        sa_input_initial_dict.pop('alpha')
        X_df.pop('alpha')
    print("  Data Preprocessing: Done")

    # ----- Creating models ----- #
    input_bounds = get_data_bounds(X_df)
    models, model_names = creatingModels(input_bounds, parameter)
    print("  Creating Models: Done : ",model_names)

    # ----- Getting bounded data ----- #
    X_df, y_df = get_bounded_data(**parameter) # Acquiring the filtered dataset from the reduced dataset
    
    # ----- Sensitivity Analysis ----- #  
    the_model = parameter['sa_17_segment_model'].replace('_',' ')      
    uncertainty_Y_dict, uncertainty_sobol_dict, sa_Y_variation_dict = sensitivity_analysis_bounds(X_df=X_df, y_df=y_df, models=models, sa_input_bounds=input_bounds, sample_size=parameter['sa_sample_size'], input_metric=metric, sa_input_initial_dict=sa_input_initial_dict, model=the_model)
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

    bounds_variation_plot(sa_Y_variation_dict, parameter['sa_output_parameter'], output_plots, is_title=False, title=title+"_small_"+the_model, x_annot=x_annot, y_annot=y_annot, legend=True)

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

    bounds_sobol(uncertainty_sobol_dict, output_plots, parameter['input_parameter_label'], dict(zip(parameter['output_parameter'], parameter['output_parameter_label'])), model_name = the_model, sobol_index='ST', 
                 fig_size=(16.2/2.54, 21.5/2.54), font_size=10, is_title=False, title=title+"_Bounds_sobol_allinone", x_annot=x_annot, y_annot="Sensitivity")
    
    bounds_mean_std(uncertainty_Y_dict, output_plots, output_parameters=to_plot_dict[to_plot]['numbers'], output_names=to_plot_dict[to_plot]['names'], \
                    model=the_model, is_title=False, title=title+"_bounds_std_region_allinone_nolegend_"+str(metric)+"_"+to_plot, x_annot=x_annot, y_annot="Output Value", \
                        all_in_one=True, annotation='legend', figsize=(3.2,9))
    for configuration in to_plot_dict.keys():
        bounds_mean_std(uncertainty_Y_dict, output_plots, output_parameters=to_plot_dict[configuration]['numbers'], output_names=to_plot_dict[configuration]['names'], \
                        model=the_model, is_title=False, title=title+"_bounds_std_region_allinone_"+str(metric)+"_"+configuration, x_annot=x_annot, y_annot="Output Value", \
                            all_in_one=True, annotation='pstd', figsize=(3.2,3))
    show_plots() if showplot else None 
    
    print("  Plotting of uq results: Done")

    print("Uncertainty Quantification: Done")

else:
    print('Unknown Input: ', run_type)