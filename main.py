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
from initialization import read_user_defined_parameters, get_data_bounds
from preprocessing import preprocessing, get_bounded_data
from models import creatingModels
from surrogate_model_comparison import kFold_Evaluation, train_and_save_models
from plotting import plot_densities, show_plots, plot_sa_results_heatmap
from plotting import plot_sa_results_17_segments
from plotting import bounds_variation_plot, bounds_mean_std, bounds_sobol
from plotting import plot_data_analysis, plot_surrogate_model_comparison
from sensitivity_analysis import sensitivity_analysis, sensitivity_analysis_perturbation

import os

# ----- Program Information ----- #
print("\n===============================================")
print("| Surrogate Modeling and Sensitivity Analysis |")
print("===============================================\n")
print("! Check Parameters in config.txt file !\n")
print()

warnings.filterwarnings("ignore") # ignores warnings

# Reading user defined hyper parameter
parameter = read_user_defined_parameters('config.txt')
showplot = parameter['plot_data']

run_type = parameter['run_type']

if not parameter['output_name']:
    parameter['output_name'] = "default"

if run_type == 'da':
    print("Analyze Data\n------------")
    X_df, y_df = preprocessing(da=True, **parameter)
    output_plots = './output_data/' + parameter['output_name'] + '/' +'data_analysis' + '/'

    # Plotting default plots
    print("Plotting Data\n------------")
    plot_data_analysis(X_df, y_df, output_path=output_plots, **parameter)
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
        plot_surrogate_model_comparison(smc_results, X_df, y_df, output_path=output_plots, models=models, kwargs=parameter)
        show_plots() if showplot else None
        
        print("  Plotting of smc results: Done")
    print("Surrogate Model Comparison: Done")

elif run_type == 'sa':  
    print("Sensitivity Analysis")

    # ----- Initialize path to save the results ----- #
    try:
        # Creating the path to save the file
        output_plots = './output_data/' + parameter['output_name'] + '/' + 'sensitivity_analysis' + '/'

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
        with open(os.path.join(output_plots,"sa_results.txt"), 'a') as sa_out_file:
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
        plot_sa_results_17_segments(sa_results, input_parameter_list, output_plots, parameter['sa_17_segment_model'].replace("_"," "), parameter['sa_sobol_indice'])
    show_plots() if showplot else None
    print("  Plotting of sa results: Done")

    print("Sensitivity Analysis: Done")

# Sensitivity analysis bounds or uncertainty quantification
elif run_type == 'uq':
    print("Uncertainty Quantification")
    # ----- Initialize Hyperparameter ----- #
    try:
        
        input_parameter_label = parameter.get('input_parameter_label', parameter['input_parameter'])
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
    
    # ----- Sensitivity Analysis ----- #    
    the_model = parameter['sa_17_segment_model'].replace("_"," ") 
    uncertainty_Y_dict, uncertainty_sobol_dict, sa_Y_variation_dict = sensitivity_analysis_perturbation(X_df=X_df, y_df=y_df, filtered_df = sa_input_dict, models=models, model=parameter['sa_17_segment_model'].replace('_',' '), sample_size=parameter['sa_sample_size'])
    print("  Perform SA: Done")

    # ----- Plotting Results ----- #
    x_annot="Scaled Input Variation in %"
    y_annot="Output Variation in %"
    title="Input Variation Percentage"

    # ----- Ploting metrics ----- #
    if not parameter["uq_output_parameter"]:
        output_parameter_list = parameter["sa_output_parameter"]
        output_parameter_labels = parameter.get('output_parameter_label', parameter['sa_output_parameter'])
        output_parameter_units = None
    else:
        output_parameter_list = [a[1:-1].split(",") for a in parameter["uq_output_parameter"]] # Acquires the grouped parameters
        output_parameter_labels = [a[1:-1].split(",") for a in parameter["uq_output_parameter_label"]] # Acquires the grouped parameters labels matching the parameters
        output_parameter_units = parameter['uq_output_units']

        output_parameter_combined_units = []
        n = 0
        for n in range(len(output_parameter_list)):
            for i in range((len(output_parameter_list[n]))):
                output_parameter_combined_units.append(parameter['uq_output_units'][n])
        output_parameter_combined_list = [a for b in output_parameter_list for a in b]
        output_parameter_combined_labels = [a for b in output_parameter_labels for a in b]

    # Plot all the variations for parameters in all the groups combined
    
    
    if parameter["uq_output_parameter"]:
        bounds_variation_plot(sa_Y_variation_dict, output_parameter_combined_list, output_parameter_combined_labels, output_plots, is_title=False, title=title+" for model "+the_model, x_annot=x_annot, y_annot=y_annot, legend=True)

        for i in range(len(output_parameter_list)):
            
            input_custom_label = [i + " (" + str(j) + " %)" for i,j in zip(input_parameter_label,parameter['input_start_perturbation'])] if len(parameter['input_start_perturbation']) != 1 else [i + " (" + str(parameter['input_start_perturbation']) + "%)" for i in input_parameter_label]

            bounds_sobol(uncertainty_sobol_dict, output_plots, input_custom_label, dict(zip(output_parameter_list[i], output_parameter_labels[i])), model_name = the_model, sobol_index='ST', 
                        fig_size=(21.5/2.54, 21.5/2.54), font_size=10, is_title=False, title=title + " Bounds in sobol for group " + str(output_parameter_list[i]), x_annot=x_annot, y_annot="Sensitivity")

            y_annot = "Output Value in " + output_parameter_units[i] if output_parameter_units else "Output Value"
            bounds_mean_std(uncertainty_Y_dict, output_plots, output_parameters=[parameter['sa_output_parameter'].index(a) for a in output_parameter_list[i]], output_names=output_parameter_list[i], \
                            model=the_model, is_title=False, title=title+" bounds with std using start point "+ str(parameter['input_start'])+" for group "+ str(output_parameter_list[i]), x_annot=x_annot, y_annot=y_annot, \
                                all_in_one=True, annotation='legend', figsize=(3.2,9))
    else:
        bounds_variation_plot(sa_Y_variation_dict, output_parameter_list, output_parameter_labels, output_plots, is_title=False, title=title+" on all parameters for model "+the_model, x_annot=x_annot, y_annot=y_annot, legend=True)

        input_custom_label = [i + " (" + str(j) + " %)" for i,j in zip(input_parameter_label,parameter['input_start_perturbation'])] if len(parameter['input_start_perturbation']) != 1 else [i + " (" + str(parameter['input_start_perturbation']) + "%)" for i in input_parameter_label]

        bounds_sobol(uncertainty_sobol_dict, output_plots, input_custom_label, dict(zip(output_parameter_list, output_parameter_labels)), model_name = the_model, sobol_index='ST', 
                    fig_size=(21.5/2.54, 21.5/2.54), font_size=10, is_title=False, title=title + " Bounds in sobol for all parameters", x_annot=x_annot, y_annot="Sensitivity")

        y_annot = "Output Value in " + output_parameter_units[i] if output_parameter_units else "Output Value"
        bounds_mean_std(uncertainty_Y_dict, output_plots, output_parameters=[parameter['sa_output_parameter'].index(a) for a in output_parameter_list], output_names=output_parameter_list, \
                        model=the_model, is_title=False, title=title+" bounds with std using start point "+ str(parameter['input_start'])+" for all parameters", x_annot=x_annot, y_annot=y_annot, \
                            all_in_one=True, annotation='legend', figsize=(3.2,9))

    show_plots() if showplot else None 
    
    print("  Plotting of sa results: Done")

    print("Sensitivity Analysis Bounds: Done")

else:
    print('Unknown Input: ', run_type)