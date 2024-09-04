import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import textwrap
import os

# Define the paths and filenames
plots = "best"
if plots == "best":
    file_info = [
        {'path': '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/BLOOD_acevedo_dinov2small_erm_irm_dial_lr1e5_bs16_classbalancing10_nofreeze/', 'filename': 'results.csv', 'label': 'No_freezing_test'},
        # {'path': '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_irm_dial_lr1e5_bs16_classbalancing10_nofreeze/', 'filename': 'bestvalepoch_validation_results.csv', 'label': 'No_freezing_val'},
        # {'path': '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_irm_dial_lr1e5_bs16_classbalancing10_halffreeze_blocks/', 'filename': 'results.csv', 'label': 'Half_blocks_frozen_test'},
        # {'path': '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_irm_dial_lr1e5_bs16_classbalancing10_halffreeze_blocks/', 'filename': 'bestvalepoch_validation_results.csv', 'label': 'Half_blocks_frozen_val'},
        # {'path': '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_dial_irm_lr1e5_bs16_classbalancing10_allfreeze/', 'filename': 'results.csv', 'label': 'Only_train_classificationlayer_test'},
        # {'path': '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_dial_irm_lr1e5_bs16_classbalancing10_allfreeze/', 'filename': 'bestvalepoch_validation_results.csv', 'label': 'Only_train_classificationlayer_val'}
    ]

# Read hyperparameters file (assuming all folders have the same hyperparameters.csv file)
hyperparameters_path = file_info[0]['path'] + 'hyperparameters.csv'
hyperparameters = pd.read_csv(hyperparameters_path, index_col=0)

# Read the data from the files
dataframes = []

for info in file_info:
    df = pd.read_csv(info['path'] + info['filename'], index_col=False, skipinitialspace=True)
    df['params'] = df['params'].apply(literal_eval)
    df['source'] = info['label']
    dataframes.append(df)

# Combine the data from all files
data = pd.concat(dataframes, axis=0)

# Strip leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Ensure param_index is treated as a categorical variable and sort it
data['param_index'] = pd.Categorical(data['param_index'].astype(str).str.strip(), ordered=True)
data.sort_values('param_index', inplace=True)

# Define the metric to be plotted
metric = 'acc'  # Change metric to accuracy

# Create a color palette based on param_index values
def create_palette(param_indices):
    palette = {}
    for index in param_indices:
        if index == '0':
            palette[index] = 'C0'
        elif index in ['1', '2', '3', '4', '5']:
            palette[index] = 'C1'
        elif index in ['6', '7', '8']:
            palette[index] = 'C2'
    return palette

palette = create_palette(data['param_index'].unique())

# Define function to create FacetGrid for specified indices
def create_facetgrid(filtered_data, title, filename_suffix, col_wrap=4, top_adjust=0.9):
    g = sns.FacetGrid(filtered_data, col='param_index', col_wrap=col_wrap, height=4, aspect=1.5, sharey=True, col_order=filtered_data['param_index'].unique())
    
    # Map boxplot with custom color palette based on param_index
    for param_index, ax in zip(g.col_names, g.axes.flat):
        sns.boxplot(x='source', y=metric, data=filtered_data[filtered_data['param_index'] == param_index], ax=ax, color=palette[param_index], order=[info['label'] for info in file_info])
        
        # Update titles with hyperparameters
        if param_index.isnumeric():
            hyperparam_info = hyperparameters.loc[int(param_index), ['method', 'params']].to_string(header=False, index=False)
        else:
            hyperparam_info = "Hyperparameters not found"
        # Wrap long parameter strings
        hyperparam_info = '\n'.join(textwrap.wrap(hyperparam_info, width=50))
        ax.set_title(hyperparam_info, fontsize=12)  # Increase title font size

    g.set_axis_labels('Source', metric.capitalize(), fontsize=14)  # Increase axis labels font size
    g.set(ylim=(0, 1))
    plt.subplots_adjust(top=top_adjust, hspace=0.4, wspace=0.3)
    # Rotate x-axis labels to avoid overlap and increase font size
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(12)  # Increase x-axis labels font size
        ax.set_xlabel('')  # Remove 'Source' below

    # Add super title for the whole plot
    if plots == "best":
        g.fig.suptitle('Results of best validation accuracy epoch', fontsize=18)  # Increase super title font size
    elif plots == "last":
        g.fig.suptitle('Results of last epoch', fontsize=18)  # Increase super title font size
    elif plots == "best_onlytest":
        g.fig.suptitle('Test results of best epoch', fontsize=18)  # Increase super title font size
    else:
        g.fig.suptitle('Results of last and best validation accuracy epoch', fontsize=18)  # Increase super title font size

    # Save the plot in the respective directories
    for info in file_info:
        plot_path = os.path.join(info['path'], f"{plots}_{filename_suffix}.png")
        g.savefig(plot_path)
    plt.show()
    plt.close()

# Create plot for all param_index values
create_facetgrid(data, 'Comparison of Accuracy Across All Param Indices and Sources', 'combined_result_all')

# Filter data for param_index 0-5 and create plot with 3 columns per row
indices_0_5 = data[data['param_index'].isin([str(i) for i in range(6)])]
create_facetgrid(indices_0_5, 'Param Indices 0-5', 'subset_0_5', col_wrap=3)

# Filter data for param_index 0 and 6-8 and create plot with adjusted top parameter
indices_0_6_8 = data[data['param_index'].isin(['0'] + [str(i) for i in range(6, 9)])]
create_facetgrid(indices_0_6_8, 'Param Indices 0 and 6-8', 'subset_0_6_8', col_wrap=4, top_adjust=0.78)
