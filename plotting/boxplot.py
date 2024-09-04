import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval  # literal_eval can safely evaluate Python expressions

# Define the paths
path1 = '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_dial_irm_lr1e5_bs16_classbalancing10_allfreeze/'
path2 = '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_irm_dial_lr1e5_bs16_classbalancing10_nofreeze/'

# Read the data from the files
file1 = pd.read_csv(path1 + 'results.csv', index_col=False, converters={"params": literal_eval}, skipinitialspace=True)
file2 = pd.read_csv(path2 + 'results.csv', index_col=False, converters={"params": literal_eval}, skipinitialspace=True)

# Add a column to each dataframe to label the source file
file1['source'] = 'Allfreeze'
file2['source'] = 'Nofreeze'

# Combine the data from both files
data = pd.concat([file1, file2], axis=0)

# Strip leading and trailing spaces from column names
data.columns = data.columns.str.strip()

# Ensure param_index is treated as a categorical variable and sort it
data['param_index'] = pd.Categorical(data['param_index'].astype(str).str.strip(), ordered=True)
data.sort_values('param_index', inplace=True)

# Define the metric to be plotted
metric = 'binary_recall'

# Define function to create FacetGrid for specified indices
def create_facetgrid(filtered_data, title, filename_suffix, col_wrap=4):
    g = sns.FacetGrid(filtered_data, col='param_index', col_wrap=col_wrap, height=4, aspect=1.5, sharey=True, col_order=filtered_data['param_index'].unique())
    g.map(sns.boxplot, 'source', metric, palette='Set2', order=['Allfreeze', 'Nofreeze'])
    g.set_titles(col_template='Param Index: {col_name}')
    g.set_axis_labels('Source', metric.capitalize())
    g.set(ylim=(0, 1))
    plt.subplots_adjust(top=0.9, hspace=0.3, wspace=0.3)
    g.fig.suptitle(title, fontsize=16)
    g.savefig(path1 + filename_suffix + '.png')
    g.savefig(path2 + filename_suffix + '.png')
    plt.show()
    plt.close()

# Create plot for all param_index values
create_facetgrid(data, 'Comparison of Binary Recall Across All Param Indices and Sources', 'combined_result_all')

# Filter data for param_index 0-5 and create plot with 3 columns per row
indices_0_5 = data[data['param_index'].isin([str(i) for i in range(6)])]
create_facetgrid(indices_0_5, 'Param Indices 0-5', 'subset_0_5', col_wrap=3)

# Filter data for param_index 0 and 6-8 and create plot
indices_0_6_8 = data[data['param_index'].isin(['0'] + [str(i) for i in range(6, 9)])]
create_facetgrid(indices_0_6_8, 'Param Indices 0 and 6-8', 'subset_0_6_8', col_wrap=4)
