import pandas as pd
from scipy.stats import ttest_rel
from ast import literal_eval
from scipy.stats import wilcoxon

# Define the columns if they are not being correctly inferred
columns = ['param_index', 'method', 'mname', 'commit', 'algo', 'epos', 'te_d', 'seed', 'params', 'acc', 'precision', 'recall', 'specificity', 'f1', 'auroc', 'binary_precision', 'binary_recall', 'binary_specificity', 'binary_f1_score', 'acc_oracle', 'acc_val', 'model_selection_epoch', 'experiment_duration']

# Load your data
base_path = '/home/aih/sina.wendrich/MA_thesis/zoutput_balanced/benchmarks/CAMELYONbalanced_center0_dinov2small_irm_10seeds_nofreeze_2024-07-26_02-28-54'
df1 = pd.read_csv(f'{base_path}/rule_results/0.csv', index_col=False, converters={"params": literal_eval}, skipinitialspace=True, names=columns, header=0)
df2 = pd.read_csv(f'{base_path}/rule_results/1.csv', index_col=False, converters={"params": literal_eval}, skipinitialspace=True, names=columns, header=0)

# Assuming 'acc' is the column for accuracy and the rows are aligned by seed in both datasets
# t_statistic, p_value = ttest_rel(df2['binary_recall'], df1['binary_recall'])

# print("T-statistic:", t_statistic)
# print("P-value:", p_value)

# if p_value < 0.05:
#     if t_statistic > 0:
#         print("Dataset 2 is statistically significantly better than Dataset 1 in terms of accuracy.")
#     else:
#         print("Dataset 2 is statistically significantly worse than Dataset 1 in terms of accuracy.")
# else:
#     print("There is no statistically significant difference in accuracy between the two datasets.")

stat, p_value = wilcoxon(df2['binary_recall'], df1['binary_recall'])

print("Wilcoxon test statistic:", stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("There is a statistically significant difference in accuracy between the two datasets.")
else:
    print("There is no statistically significant difference in accuracy between the two datasets.")