import os
import glob

def combine_results(directory, result_type):
    combined_results = ["param_index, method, mname, commit, algo, epos, te_d, seed, params, acc, precision, recall, specificity, f1, auroc, binary_precision, binary_recall, binary_specificity, binary_f1_score, acc_oracle, acc_val, model_selection_epoch, experiment_duration\n"]

    # Iterate over all results files in the directory
    for file_path in glob.glob(os.path.join(directory, f'{result_type}_results_*.csv')):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            header = lines[0]  # Skip the header
            last_epoch_lines = []
            current_seed = None

            for line in lines[1:]:
                columns = line.split(',')
                seed = columns[7].strip()

                # If the seed changes, save the last epoch line of the previous seed
                if current_seed is None:
                    current_seed = seed

                if seed != current_seed:
                    combined_results.append(last_epoch_lines[-1])  # Add last epoch line of the previous seed
                    last_epoch_lines = []
                    current_seed = seed

                last_epoch_lines.append(line)

            # Append the last line of the last seed processed
            if last_epoch_lines:
                combined_results.append(last_epoch_lines[-1])

    # Write combined results to a single file
    combined_output_path = os.path.join(directory, f'lastepoch_{result_type}_results.csv')
    with open(combined_output_path, 'w') as f:
        f.writelines(combined_results)

# Directory containing results files
log_dir_path = '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_dial_irm_lr1e5_bs16_classbalancing10_allfreeze/'

# Combine results for train, validation, and test
combine_results(log_dir_path, 'train')
combine_results(log_dir_path, 'validation')
combine_results(log_dir_path, 'test')
