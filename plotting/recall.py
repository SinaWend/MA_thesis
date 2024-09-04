import re
import matplotlib.pyplot as plt

def extract_data(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    # Split content into experiments based on the start marker
    experiments = content.split('Experiment start at:')
    data = []
    
    for experiment in experiments[1:]:  # Skip the first split as it's before the first experiment
        training_recall = []
        validation_recall = []
        test_recall = []
        epochs = []
        
        # Extract binary_recall values for each domain and epoch
        for match in re.finditer(r'epoch: (\d+).*?binary_recall': (\d+\.\d+),.*?Validation:.*?binary_recall': (\d+\.\d+),.*?Test Domain.*?binary_recall': (\d+\.\d+),', experiment, re.S):
            epoch, train_rec, val_rec, test_rec = match.groups()
            epochs.append(int(epoch))
            training_recall.append(float(train_rec))
            validation_recall.append(float(val_rec))
            test_recall.append(float(test_rec))
        
        data.append((epochs, training_recall, validation_recall, test_recall))
    
    return data

def plot_data(data):
    fig, axes = plt.subplots(len(data), 1, figsize=(10, 5 * len(data)))
    
    if len(data) == 1:
        axes = [axes]  # Make single plot iterable
    
    for i, (epochs, training_recall, validation_recall, test_recall) in enumerate(data):
        axes[i].plot(epochs, training_recall, label='Training Recall', marker='o')
        axes[i].plot(epochs, validation_recall, label='Validation Recall', marker='o')
        axes[i].plot(epochs, test_recall, label='Test Recall', marker='o')
        axes[i].set_title(f'Experiment {i + 1}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Binary Recall')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

# Usage
filename = '/home/aih/sina.wendrich/MA_thesis/DomainLab/zoutput/benchmarks/CAMELYON_center0_dinov2small_erm_dial_irm_lr1e5_bs16_classbalancing10_allfreeze/run_experiment-index=0-21556221.err'
data = extract_data(filename)
plot_data(data)
