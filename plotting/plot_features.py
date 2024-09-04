import numpy as np
import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def load_and_concatenate_features(out_dir, numbers):
    features_list = []
    labels_list = []
    domains_list = []
    slurm_job_id = None

    for number in numbers:
        for dataset_type in ['train', 'val', 'test']:
            feature_filename = [f for f in os.listdir(os.path.join(out_dir, 'features')) if f'features_{dataset_type}' in f and number in f]
            label_filename = [f for f in os.listdir(os.path.join(out_dir, 'features')) if f'labels_{dataset_type}' in f and number in f]
            domain_filename = [f for f in os.listdir(os.path.join(out_dir, 'features')) if f'domains_{dataset_type}' in f and number in f]

            if feature_filename and label_filename and domain_filename:
                features = np.load(os.path.join(out_dir, 'features', feature_filename[0]))
                labels = np.load(os.path.join(out_dir, 'features', label_filename[0]))
                domains = np.load(os.path.join(out_dir, 'features', domain_filename[0]))

                if labels.ndim > 1 and labels.shape[1] > 1:
                    labels = np.argmax(labels, axis=1)

                features_list.append(features)
                labels_list.append(labels)
                domains_list.append(domains)

                # Extract slurm job ID from filename
                if slurm_job_id is None:
                    slurm_job_id = feature_filename[0].split('_')[-1].split('.')[0]

                # Print domain distribution for each dataset type
                print(f"Domain distribution for {dataset_type} with number {number}: {np.unique(domains, return_counts=True)}")

    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_domains = np.concatenate(domains_list, axis=0)

    # Print combined domain distribution
    print(f"Combined domain distribution: {np.unique(all_domains, return_counts=True)}")

    return all_features, all_labels, all_domains, slurm_job_id

def create_legend(scatter, unique_items, title):
    from matplotlib.lines import Line2D
    colors = [scatter.cmap(scatter.norm(item)) for item in unique_items]
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=str(item))
                       for item, color in zip(unique_items, colors)]
    return legend_elements

def find_index_from_error_file(out_dir, number, new_structure=False):
    if new_structure:
        error_dir = os.path.join(out_dir, 'slurm_logs', 'run_experiment')
    else:
        error_dir = out_dir
        
    error_files = [f for f in os.listdir(error_dir) if f.endswith('.err') and number in f]
    if error_files:
        index = error_files[0].split('-')[1].split('=')[1]
        return index
    return None

def plot_combined(out_dir, numbers, new_structure=False):
    features, labels, domains, slurm_job_id = load_and_concatenate_features(out_dir, numbers)
    index = find_index_from_error_file(out_dir, numbers[0], new_structure)

    if index is None:
        index = 'unknown'

    # Print shapes for debugging
    print(f"Combined Features shape: {features.shape}")
    print(f"Combined Labels shape: {labels.shape}")
    print(f"Combined Domains shape: {domains.shape}")

    # Create UMAP plot
    umap_results = umap.UMAP(n_components=2).fit_transform(features)

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('UMAP visualization (Labels)')
    unique_labels = np.unique(labels)
    legend_elements = create_legend(scatter, unique_labels, "Labels")
    plt.legend(handles=legend_elements, title="Labels")

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=domains, cmap='Set1', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('UMAP visualization (Domains)')
    unique_domains = np.unique(domains)
    legend_elements = create_legend(scatter, unique_domains, "Domains")
    plt.legend(handles=legend_elements, title="Domains")

    plt.savefig(os.path.join(out_dir, f'UMAP_plot_combined_{index}_{slurm_job_id}.png'))
    plt.close()

    # Create t-SNE plot
    tsne_results = TSNE(n_components=2, perplexity=min(30, len(features) - 1)).fit_transform(features)

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization (Labels)')
    unique_labels = np.unique(labels)
    legend_elements = create_legend(scatter, unique_labels, "Labels")
    plt.legend(handles=legend_elements, title="Labels")

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=domains, cmap='Set1', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization (Domains)')
    unique_domains = np.unique(domains)
    legend_elements = create_legend(scatter, unique_domains, "Domains")
    plt.legend(handles=legend_elements, title="Domains")

    plt.savefig(os.path.join(out_dir, f't-SNE_plot_combined_{index}_{slurm_job_id}.png'))
    plt.close()

def load_and_plot(out_dir, dataset_type, numbers, new_structure=False):
    for number in numbers:
        feature_filename = [f for f in os.listdir(os.path.join(out_dir, 'features')) if f'features_{dataset_type}' in f and number in f]
        label_filename = [f for f in os.listdir(os.path.join(out_dir, 'features')) if f'labels_{dataset_type}' in f and number in f]
        domain_filename = [f for f in os.listdir(os.path.join(out_dir, 'features')) if f'domains_{dataset_type}' in f and number in f]

        if feature_filename and label_filename and domain_filename:
            slurm_job_id = feature_filename[0].split('_')[-1].split('.')[0]
            index = find_index_from_error_file(out_dir, number, new_structure)

            if index is None:
                index = 'unknown'

            features = np.load(os.path.join(out_dir, 'features', feature_filename[0]))
            labels = np.load(os.path.join(out_dir, 'features', label_filename[0]))
            domains = np.load(os.path.join(out_dir, 'features', domain_filename[0]))

            # Print shapes for debugging
            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Domains shape: {domains.shape}")

            # Ensure labels are single integers (not one-hot encoded)
            if (labels.ndim > 1 and labels.shape[1] > 1):
                labels = np.argmax(labels, axis=1)

            # Print shapes after processing
            print(f"Processed Labels shape: {labels.shape}")

            # Plot UMAP
            umap_results = umap.UMAP(n_components=2).fit_transform(features)
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
            plt.colorbar(scatter)
            plt.title(f'UMAP visualization ({dataset_type} - Number {number})')
            unique_labels = np.unique(labels)
            legend_elements = create_legend(scatter, unique_labels, "Labels")
            plt.legend(handles=legend_elements, title="Labels")
            plt.savefig(os.path.join(out_dir, f'UMAP_plot_{dataset_type}_{index}_{slurm_job_id}.png'))
            plt.close()

            # Plot t-SNE
            tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1))
            tsne_results = tsne.fit_transform(features)
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
            plt.colorbar(scatter)
            plt.title(f't-SNE visualization ({dataset_type} - Number {number})')
            unique_labels = np.unique(labels)
            legend_elements = create_legend(scatter, unique_labels, "Labels")
            plt.legend(handles=legend_elements, title="Labels")
            plt.savefig(os.path.join(out_dir, f't-SNE_plot_{dataset_type}_{index}_{slurm_job_id}.png'))
            plt.close()

out_dir = '/home/aih/sina.wendrich/MA_thesis/zoutput_blood/benchmarks/BLOOD_acevedo_dinov2small_vit_resnet_erm_features_2024-08-13_02-20-58'
#numbers = ['24076675', '24076680', '24076683', '24078373', '24076668', '24076671', '24076673', '24076676', '24076678', '24076681', '24076684', '24076666', '24076669', '24076672', '24076674', '24076677', '24076679', '24076682', '24076685','24076667', '24076670']
numbers = ['24132708', '24132713', '24132716', '24132725', '24132701', '24132704', '24132706', '24141961', '24141962', '24141958', '24141963', '24141959', '24141964', '24141960', '24132707', '24132710', '24132712', '24132715', '24132718','24132700', '24132703']
numbers = ['24116079', '24116076', '24116077', '24116078']
numbers = ['24148166', '24148164', '24148165']

new_structure = True

# for dataset_type in ['train', 'val', 'test']:
#     load_and_plot(out_dir, dataset_type, numbers, new_structure)

# Plot combined data
plot_combined(out_dir, numbers, new_structure)
