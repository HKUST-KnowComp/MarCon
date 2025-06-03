import itertools
from scipy.stats import spearmanr, pearsonr
import json
# import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
BASE = "your base path here"  # Replace with your actual base path

def plot_marker_confidence_heatmap(data, model_name, top_n=4, redefined_markers=None):
    model_data = data.get(model_name, {})
    if not model_data:
        print(f"No data for model: {model_name}")
        return

    datasets = list(model_data.keys())
    
    common_markers = None
    for dataset, markers in model_data.items():
        if not isinstance(markers, dict):
            print(f"Warning: Data for dataset {dataset} is not in expected format, skipping.")
            continue
        current_markers = set(markers.keys())
        if common_markers is None:
            common_markers = current_markers
        else:
            common_markers = common_markers.intersection(current_markers)
    
    if not common_markers:
        print("No common markers found across all datasets.")
        return
    
    if redefined_markers is not None:
        common_markers = common_markers.intersection(set(redefined_markers))
        
    marker_confidence = {marker.lower(): {} for marker in common_markers}  
    for dataset, markers in model_data.items():
        for marker in common_markers:
            marker_lower = marker.lower()  
            if marker in markers and isinstance(markers[marker], dict):
                marker_confidence[marker_lower][dataset] = markers[marker].get("marker_correct_ratio", None)
    
    confidence_df = pd.DataFrame(marker_confidence).T
    confidence_df = confidence_df[datasets]
    
    if confidence_df.isnull().values.any():
        print("Warning: Some missing marker_correct_ratio values found. They will be dropped.")
        confidence_df = confidence_df.fillna(0)
    
    marker_std = confidence_df.std(axis=1)
    
    if top_n > len(marker_std):
        top_n = len(marker_std)
    top_markers = marker_std.nlargest(top_n).index
    top_confidence_df = confidence_df.loc[top_markers]
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    plt.figure(figsize=(10.5, 5.7))
    
    ax = sns.heatmap(top_confidence_df, annot=False, cmap="magma_r", vmin=0.4, vmax=1, cbar=True)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    y_labels = [marker.capitalize() for marker in top_confidence_df.index]
    ax.set_yticklabels(y_labels, rotation=0, fontsize=18)

    plt.xticks(rotation=45, fontsize=18)
    
    if model_name == 'gpt-4o':
        model_name = 'GPT-4o'
    plt.title(f"{model_name}", fontsize=22, fontstyle='italic')

    plt.tight_layout()
    plt.savefig("heat_map_sample_{}.pdf".format(model_name))
    plt.show()
    



def plot_marker_confidence_heatmaps(data, model_list, top_n=4, redefined_markers=None):
    # didn't use this currently, hard to adjust the size of the figure
    num_models = len(model_list)
    
    fig = plt.figure(figsize=(8 * num_models, 5))
    gs = gridspec.GridSpec(1, num_models + 1, width_ratios=[1] * num_models + [0.05])  # 最后一列用于颜色条
    axes = [fig.add_subplot(gs[i]) for i in range(num_models)]
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15

    heatmaps = []  
    for i, model_name in enumerate(model_list):
        model_data = data.get(model_name, {})
        if not model_data:
            print(f"No data for model: {model_name}")
            continue
        
        datasets = list(model_data.keys())
        
        common_markers = None
        for dataset, markers in model_data.items():
            if not isinstance(markers, dict):
                print(f"Warning: Data for dataset {dataset} is not in expected format, skipping.")
                continue
            current_markers = set(markers.keys())
            if common_markers is None:
                common_markers = current_markers
            else:
                common_markers = common_markers.intersection(current_markers)
        
        if not common_markers:
            print("No common markers found across all datasets.")
            continue
        
        if redefined_markers is not None:
            common_markers = common_markers.intersection(set(redefined_markers))
            
        marker_confidence = {marker.lower(): {} for marker in common_markers}
        for dataset, markers in model_data.items():
            for marker in common_markers:
                marker_lower = marker.lower()
                if marker in markers and isinstance(markers[marker], dict):
                    marker_confidence[marker_lower][dataset] = markers[marker].get("marker_correct_ratio", None)
        
        confidence_df = pd.DataFrame(marker_confidence).T
        confidence_df = confidence_df[datasets]
        
        if confidence_df.isnull().values.any():
            print("Warning: Some missing marker_correct_ratio values found. They will be dropped.")
            confidence_df = confidence_df.fillna(0)
        
        marker_std = confidence_df.std(axis=1)
        
        if top_n > len(marker_std):
            top_n = len(marker_std)
        top_markers = marker_std.nlargest(top_n).index
        top_confidence_df = confidence_df.loc[top_markers]
        
        ax = axes[i]
        heatmap = sns.heatmap(top_confidence_df, annot=False, cmap="magma_r", vmin=0.4, vmax=1, 
                               cbar=False, ax=ax)  

        y_labels = [marker.capitalize() for marker in top_confidence_df.index]
        ax.set_yticklabels(y_labels, rotation=0, fontsize=18)

        ax.set_xticklabels(top_confidence_df.columns, rotation=45, fontsize=18)
        
        ax.set_title(model_name, fontsize=22, fontstyle='italic')

    cbar_ax = fig.add_subplot(gs[-1])  
    cbar = fig.colorbar(heatmap.collections[0], cax=cbar_ax, pad=0.00001)
    cbar.set_label('Marker Confidence', fontsize=18)

    plt.subplots_adjust(wspace=0.001, right=0.3)  

    plt.tight_layout()
    plt.savefig("heat_map_samples.pdf")
    plt.show()
    

def plot_marker_rank_scatter(data, model_name, top_n, redefined_markers=None):
    # didn't use this currently, the effect is not good
    model_data = data.get(model_name, {})
    if not model_data:
        print(f"No data for model: {model_name}")
        return

    datasets = list(model_data.keys())

    common_markers = None
    for dataset, markers in model_data.items():
        if not isinstance(markers, dict):
            print(f"Warning: Data for dataset {dataset} is not in expected format, skipping.")
            continue
        current_markers = set(markers.keys())
        if common_markers is None:
            common_markers = current_markers
        else:
            common_markers = common_markers.intersection(current_markers)

    if not common_markers:
        print("No common markers found across all datasets.")
        return

    if redefined_markers is not None:
        common_markers = common_markers.intersection(set(redefined_markers))

    marker_rank_pct = {marker: {} for marker in common_markers}

    for dataset, markers in model_data.items():
        total_markers = len(markers)  
        sorted_markers = sorted(markers.items(),
                                key=lambda x: x[1].get("marker_correct_ratio", 0),
                                reverse=True)
        for rank, (marker, values) in enumerate(sorted_markers, start=1):
            if marker in common_markers:
                rank_percentage = (rank / total_markers) * 100
                marker_rank_pct[marker][dataset] = rank_percentage

    rank_pct_df = pd.DataFrame(marker_rank_pct).T
    rank_pct_df = rank_pct_df[datasets]

    marker_std = rank_pct_df.std(axis=1)

    if top_n > len(marker_std):
        top_n = len(marker_std)
    top_markers = marker_std.nlargest(top_n).index
    rank_pct_df = rank_pct_df.loc[top_markers]

    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(10, 8))
    for marker in rank_pct_df.index:
        plt.plot(rank_pct_df.columns, rank_pct_df.loc[marker], marker='o', linestyle='-', label=marker, alpha=0.8)

    plt.xlabel("Dataset", fontsize=15)
    plt.ylabel("Ranking Percentage (%)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper center', bbox_to_anchor=(0.50, 1.15), fontsize=17, markerscale=0.3)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("scatter_plot_{}.pdf".format(model_name))
    plt.show()

    


def calculate_model_marker_avgcv(all_marker_dic, marker_count):
    # C-AvgCV
    model_avgcv = {}
    model_max_deviation = {}
    for model, datasets in all_marker_dic.items():
        marker_correct_ratios = {}
        
        # Collect all correct ratios for each marker across datasets
        for dataset, markers in datasets.items():
            for marker, stats in markers.items():
                if marker not in marker_correct_ratios:
                    marker_correct_ratios[marker] = []
                marker_correct_ratios[marker].append(stats["marker_correct_ratio"])
        
        # Compute CV for each marker
        marker_cvs = []
        marker_deviations = {}
        for marker, ratios in marker_correct_ratios.items():
            if len(ratios) > 1:  # CV is undefined for single-value cases
                mean = np.mean(ratios)
                std = np.std(ratios, ddof=1)  # Use ddof=1 for sample standard deviation
                if mean != 0:
                    cv = std / mean
                    marker_cvs.append(cv)
                marker_deviations[marker] = np.max(np.abs(np.diff(ratios)))  # Compute max deviation as the largest difference between consecutive elements
                   
        
        # Compute AvgCV-1 for the model
        if marker_cvs:
            model_avgcv[model] = np.mean(marker_cvs)
        else:
            model_avgcv[model] = None  # Handle edge cases where no CV could be computed
        model_max_deviation[model] = marker_deviations
                
    with open("{}/rebuttal/C-AvgCV_thres={}.json".format(BASE, marker_count), 'w', encoding='utf-8') as f:
        json.dump(model_avgcv, f, indent=4, ensure_ascii=False)

def calculate_acc_cv_cv(all_cvs, all_marker_accs, marker_count):
    
    # appendix: correlation between I-AvgCV and acc on datasets
    
    all_dataset_cv = {}
    datasets = set()
    for model_data in all_cvs.values():
        datasets.update(model_data.keys())

    for dataset in datasets:
        cv_list = []
        acc_list = []
        
        for model in all_cvs:
            if dataset in all_cvs[model] and dataset in all_marker_accs.get(model, {}):
                cv_list.append(all_cvs[model][dataset])
                acc_list.append(all_marker_accs[model][dataset])
        
        if len(cv_list) > 1:
            corr, p_value = pearsonr(cv_list, acc_list)
            all_dataset_cv[dataset] = corr
            # print(f"{dataset}: Pearson correlation = {corr:.3f}, p-value = {p_value:.3f}")
        else:
            print(f"{dataset}: data point is not enough for correlation calculation.")
    
    print("Correlation between I-AvgCV and acc on datasets:")
    for dataset, corr in all_dataset_cv.items():
        print(f"{dataset}: Pearson correlation = {corr:.3f}")
        
            
    with open("{}/acc_cv_correlation_thres={}.json".format(BASE, marker_count), 'w', encoding='utf-8') as f:
        json.dump(all_dataset_cv, f, indent=4, ensure_ascii=False)

def plot_histogram_and_kde(all_marker_dic, model_name, dataset_name):
    correct_ratios = []
    counts = []

    for dataset, markers in all_marker_dic[model_name].items():
        if dataset == dataset_name:
            for marker, data in markers.items():
                correct_ratios.append(data["marker_correct_ratio"])
                counts.append(data["marker_count"])
    
    if len(correct_ratios) == 0 or len(counts) == 0:
        print(f"No data found for {model_name} - {dataset_name}!")
        return

    weighted_correct_ratios = np.repeat(correct_ratios, counts)

    plt.figure(figsize=(10, 5))
    plt.hist(weighted_correct_ratios, bins=20, alpha=0.6, color='g', label='Histogram')

    sns.kdeplot(weighted_correct_ratios, color='r', label='KDE Plot', fill=True, bw_adjust=0.5)

    plt.title(f"Distribution of Marker Correct Ratios for {model_name} - {dataset_name}")
    plt.xlabel("Correct Ratios")
    plt.ylabel("Density / Frequency")
    plt.legend()
    plt.show()

def calculate_concentration(all_marker_dic, marker_count):
    # I-AvgCV
    concentration_dic = {}
    all_cvs = {}
    marker_confidence_cvs = {}  
    for model, dataset_dict in all_marker_dic.items():
        model_avg_variance = 0  
        model_avg_std = 0  
        model_avg_cv = 0  
        model_all_cv = {}
        model_all_confidence_cv = []  

        for dataset, markers in dataset_dict.items():
            correct_ratios = [marker["marker_correct_ratio"] for marker in markers.values()]
            
            variance = np.var(correct_ratios, ddof=1) 
            
            std = np.std(correct_ratios, ddof=1) 
            
            mean_correct_ratio = np.mean(correct_ratios)
            if mean_correct_ratio != 0:
                cv = std / mean_correct_ratio
            else:
                cv = 0 
                
            model_avg_variance += variance
            model_avg_std += std
            model_avg_cv += cv
            model_all_cv[dataset] = cv

            if len(correct_ratios) > 1:
                confidence_variance = np.var(correct_ratios, ddof=1)
                confidence_std = np.std(correct_ratios, ddof=1)
                confidence_mean = np.mean(correct_ratios)
                if confidence_mean != 0:
                    confidence_cv = confidence_std / confidence_mean
                else:
                    confidence_cv = 0
                model_all_confidence_cv.append(confidence_cv)

        model_avg_variance /= len(dataset_dict)
        model_avg_std /= len(dataset_dict)
        model_avg_cv /= len(dataset_dict)
        
        concentration_dic[model] = {
            "avg_variance": model_avg_variance,
            "avg_std": model_avg_std,
            "avg_cv": model_avg_cv
        }
        
        if model_all_confidence_cv:
            avg_confidence_cv = np.mean(model_all_confidence_cv)
            marker_confidence_cvs[model] = avg_confidence_cv
        else:
            marker_confidence_cvs[model] = None  
        
        all_cvs[model] = model_all_cv

    with open("{}/I_AvgCV_thres={}.json".format(BASE, marker_count), 'w', encoding='utf-8') as f:
        json.dump(marker_confidence_cvs, f, indent=4, ensure_ascii=False)

        
def calculate_dataset_avg_cv(all_marker_dic, marker_count):
    # In-domain Number cv
    dataset_cv_dic = {}
    dataset_counts = {}
    
    for model, dataset_dict in all_marker_dic.items():
        for dataset, markers in dataset_dict.items():
            correct_ratios = [marker["marker_correct_ratio"] for marker in markers.values()]
            
            std = np.std(correct_ratios, ddof=1)  
            
            mean_correct_ratio = np.mean(correct_ratios)
            cv = std / mean_correct_ratio if mean_correct_ratio != 0 else 0
            
            if dataset not in dataset_cv_dic:
                dataset_cv_dic[dataset] = 0
                dataset_counts[dataset] = 0
            
            dataset_cv_dic[dataset] += cv
            dataset_counts[dataset] += 1
    
    for dataset in dataset_cv_dic:
        dataset_cv_dic[dataset] /= dataset_counts[dataset]
    
    with open("{}/dataset_avg_cv_thres={}.json".format(BASE, marker_count), 'w', encoding='utf-8') as f:
        json.dump(dataset_cv_dic, f, indent=4, ensure_ascii=False)
    
    return dataset_cv_dic
        
def calculate_concentration_number(all_number_dic, marker_count):
    concentration_dic = {}
    for model, dataset_dict in all_number_dic.items():
        model_avg_variance = 0  
        model_avg_std = 0  
        model_avg_cv = 0  
        for dataset, numbers in dataset_dict.items():

            variance = np.var(numbers, ddof=1)  
            
            std = np.std(numbers, ddof=1)  
            
            mean_number = np.mean(numbers)
            if mean_number != 0:
                cv = std / mean_number
            else:
                cv = 0 
                
            model_avg_variance += variance
            model_avg_std += std
            model_avg_cv += cv
        
        model_avg_variance /= len(dataset_dict)
        model_avg_std /= len(dataset_dict)
        model_avg_cv /= len(dataset_dict)
        
        concentration_dic[model] = {
            "avg_variance": model_avg_variance,
            "avg_std": model_avg_std,
            "avg_cv": model_avg_cv
        }
    
    with open("{}/concentration_extent_number.json".format(BASE), 'w', encoding='utf-8') as f:
        json.dump(concentration_dic, f, indent=4, ensure_ascii=False)

def spearman_correlation(all_marker_dic, marker_count):
    # MRC
    model_avg_spearman = {}  

    for model, dataset_dict in all_marker_dic.items():
        spearman_coeffs = [] 
        datasets = list(dataset_dict.keys())
        
        for ds1, ds2 in itertools.combinations(datasets, 2):
            markers1 = dataset_dict[ds1]
            markers2 = dataset_dict[ds2]
            
            common_markers = set(markers1.keys()) & set(markers2.keys())
            if len(common_markers) < 2:
                continue
            
            values1_count = [markers1[m]["marker_count"] for m in common_markers]
            values2_count = [markers2[m]["marker_count"] for m in common_markers]
            
            values1_ratio = [markers1[m]["marker_correct_ratio"] for m in common_markers]
            values2_ratio = [markers2[m]["marker_correct_ratio"] for m in common_markers]
            
            rho_count, _ = spearmanr(values1_count, values2_count)
            rho_ratio, _ = spearmanr(values1_ratio, values2_ratio)
            
            spearman_coeffs.append((rho_count, rho_ratio))
        
        if spearman_coeffs:
            avg_rho_count = sum(rho[0] for rho in spearman_coeffs) / len(spearman_coeffs)
            avg_rho_ratio = sum(rho[1] for rho in spearman_coeffs) / len(spearman_coeffs)
        else:
            avg_rho_count = avg_rho_ratio = None
        
        model_avg_spearman[model] = {
            "avg_rho_count": avg_rho_count,
            "avg_rho_ratio": avg_rho_ratio
        }
    
    with open("{}/marker_all_rank_spearman_thres={}.json".format(BASE, marker_count), 'w', encoding='utf-8') as f:
        json.dump(spearman_coeffs, f, indent=4, ensure_ascii=False)

    with open("{}/marker_rank_spearman_thres={}.json".format(BASE, marker_count), 'w', encoding='utf-8') as f:
        json.dump(model_avg_spearman, f, indent=4, ensure_ascii=False)
    

def compute_model_stability_correlations(all_marker_acc_dic, all_marker_dic):
    # Model capability impact

    def compute_cv_stability(model_data):
        marker_values = {}
        for dataset, markers in model_data.items():
            for marker, marker_info in markers.items():
                ratio = marker_info["marker_correct_ratio"]
                marker_values.setdefault(marker, []).append(ratio)

        epsilon = 1e-8  
        marker_cv = {marker: np.std(values) / (np.mean(values) + epsilon) for marker, values in marker_values.items()}

        cv_stability_score = np.mean(list(marker_cv.values()))
        return cv_stability_score

    def compute_ranking_consistency(model_data):
        dataset_rankings = {}
        for dataset, markers in model_data.items():
            marker_list = [(marker, info["marker_correct_ratio"]) for marker, info in markers.items()]
            sorted_marker_list = sorted(marker_list, key=lambda x: x[1], reverse=True)
            ranking = {marker: rank for rank, (marker, _) in enumerate(sorted_marker_list, start=1)}
            dataset_rankings[dataset] = ranking

        correlations = []
        dataset_keys = list(dataset_rankings.keys())
        for ds1, ds2 in itertools.combinations(dataset_keys, 2):
            common_markers = set(dataset_rankings[ds1].keys()).intersection(dataset_rankings[ds2].keys())
            if len(common_markers) < 2:
                continue 
            ranks1 = [dataset_rankings[ds1][marker] for marker in common_markers]
            ranks2 = [dataset_rankings[ds2][marker] for marker in common_markers]
            corr, _ = spearmanr(ranks1, ranks2)
            correlations.append(corr)
        
        avg_corr = np.mean(correlations) if correlations else None
        return avg_corr

    avg_accuracy = {model: np.mean(list(dataset_acc.values())) for model, dataset_acc in all_marker_acc_dic.items()}

    model_metrics = {}
    for model, model_data in all_marker_dic.items():
        cv_stability = compute_cv_stability(model_data)
        ranking_consistency = compute_ranking_consistency(model_data)
        model_metrics[model] = {
            "cv_stability": cv_stability,
            "ranking_consistency": ranking_consistency
        }

    common_models = set(avg_accuracy.keys()).intersection(model_metrics.keys())
    acc_list = []
    cv_stability_list = []
    ranking_consistency_list = []

    for model in common_models:
        if model_metrics[model]["ranking_consistency"] is None:
            continue
        acc_list.append(avg_accuracy[model])
        cv_stability_list.append(model_metrics[model]["cv_stability"])
        ranking_consistency_list.append(model_metrics[model]["ranking_consistency"])

    if len(acc_list) < 2:
        corr_cv = None
        corr_rank = None
    else:
        corr_cv = np.corrcoef(acc_list, cv_stability_list)[0, 1]
        corr_rank = np.corrcoef(acc_list, ranking_consistency_list)[0, 1]

    return corr_cv, corr_rank

def calculate_marker_model_correlation(all_marker_dic, all_marker_acc_dic):
    # MAC
    model_correlations = {}

    # Iterate over each model in all_marker_dic
    for model_name, datasets in all_marker_dic.items():
        # Identify markers that are shared across all datasets for the model
        shared_markers = set(datasets[next(iter(datasets))].keys())  # Start with markers from the first dataset
        for dataset_name, markers in datasets.items():
            shared_markers &= set(markers.keys())  # Find common markers across all datasets
        
        # Calculate the correlation for each shared marker
        marker_correlations = []
        for marker in shared_markers:
            marker_accuracies = []
            overall_accuracies = []

            # Collect marker accuracies and overall model accuracies for each dataset
            for dataset_name, markers in datasets.items():
                marker_accuracy = markers[marker]["marker_correct_ratio"]
                overall_accuracy = all_marker_acc_dic[model_name].get(dataset_name)

                if overall_accuracy is not None:
                    marker_accuracies.append(marker_accuracy)
                    overall_accuracies.append(overall_accuracy)

            # Calculate the correlation coefficient between marker accuracies and overall model accuracies
            if len(marker_accuracies) > 1:  # Ensure we have enough data points
                correlation, _ = pearsonr(marker_accuracies, overall_accuracies)
                marker_correlations.append(correlation)

        # Compute the average correlation for this model
        if marker_correlations:
            model_correlations[model_name] = np.mean(marker_correlations)
    
    print("Model-Marker Correlations:", model_correlations)

    return model_correlations

def calculate_rankings(all_marker_dic):
    rankings = {}
    
    for model, datasets in all_marker_dic.items():
        rankings[model] = {}
        for dataset, markers in datasets.items():
            marker_list = []
            for marker, metrics in markers.items():
                marker_list.append({
                    'Marker': marker,
                    'Correct Ratio': metrics['marker_correct_ratio'],
                    'Marker Count': metrics['marker_count']
                })
            marker_list.sort(key=lambda x: x['Correct Ratio'], reverse=True)
            for rank, marker_info in enumerate(marker_list):
                normalized_rank = rank / (len(marker_list) - 1)  
                rankings[model].setdefault(dataset, {})[marker_info['Marker']] = {
                    'normalized_ranking': normalized_rank,
                    'confidence': marker_info['Correct Ratio'],
                    'marker_count': metrics['marker_count']
                }
    
    return rankings


def plot_marker_rankings(all_marker_dic, model_name, marker_num_in_graph, redefined_markers=None):
    all_datasets = list(all_marker_dic[model_name].keys())
    common_markers = set(all_marker_dic[model_name][all_datasets[0]].keys())
    for dataset in all_datasets[1:]:
        common_markers &= set(all_marker_dic[model_name][dataset].keys())
    
    marker_rankings = {}
    for marker in common_markers:
        rankings = []
        for dataset in all_datasets:
            marker_data = all_marker_dic[model_name][dataset].get(marker, None)
            if marker_data:
                correct_ratio = marker_data["marker_correct_ratio"]
                sorted_correct_ratios = sorted(
                    [data["marker_correct_ratio"] for data in all_marker_dic[model_name][dataset].values()],
                    reverse=True
                )
                ranking = sorted_correct_ratios.index(correct_ratio) + 1 
                rankings.append(ranking / len(all_marker_dic[model_name][dataset]))  # Normalize ranking between 0 and 1
        marker_rankings[marker] = rankings
    
    marker_variances = {marker: np.var(rankings) for marker, rankings in marker_rankings.items()}
    sorted_markers = sorted(marker_variances, key=marker_variances.get, reverse=True)
    markers_to_plot = sorted_markers[:marker_num_in_graph]
    if redefined_markers is not None:
        markers_to_plot = redefined_markers
    
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    marker_symbols = ['*', 'h', 's', 'p', 'H', '+', 'x', 'D', '^', 'v']
    marker_colors = plt.cm.get_cmap('tab10', len(all_datasets))
    
    for idx, marker in enumerate(markers_to_plot):
        rankings = marker_rankings[marker]
        
        for dataset_idx, dataset in enumerate(all_datasets):
            ax.scatter(dataset, rankings[dataset_idx], label=marker.capitalize() if dataset_idx == 0 else "", 
                       marker=marker_symbols[idx % len(marker_symbols)], 
                       color=marker_colors(dataset_idx), s=400, edgecolor='black', alpha=0.5)
    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Normalized Ranking', fontsize=18)
    if model_name == 'gpt-4o':
        model_name = 'GPT-4o'
    ax.set_title(f'{model_name}', fontsize=18, fontstyle='italic')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', bbox_to_anchor=(1, 0.07), 
              labelspacing=0.1, handletextpad=0.07, markerscale=0.5, fontsize=12)
    
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("marker_rankings_{}.pdf".format(model_name))
    plt.show()