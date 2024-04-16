from collections import Counter, defaultdict
import os, json, cv2, random
import torch
import matplotlib.pyplot as plt 
import matplotlib
from PIL import Image
import tqdm


def calcStats(gt_labels_list, predicted_labels_list):
    """Calculate the following statistics between two sets of labels:
        1. FN: Set of labels that are in gt_labels but not in predicted_labels
        2. FP: Set of labels that are in predicted_labels but not in gt_labels
        3. Jaccard Index (IoU)L |A and B| / |A or B|
        4. Precision: |A and B| / |A|
        5. Recall: |A and B| / |B|
        6. F1 Score: 2 * Precision * Recall / (Precision + Recall) 
    args:
        gt_labels_list: List of sets of ground truth labels
        predicted_labels_list: List of sets of predicted labels  
    """
    true_categories_frequency = Counter()
    predicted_categories_frequency = Counter()
    FN_counter = Counter()
    FP_counter = Counter()
    jaccard_indices_list = []
    precisions_list = []
    recalls_list = []
    f1_scores_list = []
    for (gt_labels, predicted_labels) in tqdm.tqdm(zip(gt_labels_list, predicted_labels_list)):
        gt_labels = set(gt_labels)
        predicted_labels = set(predicted_labels)
        
        FN = gt_labels - predicted_labels
        FP = predicted_labels - gt_labels
        
        intersection = gt_labels.intersection(predicted_labels)
        union = gt_labels.union(predicted_labels)
        
        jaccard_index = len(intersection) / len(union)
        precision = len(intersection) / len(predicted_labels)
        recall = len(intersection) / len(gt_labels)
        f1_score = 2 * precision * recall / (precision + recall)

        true_categories_frequency.update(gt_labels)
        predicted_categories_frequency.update(predicted_labels)
        FN_counter.update(FN)
        FP_counter.update(FP)
        jaccard_indices_list.append(jaccard_index)
        precisions_list.append(precision)
        recalls_list.append(recall)
        f1_scores_list.append(f1_score)

    FN_rate = {k: (v / true_categories_frequency[k], true_categories_frequency[k]) for k, v in FN_counter.items()}
    FP_rate = {k: (v / predicted_categories_frequency[k], predicted_categories_frequency[k]) for k, v in FP_counter.items()}
    return FN_rate, FP_rate, jaccard_indices_list, precisions_list, recalls_list, f1_scores_list


def plotCategoryRates(category_rate, title):
    """Plot the distribution of the FN rate by categories.

    Args:
        category_rate (dict): A dictionary of the form {category: (fn_rate, frequency)}
    """
    if 255 in category_rate:
        del category_rate[255]
    
    categories, values = zip(*sorted(category_rate.items()))
    fn_rates, frequencies = zip(*values)

    plt.bar([CLASSES[category] for category in categories], np.array(fn_rates), width=0.5) 
    plt.xlabel('Category')
    plt.xticks(rotation=90) 
    plt.ylabel(f'{title} Rate')
    plt.title(f'{title} by category. Numbers at the top of the bars are frequencies of categories')

    # Add legend with absolute values
    for i, v in enumerate(fn_rates):
        plt.text(i, v, f'{frequencies[i]}', ha='center', va='bottom')
    figure = plt.gcf() # get current figure
    figure.set_size_inches(15, 15)
    plt.tight_layout()
    plt.savefig(f'{title}.png', dpi=1000, bbox_inches='tight')
    plt.show()

def readMasks(gt_mask_dir, predictions_json):
    """Read the masks from the directories and return the list of the masks"""
    gt_and_predicted_labels = defaultdict(list)
    for file_name in tqdm.tqdm(os.listdir(gt_mask_dir)):
        mask = np.array(Image.open(os.path.join(gt_mask_dir, file_name)))
        gt_and_predicted_labels[file_name].append(list(set(mask.flatten())))
    with open(predictions_json, 'r') as f:
        predictions = json.load(f)
        for prediction in tqdm.tqdm(predictions):
            file_name = prediction['file_name'].split('/')[-1]
            if len(gt_and_predicted_labels[file_name]) == 1:
                gt_and_predicted_labels[file_name].append([])
            gt_and_predicted_labels[file_name][1].append(prediction['category_id'])

    return zip(*gt_and_predicted_labels.values())

def main(gt_mask_dir, predictions_json):
    gt_labels_list, predicted_labels_list = readMasks(gt_mask_dir, predictions_json)
    FN_rate, FP_rate, jaccard_indices_list, precisions_list, recalls_list, f1_scores_list = calcStats(gt_labels_list, predicted_labels_list)
    # Plot FN and FP
    plotCategoryRates(FN_rate, 'FN')
    plotCategoryRates(FP_rate, 'FP')
    # Stats on precisions, recalls, jaccard and f1_scores.
    df_describe = pd.DataFrame({"Precision": precisions_list, 'Recall': recalls_list, 'f1': f1_scores_list, 'Jaccard': jaccard_indices_list})
    stats = df_describe.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9]).round(decimals=2)
    
    plt.figure(figsize=(6, 4))
    table = plt.table(cellText=stats.values,
                      colLabels=stats.columns,
                      rowLabels=stats.index,  # Add row labels (index)
                      loc='center',
                      cellLoc='center',
                      colWidths=[0.3, 0.3, 0.3],  # Adjust column widths as needed
                     )
    
    plt.axis('off')  # Hide axis
    
    # Save the table as an image
    plt.savefig('table_image.png', bbox_inches='tight')

