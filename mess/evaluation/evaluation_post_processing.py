import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch 
import tqdm
import os

from collections import defaultdict
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt 
from mess import datasets
import pycocotools.mask as mask_util

################################## ADDED BY MEKHRON ############################

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    #plt.figure(figsize=(20,20))
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_metrics(results_path, classes, output_path):
    with open(results_path, 'rb') as f:
        results = torch.load(f)
    df = pd.DataFrame(index=['mean across classes'] + list(classes), columns=['IoU', 'ACC'])
    df.loc['mean across classes'] = np.round([results['mIoU'], results['mACC']], decimals=2)
    for class_name in classes:
        df.loc[class_name] = np.round([results[f'IoU-{class_name}'], results[f'ACC-{class_name}']], decimals=2)
    df['IoU'] = df['IoU'].astype('float')
    df['ACC'] = df['ACC'].astype('float')

    plt.figure(figsize=(15,15))

    # Plot the heatmap at the specified position
    im, _ = heatmap(df, df.index, df.columns,
                    cmap="YlGn", cbarlabel="Metric value")
    annotate_heatmap(im, valfmt="{x:.2f}", size=7);
    figure = plt.gcf() # get current figure
    figure.set_size_inches(15, 15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=1000, bbox_inches='tight')
    plt.show()

def plot_results(dataset_name, predictions, output_dir, number_of_images=50):
    os.makedirs(output_dir, exist_ok=True)
    group_predictions = defaultdict(list)
    for pred in predictions:
        if pred['file_name'] in group_predictions or len(group_predictions) < number_of_images:
            group_predictions[pred["file_name"]].append(pred)
    
    dicts = list(DatasetCatalog.get(dataset_name))
    metadata = MetadataCatalog.get(dataset_name)
    for image in dicts:
        if image['file_name'] in group_predictions:
            img = cv2.imread(image['file_name'], cv2.IMREAD_COLOR)[:, :, ::-1]
            sem_seg = cv2.imread(image['sem_seg_file_name'], cv2.IMREAD_COLOR)[:, :, ::-1][:, :, 2]
            gt_vis = Visualizer(img, metadata)
            gt_vis.draw_sem_seg(sem_seg)
            pred_vis = Visualizer(img, metadata)
            for mask in group_predictions[image['file_name']]:
                try:
                    mask_color = [x / 255 for x in metadata.stuff_colors[mask['category_id']]]
                except (AttributeError, IndexError):
                    mask_color = None
            
                binary_mask = mask_util.decode(mask['segmentation'])
                text = metadata.stuff_classes[mask['category_id']]
                pred_vis.draw_binary_mask(
                    binary_mask,
                    color=mask_color,
                    #edge_color=(1.0, 1.0, 240.0 / 255),
                    text=text,
                )
                
            split_bar = np.zeros((img.shape[0], 50, img.shape[2]))
            concat = np.concatenate((img, split_bar, gt_vis.output.get_image(), split_bar, pred_vis.output.get_image()), axis=1)
            cv2.imwrite(os.path.join(output_dir, f'catseg_{dataset_name}_'  + image['file_name'].split('/')[-1]), concat[:, :, ::-1])