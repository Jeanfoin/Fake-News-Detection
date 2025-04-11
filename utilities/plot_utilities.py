from typing import (
	Iterable, 
	Union, 
	Tuple, 
	Dict
)
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision
from PIL import Image

from sklearn.metrics import (
    roc_curve,
    auc
)

def is_light_or_dark(rgbColor):
    """
    Determines whether a given RGB color is light or dark.

    Parameters:
    rgbColor (tuple): A tuple representing the RGB color. It can be in the format (R, G, B) or (R, G, B, A).

    Returns:
    str: "light" if the color is light, "dark" if the color is dark.
    """
    if len(rgbColor) == 3:
        r, g, b = rgbColor
    else:
        r, g, b, _ = rgbColor

    if max(r, g, b) > 1:
        r, g, b = r/255, g/255, b/255

    brightness = r * 0.299 + g * 0.587 + b * 0.114
    return "light" if brightness > 0.5 else "dark"



def figure_to_tensor(fig: mpl.figure, fig_format: str = "jpeg"):
    """
    Converts a Matplotlib figure to a PyTorch tensor.

    Parameters:
    fig (mpl.figure): The Matplotlib figure to convert.
    fig_format (str, optional): The format in which to save the figure. Default is "jpeg".

    Returns:
    torch.Tensor: The PyTorch tensor representation of the figure.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=fig_format, bbox_inches="tight")
    buf.seek(0)
    im = torchvision.transforms.ToTensor()(Image.open(buf))
    plt.close(fig)
    return im


def simple_colormap(value: int):
    """
    Maps an integer value to a predefined color.

    Parameters:
    value (int): The integer value to map to a color.

    Returns:
    str: The color corresponding to the input value. Returns 'gray' if the value is not in the predefined dictionary.
    """
    color_dict = {0: "C0", 1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5"}
    return color_dict.get(value, 'gray')

def Rstyle_spines(
    ax: mpl.axes,
    spines_left: Iterable[str] = ["left", "bottom"],
    offset: float = 0,
    lw: float = 2,
):
    """
    This function changes the graph spines to make them
    look like R-styled
    """

    for loc, spine in ax.spines.items():
        if loc in spines_left:
            spine.set_position(("outward", offset))
            spine.set_linewidth(lw)
        else:
            spine.set_color("none")

    if "left" in spines_left:
        ax.yaxis.set_ticks_position("left")
    else:
        ax.yaxis.set_ticks([])
    if "bottom" in spines_left:
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.xaxis.set_ticks([])

def plot_triangle_corr_matrix(
    corr: pd.DataFrame,
    ax: mpl.axes,
    mask_half: str = "upper",
    annotation: bool = False,
    label_rotation: float = 55,
    annot_fs: float = 16,
    ticks_fs: float = 18,
    cbar_fs: float = 18,
    highlight: bool = False,
    high_threshold: float = 0.5,
    show_nan: bool = True,
    cramers: bool = False,
):
    """
    This functions aims at plotting the correlation heatmap
    """

    if cramers:
        cmap = plt.get_cmap("Reds")
        vmin, vmax = [0, 1]
    else:
        cmap = plt.get_cmap("seismic")
        vmin, vmax = [-1, 1]

    if mask_half == "upper":
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_tri = np.ma.masked_where(mask, corr)
        colorbar_loc = "right"
        xlabel_pos = "bottom"
        xlabel_rotation = label_rotation
        ylabel_pos = "left"

    elif mask_half == "lower":
        mask = np.tril(np.ones_like(corr_tri, dtype=bool))
        corr_tri = np.ma.masked_where(mask, corr)
        colorbar_loc = "left"
        xlabel_pos = "top"
        xlabel_rotation = -label_rotation
        ylabel_pos = "right"

    cax = ax.matshow(corr_tri, cmap=cmap, vmin=-1, vmax=1)

    cbar = plt.gcf().colorbar(cax, fraction=0.046, pad=0.04, location=colorbar_loc)
    cbar.ax.tick_params(labelsize=cbar_fs)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.xaxis.set_ticks_position(xlabel_pos)
    ax.yaxis.set_ticks_position(ylabel_pos)
    ax.set_xticklabels(
        corr.columns, rotation=xlabel_rotation, ha="right", fontsize=ticks_fs
    )
    ax.set_yticklabels(corr.columns, fontsize=ticks_fs)

    ax.spines[["right", "top", "left", "bottom"]].set_visible(False)

    if annotation:
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                if not mask[i, j] and not np.isnan(corr.iloc[i, j]):
                    val = corr.iloc[i, j]
                    color = (
                        "w"
                        if is_light_or_dark(cmap((val - vmin) / (vmax - vmin)))
                        == "dark"
                        else "k"
                    )
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=annot_fs,
                    )

    if highlight:
        highlight_mask = np.abs(corr) >= high_threshold

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if highlight_mask.iloc[i, j] and i > j:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="black",
                            lw=3,
                        )
                    )

    if show_nan:
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                if np.isnan(corr.iloc[i, j]) and i > j:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            edgecolor="black",
                            facecolor="black",
                            lw=1,
                        )
                    )


def multicolumn_barplot(
    df: pd.DataFrame,
    fig: mpl.figure,
    width: str,
    y: str,
    color: str = None,
    colormap = None,
    spacing=0.4,
    num_col: int = 2,
    tick_fs: int = 20,
    add_legend: bool = False,
):
    """This functions produces a multicolumn horizontal bar plot with customizable colors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
    fig : matplotlib figure
        The figure to add the axes to
    width : str
        Column name for bar width values
    y : str
        Column name for bar labels
    color : str, optional
        Column name for color values. If provided, colormap must also be provided
    colormap : callable, optional
        Function that maps values in df[color] to matplotlib colors
    spacing : float, default=0.4
        Spacing between columns
    num_col : int, default=2
        Number of columns
    tick_fs : int, default=20
        Font size for tick labels
    add_legend : bool, default=False
        Whether to add a legend for the colors
        
    Returns:
    --------
    list
        List of axes objects
    """
    n_features = len(df)
    xrange = (df[width].min(), df[width].max())
    axs = []
    parts = 1 / (num_col + (num_col - 1) * spacing)
    
    # Create dict to track unique colors for legend
    legend_elements = {}
    
    for i in range(num_col):
        axs.append(fig.add_axes([i * (parts + spacing * parts), 0, parts, 0.95]))
        start_idx = i * n_features // num_col
        end_idx = min((i + 1) * n_features // num_col, n_features)
        
        # Get the slice of data for this column
        data_slice = df.iloc[start_idx:end_idx][::-1]
        
        # Determine colors for bars
        if color and colormap:
            bar_colors = [colormap(val) for val in data_slice[color]]
            
            # Keep track of unique colors for legend
            if add_legend:
                for val in data_slice[color].unique():
                    if val not in legend_elements:
                        legend_elements[val] = colormap(val)
        else:
            bar_colors = "blue"  # Default color
        
        # Create bars
        bars = axs[i].barh(
            data_slice[y],
            data_slice[width],
            color=bar_colors,
        )
        
        axs[i].set_xlim(xrange)
        axs[i].tick_params(axis="both", which="major", labelsize=tick_fs)
        Rstyle_spines(axs[i], lw=1)
    
    # Add legend to the first axis if requested
    if add_legend and legend_elements and axs:
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=color, label=str(key)) 
                         for key, color in legend_elements.items()]
        axs[0].legend(handles=legend_patches, loc='upper right')
    
    fig.subplots_adjust(wspace=spacing)
    return axs

def plot_ngram_distribution(df: pd.DataFrame, 
                            ax: mpl.axes=None):
    """
    Plot a horizontal bar chart comparing the distribution of n-grams between two classes (0 and 1).
    
    Parameters:
    df (pd.DataFrame): DataFrame containing n-grams and their counts for each class.
                       Expected columns: 'ngram', 0, 1 where 0 and 1 are class counts.
    ax (matplotlib.axes.Axes, optional): Existing axes object to plot on. If None, creates a new figure and axes.
    
    Returns:
    tuple: (fig, ax) containing the figure and axes objects with the plotted distribution
    
    Notes:
    - The n-grams are displayed in reverse order (bottom to top) with the most common n-grams at the top
    - Blue bars represent counts for class 0, red bars represent counts for class 1
    - The function applies Rstyle_spines formatting to the axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    df_ = df.copy()
    df_ = df_.iloc[::-1]

    ngrams = [" ".join(ng) for ng in df_['ngram']]
    counts_0 = df_[0]
    counts_1 = df_[1]

    y = np.arange(len(ngrams))
    
    ax.barh(y=y, width=counts_0, color='blue', label='0')
    ax.barh(y=y, width=counts_1, left=counts_0, color='red', label='1')

    ax.set_yticks(ticks=y, labels=ngrams)
    Rstyle_spines(ax, lw=1)
    ax.legend(fontsize=24)
    ax.tick_params(axis="both", labelsize=16)
    return fig, ax


def plot_classifiers_scores(classifiers_scores: Dict[str, float]):
    """
    Plots the mean f1-scores of classifiers on training and test folds.

    Parameters:
    classifiers_scores (Dict[str, float]): A dictionary where the keys are classifier names and the values are dictionaries
                                           containing 'train_score' and 'test_score' lists.

    Returns:
    tuple: A tuple containing the Matplotlib figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for k, scores in classifiers_scores.items():
        ax.errorbar(np.mean(scores["train_score"]), 
                    np.mean(scores["test_score"]), 
                    xerr=np.std(scores["train_score"]),
                    yerr=np.std(scores["test_score"]), 
                    lw=2,
                    label=k)

    min_ = min(ax.get_xlim()[0], ax.get_ylim()[0])
    x = np.linspace(min_, 1, 100)
    ax.plot(x, x, color="k", ls="--")
    ax.legend(fontsize=20)
    ax.set_xlabel("Mean f1-score on training folds", fontsize=20)
    ax.set_ylabel("Mean f1-score on test folds", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_title("Classifiers score", fontsize=22)
    Rstyle_spines(ax, lw=1) 
    return fig, ax



def plot_confusion_matrix(
    cm: np.ndarray, labels_dict: Dict = None, figsize: Tuple[float, float] = (12, 12)
) -> Tuple[mpl.figure, mpl.axes]:
    """
    Plots a confusion matrix with normalized values and includes annotations with percentages and counts.

    Args:
        cm (np.ndarray): The confusion matrix (2D array).
        labels_dict (Dict, optional): A dictionary mapping label indices to label names. Default is None.
        figsize (Tuple[float, float], optional): The size of the figure to be plotted. Default is (12, 12).

    Returns:
        Tuple[mpl.figure, mpl.axes]: The figure and axes of the plot.
    """
    cm_norm = cm / np.sum(cm, axis=0)

    cmap = plt.get_cmap("seismic")
    vmin, vmax = [0, 1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(cm_norm, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    for i, (per, cnt) in enumerate(zip(np.nditer(cm_norm), np.nditer(cm))):
        x = i // cm.shape[0]
        y = i % cm.shape[0]
        color = "w" if is_light_or_dark(cmap(cm_norm[x][y])) == "dark" else "k"
        ax.text(
            x,
            y,
            f"{100 * cm_norm[x][y]:.2f}%\n({cm[x][y]})",
            color=color,
            ha="center",
            va="center",
            fontsize=15,
        )

    if labels_dict is not None:
        ax.set_xticks(np.arange(cm.shape[0]), list(labels_dict))
        ax.set_yticks(np.arange(cm.shape[0]), list(labels_dict))

    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel("Predictions", fontsize=20)
    ax.set_ylabel("True labels", fontsize=20)

    return fig, ax

def plot_full_roc_auc(
    predictions: np.array, labels: np.array, num_classes: int, figsize: Tuple = (10, 10)
) -> Tuple[mpl.figure, mpl.axes]:
    """
    Plots the full ROC curve and AUC for multi-class classification, including micro and macro averages.

    Args:
        predictions (np.array): The predicted probabilities (2D array for multi-class).
        labels (np.array): The true class labels.
        num_classes (int): The number of classes in the classification task.
        figsize (Tuple, optional): The size of the figure to be plotted. Default is (10, 10).

    Returns:
        Tuple[mpl.figure, mpl.axes]: The figure and axes of the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if num_classes > 2:
        # Setting the dictionaries
        fpr_d, tpr_d, roc_auc_d = dict(), dict(), dict()

        # One-hot encoding the labels
        oh_labels = np.zeros((labels.size, labels.max() + 1), dtype=int)
        oh_labels[np.arange(labels.size), labels] = 1

        # Grid for macro average
        fpr_grid = np.linspace(0.0, 1.0, 1000)
        mean_tpr = np.zeros_like(fpr_grid)

        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(labels, predictions[:, i], pos_label=i)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"ROC curve class {i} (AUC = {roc_auc:.2f})")

            # Macro average
            fpr_d[i], tpr_d[i], _ = roc_curve(oh_labels[:, i], predictions[:, i])
            roc_auc_d[i] = auc(fpr_d[i], tpr_d[i])
            mean_tpr += np.interp(fpr_grid, fpr_d[i], tpr_d[i])

        # Micro average
        fpr_d["micro"], tpr_d["micro"], _ = roc_curve(
            oh_labels.ravel(), predictions.ravel()
        )
        roc_auc_d["micro"] = auc(fpr_d["micro"], tpr_d["micro"])

        # Macro average
        mean_tpr /= num_classes
        fpr_d["macro"] = fpr_grid
        tpr_d["macro"] = mean_tpr

        roc_auc_d["macro"] = auc(fpr_d["macro"], tpr_d["macro"])

        ax.plot(
            fpr_d["micro"],
            tpr_d["micro"],
            label=f"micro-average ROC curve (AUC = {roc_auc_d['micro']:.2f})",
            color="k",
            linestyle="-.",
            linewidth=3,
            zorder=2,
        )

        ax.plot(
            fpr_d["macro"],
            tpr_d["macro"],
            label=f"macro-average ROC curve (AUC = {roc_auc_d['macro']:.2f})",
            color="k",
            linestyle=":",
            linewidth=3,
            zorder=2,
        )
    else:
        fpr, tpr, _ = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, label=f"(AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)
    ax.legend(loc="lower right")

    return fig, ax
