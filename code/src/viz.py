import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_corr(corr, annot=True, cmap='RdYlGn', center=0, square=True):
    """
    Plot a heatmap correlation plot.

    Parameters:
    - corr: Correlation matrix
    - annot: Whether to annotate the heatmap with correlation values (default is True)
    - cmap: Colormap for the heatmap (default is 'RdYlGn')
    - center: Center value for the colormap (default is 0)
    - square: Whether to force the plot to be square (default is True)

    Returns:
    - None
    """
    # Creating an upper triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Creating a heatmap correlation plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, center=center, square=square)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr_list, tpr_list, roc_auc_list, labels):
    """
    Plot ROC curves for multiple models on the same graph.

    Parameters:
    - fpr_list: List of false positive rates for each model
    - tpr_list: List of true positive rates for each model
    - roc_auc_list: List of ROC AUC scores for each model
    - labels: List of labels for each model

    Returns:
    - None
    """
    plt.clf()

    for fpr, tpr, roc_auc, label in zip(fpr_list, tpr_list, roc_auc_list, labels):
        plt.plot(fpr, tpr, label=f'{label} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot a confusion matrix.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - labels: List of class labels (default is None)

    Returns:
    - None
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    display_labels = labels if labels is not None else ['True', 'False']

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cm_display.plot()

    plt.show()

def plot_histogram_with_stats(data_series):
    """
    Plots a histogram with mean and median lines for the given data series.

    Parameters:
    data_series (pandas.Series): The data series for which to generate the histogram.

    Returns:
    None
    """
    sns.histplot(data_series)
    plt.axvline(data_series.mean(), color='r', linestyle='--')
    plt.axvline(data_series.median(), color='g', linestyle='-')
    
    plt.legend({'Mean': data_series.mean(), 'Median': data_series.median()})
    plt.show()

def plot_boxplot(df, x_col, y_col):
    """
    Function to create a boxplot using Seaborn.

    Parameters:
        - df (DataFrame): The DataFrame containing the data.
        - x_col (str): The column name for the x-axis.
        - y_col (str): The column name for the y-axis.
    """
    plt.figure(figsize=(6,5))
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.show()

def plot_histograms(df, columns=None, bins=10, figsize=(15, 10), xscale='linear', yscale='linear'):
    """
    Plot histograms for specified columns in a DataFrame.
    
    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - columns (list): List of column names to plot. If None, plot all numeric columns.
    - bins (int): Number of bins for the histograms. Default is 10.
    - figsize (tuple): Size of the figure. Default is (15, 10).
    - xscale (str): Scale for the x-axis (e.g., 'linear', 'log'). Default is 'linear'.
    - yscale (str): Scale for the y-axis (e.g., 'linear', 'log'). Default is 'linear'.
    """
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # To create a 3-column layout
    fig, axs = plt.subplots(num_rows, 3, figsize=figsize)
    axs = axs.flatten()  # Flatten the 2D array of axes into 1D for easy iteration

    for i, col in enumerate(columns):
        axs[i].hist(df[col], bins=bins, edgecolor='black')
        axs[i].set_title(col)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].set_xscale(xscale)
        axs[i].set_yscale(yscale)

    # Remove empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

def plot_categorical_countplots(df, figsize=(15, 5), max_cols=3, yscale='linear', xscale='linear', max_count=10):
    """
    Plot count plots for categorical columns in the DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - figsize (tuple): Size of each individual subplot. Default is (15, 5).
    - max_cols (int): Maximum number of columns in each row of subplots. Default is 3.    
    - xscale (str): Scale for the x-axis (e.g., 'linear', 'log'). Default is 'linear'.
    - yscale (str): Scale for the y-axis (e.g., 'linear', 'log'). Default is 'linear'.
    """
    # Filter columns with less than or equal to 10 unique values and are of type 'object'
    categorical_columns = [column for column in df.columns if df[column].nunique() <= max_count and df[column].dtype == 'object']

    # Determine the number of rows and columns for subplots
    num_plots = len(categorical_columns)
    num_rows = (num_plots + max_cols - 1) // max_cols
    num_cols = min(num_plots, max_cols)

    # Create subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(figsize[0], figsize[1]*num_rows))
    axs = axs.flatten()  # Flatten the 2D array of axes into 1D for easy iteration

    # Plot count plots for each categorical column
    for i, column in enumerate(categorical_columns):
        sns.countplot(data=df, x=column, ax=axs[i])
        axs[i].set_title(f'Count plot of {column}')
        axs[i].set_xlabel(column)
        axs[i].set_ylabel('Count')
        # checking for scales as input 
        if xscale:
            axs[i].set_xscale(xscale)
        if yscale:
            axs[i].set_yscale(yscale)

    # Remove empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    plt.show()