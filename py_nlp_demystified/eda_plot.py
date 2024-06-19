from typing import List, Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def subplot_no_data(
    ax: plt.Axes, column_title: str, row_title: str, fontsize: int
    ) -> plt.Axes:
    """ Add a subplot with no data to the figure. """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(column_title, fontsize=fontsize)
    ax.set_ylabel(row_title, fontsize=fontsize)
    ax.set_facecolor('black')  # Set the background color to black
    ax.text(  # Add text in the middle
        0.5, 0.5, 'Not Applicable to this Model', fontsize=12, ha='center',
        va='center', color='white')
    return ax


def subplot_length_data(
    ax: plt.Axes, model: str, data: dict,
    column_title: str, row_title: str, fontsize: int
    ) -> plt.Axes:
    """
    Takes a distribution of data as a list of values, and adds a subplot
    of a violin plot and box plot to the figure.
    """
    sns.violinplot(data[model], ax=ax, color='lightgray')
    sns.boxplot(data[model], ax=ax, widths=0.1, color='lightgreen')
    ax.set_title(column_title, fontsize=fontsize)
    ax.set_ylabel(row_title, fontsize=fontsize)
    return ax


def plot_vector_lengths(lengths_context, lengths_contextless):
    """
    Plots the distribution of vector lengths for each model with and without
    context.

    Parameters:
        lengths_context (Dict[str, List[float]])
            A dictionary where keys are model names and values are lists of
            vector lengths for each embedding in the model with context.
        lengths_contextless (Dict[str, List[float]])
            A dictionary where keys are model names and values are lists of
            vector lengths for each embedding in the model without context.
    """
    models = ['fasttext', 'spacy_cnn', 'spacy_trf', 'bert', 'gpt2']

    fig, axs = plt.subplots(len(models), 2, figsize=(10, len(models)*3.5))
    fontsize = 10

    for i, model in enumerate(models):

        # Plot the Left Column of Subplot Grid: With Context
        ax=axs[i, 0]
        column_title = "With context"
        row_title = f"{model}\nVector Lengths"
        if model in lengths_context:
            ax = subplot_length_data(
                ax, model, lengths_context,
                column_title, row_title, fontsize)
        else:
            ax = subplot_no_data(ax, column_title, row_title, fontsize)

        # Plot the Right Column of Subplot Grid: Without Context
        ax = axs[i, 1]
        column_title = "Without context"
        if model in lengths_contextless:
            ax = subplot_length_data(
                ax, model, lengths_contextless,
                column_title, row_title, fontsize)
        else:
            ax = subplot_no_data(ax, column_title, row_title, fontsize)

    plt.tight_layout()
    plt.show()


def subplot_embedding_2d(
    embeddings: List[List[float]],# clusters: List[int],
    ax: plt.Axes, column_title: str, row_title: str, fontsize: int
    ):
    """ Add a subplot of a 2D scatter plot to the figure. """
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    ax.scatter(x, y) #, c=clusters)
    ax.set_title(column_title, fontsize=fontsize)
    ax.set_ylabel(row_title, fontsize=fontsize)
    return ax


def plot_embedding_space(
    embeddings_context: Dict[str, List[List[float]]],
    embeddings_contextless: Dict[str, List[List[float]]],
    clusters_context: Dict[str, List[int]],
    clusters_contextless: Dict[str, List[int]],
    ):
    """
    Plots the 2D embedding space for each model with and without context.
    """
    models = ['fasttext', 'spacy_cnn', 'spacy_trf', 'bert', 'gpt2']

    fig, axs = plt.subplots(len(models), 2, figsize=(10, len(models)*3.5))
    fontsize = 10

    for i, model in enumerate(models):

        # Plot the Left Column of Subplot Grid: With Context
        ax=axs[i, 0]
        column_title = "With context"
        row_title = f"{model}\nidk"
        if model in embeddings_context:
            ax = subplot_embedding_2d(
                embeddings_context[model],# clusters_context[model],
                ax, column_title, row_title, fontsize)
        else:
            ax = subplot_no_data(ax, column_title, row_title, fontsize)

        # Plot the Right Column of Subplot Grid: Without Context
        ax = axs[i, 1]
        column_title = "Without context"
        if model in embeddings_contextless:
            ax = subplot_embedding_2d(
                embeddings_contextless[model],# clusters_contextless[model],
                ax, column_title, row_title, fontsize)
        else:
            ax = subplot_no_data(ax, column_title, row_title, fontsize)

    # plt.colorbar()
    plt.tight_layout()
    plt.show()


def skip_neighbors(model_neighbors: dict, num_neighbors: int) -> dict:
    """
    Skip neighbors in the model_neighbors dictionary to reduce the number of
    neighbors to num_neighbors. Used in the plot_nn_distance_distribution
    function.

    Parameters:
        model_neighbors (Dict[str, Dict[str, List[List[int]]]])
            A dictionary where keys are model names and values are dicts where
            keys are the type of distance metric and values are lists of the
            indices of the n nearest neighbors for each embedding in the model.
        num_neighbors (int)
            The number of neighbors to keep.

    Returns:
        Dict[str, Dict[str, List[List[int]]]]
            A dictionary where keys are model names and values are dicts where
            keys are the type of distance metric and values are lists of the
            indices of the n nearest neighbors for each embedding in the model.
    """
    neighbor_indices = defaultdict(dict)
    for model_name, values in model_neighbors.items():
        for distance_name, distances in values.items():

            # Select every n-th neighbor
            selected_neighbors = []
            selected_indices = []
            for neighbors in distances:

                indices = np.linspace(
                    0, len(neighbors) - 1, num_neighbors, dtype=int)
                selected_neighbors.append(
                    [neighbors[i] for i in indices])
                selected_indices.append(indices)

            model_neighbors[model_name][distance_name] = selected_neighbors
            neighbor_indices[model_name][distance_name] = selected_indices

    return model_neighbors, neighbor_indices


def plot_nn_distance_distribution(
    model_neighbors, num_neighbors: int = 0, plot_type='kde'):

    # Clean Inputs
    if not isinstance(num_neighbors, int):
        raise ValueError('num_neighbors must be an integer')
    if plot_type not in ['kde', 'hist']:
        raise ValueError('plot_type argument must be one of "kde" or "hist"')

    # Reduce Number of Neighbors to Plot
    if num_neighbors > 0:
        model_neighbors, neighbor_indices = skip_neighbors(
            model_neighbors, num_neighbors)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Set Specific Order for Graphing - comment out for general use
    model_names = model_neighbors.keys()
    model_names = ['fasttext', 'spacy_cnn', 'spacy_trf', 'bert', 'gpt2']

    # Each Model gets a FacetGrid where
    # rows are the neighbor number and columns are the distance metric
    for model_name in model_names:

        # Set Specific Order for Graphing - comment out for general use
        distance_metrics = model_neighbors[model_name].keys()
        distance_metrics = ['euclidean', 'cosine']

        # Create a DataFrame of all embeddings for a model
        df = []
        for distance_metric in distance_metrics:
            temp_df = pd.DataFrame(model_neighbors[model_name][distance_metric])
            temp_df = temp_df.melt(var_name="Neighbor", value_name="Distance")
            temp_df['Metric'] = distance_metric
            df.append(temp_df)
        df = pd.concat(df)
        df["Neighbor"] = df["Neighbor"].apply(lambda x: f"NN {x + 1}        ,")

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(
            df, row="Neighbor", col="Metric", hue="Neighbor",
            aspect=20, height=1.1, palette=pal, sharey=False, sharex='col')

        # Draw the distribution of the neighbor distances in a few steps
        if plot_type == 'hist':
            g.map(sns.histplot, "Distance", bins=30, linewidth=1.5)
            g.refline(
                y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        elif plot_type == 'kde':
            g.map(  # Draw the filled density plot
                sns.kdeplot, "Distance", bw_adjust=.5, clip_on=False,
                fill=True, alpha=1, linewidth=1.5)
            g.map(  # Draw the line density plot
                sns.kdeplot, "Distance", bw_adjust=.5, clip_on=False,
                color="w", lw=2)
        else:
            raise ValueError('Invalid plot_type argument')

        # Label the FacetGrid rows (i.e., the neighbor number)
        def label(x, color, label):
            ax = plt.gca()
            ax.text(
                0, .2, label, fontweight="bold", color=color, ha="left",
                va="center", transform=ax.transAxes, fontsize=25)
        g.map(label, "Distance")

        # Set the row subplots to overlap
        if plot_type == 'kde':
            g.figure.subplots_adjust(hspace=-.7)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        # Y axis label for all rows of the facet grid (i.e., model name)
        g.figure.text(
            0.0, 0.5, model_name, va='center',
            rotation='vertical', fontsize=30)

        # Label the Facet Grid columns (i.e., distance metric type)
        for ax, title in zip(g.axes[0], distance_metrics):
            ax.set_title(title.capitalize(), fontsize=30, weight='bold')

        # Set x-tick label size and x-label size
        for ax in g.axes.flatten():
            ax.tick_params(axis='x', labelsize=20)
            ax.set_xlabel("Distance in Embedding Space", fontsize=30)

        plt.show()


def plot_distance_distribution(
    distances_to_plot: Dict[str, Dict[str, List[float]]],
    plot_type: str = 'kde'
    ):
    """
    Plots the distribution of distances for each model and distance metric.
    Difference distances can be given:
        E.g., distances between uses of the same word.
        E.g., distances between a word and its lemmatized form.

    Parameters:
        distances_to_plot (Dict[str, Dict[str, List[float]]])
            A dictionary where keys are model names and values are dictionaries
            where keys are the type of distance metric and values are lists of
            distances between words.
        plot_type (str)
            The type of plot to create. Must be one of 'kde' or 'hist'.
    """
    # Set Specific Order for Graphing - comment out for general use
    model_names = distances_to_plot.keys()
    model_names = ['spacy_cnn', 'spacy_trf', 'bert', 'gpt2']

    color = sns.cubehelix_palette(10, rot=-.25, light=.7)[3]

    for model_name in model_names:

        # Set Specific Order for Graphing - comment out for general use
        distance_metrics = distances_to_plot[model_name].keys()
        distance_metrics = ['euclidean', 'cosine']

        # Create subplots
        fig, axs = plt.subplots(
            1, len(distance_metrics), figsize=(10, 2))

        # Loop over each subplot and plot the data
        for i, distance_metric in enumerate(distance_metrics):
            ax = axs[i]
            distances = distances_to_plot[model_name][distance_metric]

            # Plot distribution of distances
            if plot_type == 'hist':
                sns.histplot(
                    distances, bins=30, linewidth=1.5, ax=ax, color=color)
                ax.axhline(y=0, linewidth=2, linestyle="-", color=None)
            elif plot_type == 'kde':
                sns.kdeplot(
                    distances, bw_adjust=.5, fill=True,
                    alpha=1, linewidth=1.5, ax=ax, color=color)
            else:
                raise ValueError('Invalid plot_type argument')

            # Set Titles
            y_label = model_name if i == 0 else ''
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_xlabel('Distance in Embedding Space', fontsize=10)
            ax.set_title(distance_metric.capitalize(), fontsize=10)

            # Make the plot look nicer
            ax.tick_params(axis='both', labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Remove y-tick values which are dependent on the distance scale
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()


def plot_re_distance_distribution(
    re_distances_context: Dict[str, Dict[str, List[float]]],
    re_distances_contextless: Dict[str, Dict[str, List[float]]],
    plot_type: str = 'kde'):
    """
    Plots the distribution of distances between verbs and their re counterparts
    for (1) infinitive verbs without context and (2) verbs in context, for
    each model and each distance metric.

    Parameters:
        re_distances_context (Dict[str, Dict[str, List[float]]])
            A dictionary where keys are model names and values are dictionaries
            where keys are the type of distance metric and values are lists of
            distances between embeddings of verbs without context.
        re_distances_contextless (Dict[str, Dict[str, List[float]]])
            A dictionary where keys are model names and values are dictionaries
            where keys are the type of distance metric and values are lists of
            distances between embeddings of verbs with context.
        plot_type (str)
            The type of plot to create. Must be one of 'kde' or 'hist'.
    """
    # Set Specific Order for Graphing - comment out for general use
    model_names = re_distances_context.keys()
    model_names = ['spacy_cnn', 'spacy_trf', 'bert', 'gpt2']

    color_inf = sns.cubehelix_palette(10, rot=-.25, light=.7)[3]
    color_context = sns.color_palette("pastel")[2]

    for model_name in model_names:

        # Set Specific Order for Graphing - comment out for general use
        distance_metrics = re_distances_context[model_name].keys()
        distance_metrics = ['euclidean', 'cosine']

        # Create subplots
        fig, axs = plt.subplots(
            1, len(distance_metrics), figsize=(10, 2))

        # Loop over each subplot and plot the data
        for i, distance_metric in enumerate(distance_metrics):
            ax = axs[i]
            distances_context = \
                re_distances_context[model_name][distance_metric]
            distances_contextless = \
                re_distances_contextless[model_name][distance_metric]


            # Plot distribution of distances
            if plot_type == 'hist':
                sns.histplot(
                    distances_context, bins=30, linewidth=1.5, alpha=0.5,
                    ax=ax, color=color_context, label='context')
                sns.histplot(
                    distances_contextless, bins=30, linewidth=1.5, alpha=0.5,
                    ax=ax, color=color_inf, label='contextless')
                ax.axhline(y=0, linewidth=2, linestyle="-", color=None)
            elif plot_type == 'kde':
                sns.kdeplot(
                    distances_context, bw_adjust=.5, fill=True, alpha=0.5,
                    linewidth=1.5, ax=ax, color=color_context, label='context')
                sns.kdeplot(
                    distances_contextless, bw_adjust=.5, fill=True, alpha=0.5,
                    linewidth=1.5, ax=ax, color=color_inf, label='contextless')
            else:
                raise ValueError('Invalid plot_type argument')

            # Set Titles
            y_label = model_name if i == 0 else ''
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_xlabel('Distance in Embedding Space', fontsize=10)
            ax.set_title(distance_metric.capitalize(), fontsize=10)

            # Make the plot look nicer
            legend = ax.legend(fontsize=8.5)
            legend.get_frame().set_facecolor('white')
            ax.tick_params(axis='both', labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Remove y-tick values which are dependent on the distance scale
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()