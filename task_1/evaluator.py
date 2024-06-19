import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ace_tools as tools


from task_1.data_handler import DataHandler

class Evaluator(DataHandler):

    def __init__(self):
        super().__init__()

    def load_data(self, filename):
        filepath = os.path.join(self.path_results, filename)
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data

    def load_distance_data(self, embedding_types, distance_metrics):
        data_frames = []
        for embedding_type in embedding_types:
            for distance_metric in distance_metrics:
                filename = f'distances_{distance_metric}_{embedding_type}.pkl'
                data = self.load_data(filename)
                data['embedding_type'] = embedding_type
                data['distance_metric'] = distance_metric
                data_frames.append(data)
        return pd.concat(data_frames)

    def load_evaluation_data(self, embedding_types, distance_metrics):
        data_frames = []
        for embedding_type in embedding_types:
            for distance_metric in distance_metrics:
                filename = f'evaluations_{distance_metric}_{embedding_type}.pkl'
                data = self.load_data(filename)
                data['embedding_type'] = embedding_type
                data['distance_metric'] = distance_metric
                data_frames.append(data)
        return pd.concat(data_frames)

    def compute_distance_statistics(self, embedding_types, distance_metrics):
        data = self.load_distance_data(embedding_types, distance_metrics)
        stats = ['mean', 'median', 'std', 'quantile_25', 'quantile_50', 'quantile_75']
        index = pd.MultiIndex.from_product([distance_metrics, embedding_types], names=['distance_metric', 'embedding_type'])
        distance_stats_df = pd.DataFrame(index=index, columns=stats)

        for distance_metric in distance_metrics:
            for embedding_type in embedding_types:
                subset = data[(data['distance_metric'] == distance_metric) & (data['embedding_type'] == embedding_type)]
                distance_stats_df.loc[(distance_metric, embedding_type), 'mean'] = subset['distance'].mean()
                distance_stats_df.loc[(distance_metric, embedding_type), 'median'] = subset['distance'].median()
                distance_stats_df.loc[(distance_metric, embedding_type), 'std'] = subset['distance'].std()
                distance_stats_df.loc[(distance_metric, embedding_type), 'quantile_25'] = subset['distance'].quantile(0.25)
                distance_stats_df.loc[(distance_metric, embedding_type), 'quantile_50'] = subset['distance'].quantile(0.5)
                distance_stats_df.loc[(distance_metric, embedding_type), 'quantile_75'] = subset['distance'].quantile(0.75)

        return distance_stats_df

    def compute_evaluation_statistics(self, embedding_types, distance_metrics):
        data = self.load_evaluation_data(embedding_types, distance_metrics)
        data['accuracy'] = data['correct_ranking'].astype(int)

        stats = ['mean_accuracy', 'total_correct', 'total_queries']
        index = pd.MultiIndex.from_product([distance_metrics, embedding_types], names=['distance_metric', 'embedding_type'])
        eval_stats_df = pd.DataFrame(index=index, columns=stats)

        for distance_metric in distance_metrics:
            for embedding_type in embedding_types:
                subset = data[(data['distance_metric'] == distance_metric) & (data['embedding_type'] == embedding_type)]
                eval_stats_df.loc[(distance_metric, embedding_type), 'mean_accuracy'] = subset['accuracy'].mean()
                eval_stats_df.loc[(distance_metric, embedding_type), 'total_correct'] = subset['accuracy'].sum()
                eval_stats_df.loc[(distance_metric, embedding_type), 'total_queries'] = len(subset)

        return eval_stats_df

    def plot_distance_statistics(self, embedding_types, distance_metrics):
        df = self.compute_distance_statistics(embedding_types, distance_metrics)
        tools.display_dataframe_to_user(name="Distance Statistics", dataframe=df)

    def plot_evaluation_statistics(self, embedding_types, distance_metrics):
        df = self.compute_evaluation_statistics(embedding_types, distance_metrics)
        tools.display_dataframe_to_user(name="Evaluation Statistics", dataframe=df)

    def plot_distance_distribution(self, embedding_types, distance_metrics):
        data = self.load_distance_data(embedding_types, distance_metrics)
        g = sns.FacetGrid(data, row='distance_metric', col='embedding_type', margin_titles=True, height=4)
        g.map_dataframe(sns.histplot, x='distance', hue='rank', multiple='stack', palette='viridis')
        g.set_axis_labels("Distance", "Frequency")
        g.add_legend()
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Distribution of Distances by Rank, Embedding Type, and Distance Metric')
        plt.show()

    def plot_evaluation_accuracy(self, embedding_types, distance_metrics):
        data = self.load_evaluation_data(embedding_types, distance_metrics)
        data['accuracy'] = data['correct_ranking'].astype(int)

        accuracy_summary = data.groupby(['distance_metric', 'embedding_type']).agg({'accuracy': 'mean'}).reset_index()

        g = sns.FacetGrid(accuracy_summary, row='distance_metric', margin_titles=True, height=4)
        g.map_dataframe(sns.barplot, x='embedding_type', y='accuracy', palette='viridis')
        g.set_axis_labels("Embedding Type", "Accuracy")
        g.add_legend()
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Evaluation Accuracy by Embedding Type and Distance Metric')
        plt.show()
