from task_1.data_preprocessor import DataPreprocessor
from task_1.embedding_matcher import EmbeddingMatcher
from task_1.evaluator import Evaluator

"""
Task 1 - Query-Product Ranking

Given a user specified query and a list of matched products, the goal of this
task is to rank the products so that the relevant products are ranked above the
non-relevant ones.
"""

# Experiment Variables
distance_metrics = ['cosine', 'euclidean']
embedding_types = ['cls', 'mean']
embedding_batch_size = 32

# Load and Preprocess Data
data_preprocessor = DataPreprocessor()
df_train, df_val = data_preprocessor.get_training_data()

# Create Embeddings, Match Products
embedding_matcher = EmbeddingMatcher(batch_size=embedding_batch_size)
for distance_metric in distance_metrics:
    for embedding_type in embedding_types:
        embedding_matcher.main(df_train, distance_metric, embedding_type)

# Evaluate Results
evaluator = Evaluator()
evaluator.plot_distance_statistics(embedding_types, distance_metrics)
evaluator.plot_evaluation_statistics(embedding_types, distance_metrics)
evaluator.plot_distance_distribution(embedding_types, distance_metrics)
evaluator.plot_evaluation_accuracy(embedding_types, distance_metrics)
