import os

class Utils_DataHandler:
    def __init__(self, task: str):
        self.path_raw_data = 'raw_data'
        self.path_results = os.path.join('results', task)
        self.path_models = os.path.join('models', task)
        self.path_raw_examples = os.path.join(
            self.path_raw_data, 'shopping_queries_dataset_examples.parquet')
        self.path_raw_products = os.path.join(
            self.path_raw_data, 'shopping_queries_dataset_products.parquet')
        self.create_directories()

    def create_directories(self):
        os.makedirs(self.path_raw_data, exist_ok=True)
        os.makedirs(self.path_results, exist_ok=True)
        os.makedirs(self.path_models, exist_ok=True)