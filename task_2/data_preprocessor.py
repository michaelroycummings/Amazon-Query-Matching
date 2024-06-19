import pandas as pd
from sklearn.model_selection import train_test_split

from task_2.data_handler import DataHandler

class DataPreprocessor(DataHandler):

    def __init__(self):
        super().__init__()
        self.locale = 'us'
        self.random_state = 72

    def load_raw_data(self):
        '''
        Raw data columns:
            example_id, query, query_id, product_id, product_locale, esci_label,
            small_version, large_version, split,
            product_title, product_description, product_bullet_point,
            product_brand, product_color and source
        '''
        df_examples = pd.read_parquet(self.path_raw_examples)
        df_products = pd.read_parquet(self.path_raw_products)
        df_examples_products = pd.merge(
            df_examples,
            df_products,
            how='left',
            left_on=["product_locale", "product_id"],
            right_on=["product_locale", "product_id"]
        )
        df_examples_products.rename(columns={
            "product_description": "product_desc",
            "esci_label": "esci"}, inplace=True)
        return df_examples_products

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(
            subset=["query", "product_title", "product_desc", "esci"])
        return df

    def get_task_2_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["product_locale"] == self.locale]
        df = df[df["large_version"] == 1]
        return df

    def main(self, train_or_test: str) -> pd.DataFrame:
        df = self.load_raw_data()
        df = df[df["split"] == train_or_test]
        df = self.clean_raw_data(df)
        df = self.get_task_2_data(df)
        return df
