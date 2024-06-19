import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from task_1.data_handler import DataHandler

class DataPreprocessor(DataHandler):

    def __init__(self):
        super().__init__()
        self.locale = 'us'
        self.random_state = 72
        self.esci_gain = {
            'E' : 1.0,
            'S' : 0.1,
            'C' : 0.01,
            'I' : 0.0,
        }
        self.esci_rank = {
            "E" : 4,
            "S" : 2,
            "C" : 3,
            "I" : 1,
        }

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

    def get_task_1_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["product_locale"] == self.locale]
        df = df[df["small_version"] == 1]
        return df

    def prep_data_llm(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter data relevant for task 1
        df = self.get_task_1_data(df)

        # Assign rank to each example
        df["rank"] = df["esci"].apply(lambda esci: self.esci_rank[esci])

        df = df[["query_id", "query", "product_id", "product_title",
                 "product_desc", "rank"]]
        return df

    def get_training_data(self) -> pd.DataFrame:
        df = self.load_raw_data()
        df = df[df["split"] == "train"]
        df = self.clean_raw_data(df)
        df = self.get_task_1_data(df)
        df = self.prep_data_llm(df)

        ## Split training data into train and val sets
        list_query_id = df["query_id"].unique()
        list_query_id_train, list_query_id_val = train_test_split(list_query_id, test_size=0.3, random_state=self.random_state)
        df_train = df[df["query_id"].isin(list_query_id_train)]
        df_val = df[df["query_id"].isin(list_query_id_val)]

        return df_train, df_val

    def get_test_data(self) -> pd.DataFrame:
        df = self.load_raw_data()
        df = df[df["split"] == "test"]
        df = self.clean_raw_data(df)
        df = self.get_task_1_data(df)
        df = self.prep_data_llm(df)
        return df