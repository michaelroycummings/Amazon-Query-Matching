
import os
import pickle
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine, euclidean

from task_1.data_handler import DataHandler

class EmbeddingMatcher(DataHandler):
    def __init__(self, batch_size=32):
        super().__init__()
        self.model_name='bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.batch_size = batch_size
        self.filename_embeddings_cls = 'embeddings_cls.pkl'
        self.filename_embeddings_mean = 'embeddings_mean.pkl'

    def encode(self, texts, method='mean'):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if method == 'cls':
            return outputs.last_hidden_state[:, 0, :]
        elif method == 'mean':
            return outputs.last_hidden_state.mean(dim=1)

    def create_embeddings(self, dataframe):
        unique_products = dataframe.drop_duplicates(subset=['product_id'])
        product_texts = (unique_products['product_title'] + " " + unique_products['product_desc']).tolist()
        product_ids = unique_products['product_id'].tolist()

        embeddings_cls = {}
        embeddings_mean = {}

        # Process texts in batches and store embeddings
        for i in range(0, len(product_texts), self.batch_size):
            print(i)
            batch_texts = product_texts[i:i + self.batch_size]
            batch_cls_embeddings = self.encode(batch_texts, 'cls')
            batch_mean_embeddings = self.encode(batch_texts, 'mean')
            for j, (cls_emb, mean_emb) in enumerate(
                zip(batch_cls_embeddings, batch_mean_embeddings)):
                embeddings_cls[product_ids[i + j]] = cls_emb.cpu().numpy()
                embeddings_mean[product_ids[i + j]] = mean_emb.cpu().numpy()

        self.save_embeddings(embeddings_cls, self.filename_embeddings_cls)
        self.save_embeddings(embeddings_mean, self.filename_embeddings_mean)

    def save_embeddings(self, embeddings, filename):
        filepath = os.path.join(self.path_results, filename)
        with open(filepath, 'wb') as file:
            pickle.dump(embeddings, file)

    def get_embeddings(self, dataframe, embedding_type):
        if embedding_type == 'cls':
            path_embeddings = os.path.join(self.path_results, self.filename_embeddings_cls)
        elif embedding_type == 'mean':
            path_embeddings = os.path.join(self.path_results, self.filename_embeddings_mean)
        else:
            raise ValueError("Unsupported embedding type")

        if not os.path.exists(path_embeddings):
            print(f"{embedding_type.capitalize()} embeddings file not found for {embedding_type}. Creating new embeddings...")
            self.create_embeddings(dataframe)

        print(f"Loading {embedding_type.capitalize()} embeddings from file...")
        with open(path_embeddings, 'rb') as file:
            embeddings = pickle.load(file)
        return embeddings

    def compute_distance(self, query_emb, product_emb, distance_metric):
        if distance_metric == 'cosine':
            return cosine(query_emb, product_emb)
        elif distance_metric == 'euclidean':
            return euclidean(query_emb, product_emb)
        else:
            raise ValueError("Unsupported distance metric")

    def match_products(self, dataframe, embeddings, distance_metric):
        distances = []
        for _, row in dataframe.iterrows():
            query_emb = self.encode(row['query'])[0]
            product_emb = embeddings[row['product_id']]
            distance = self.compute_distance(query_emb, product_emb, distance_metric)
            distances.append(distance)
        dataframe['distance'] = distances
        return dataframe

    def evaluate_ranking(self, distances_df):
        ranking_results = []
        grouped = distances_df.groupby('query_id')
        for query_id, group in grouped:
            rank_1 = group[group['rank'] == 1]['distance']
            rank_4 = group[group['rank'] == 4]['distance']
            correct_ranking = rank_1.quantile(0.95) < rank_4.quantile(0.95)
            ranking_results.append({'query_id': query_id, 'correct_ranking': correct_ranking})
        return pd.DataFrame(ranking_results)

    def save_results(self, results_df, dist_or_eval, distance_metric, embedding_type):
        filename = f'{dist_or_eval}_{distance_metric}_{embedding_type}.pkl'
        filepath = os.path.join(self.path_results, filename)
        with open(filepath, 'wb') as file:
            pickle.dump(results_df, file)

    def main(self, dataframe, distance_metric, embedding_type):
        ''' Run the embedding matcher '''
        embeddings = self.get_embeddings(dataframe, embedding_type)
        distances_df = self.match_products(dataframe, embeddings, distance_metric)
        evaluation_df = self.evaluate_ranking(distances_df)
        self.save_results(distances_df, 'distances', distance_metric, embedding_type)
        self.save_results(evaluation_df, 'evaluations', distance_metric, embedding_type)
