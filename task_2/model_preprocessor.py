import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

class ModelPreprocessor:
    def __init__(self, dataframe, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.cat_map = None

    def main(self, input_type):
        texts = self.dataframe[['product_title', 'product_desc']]
        texts = texts.apply(lambda x: ' '.join(x) if input_type == 'combined' else self.tokenizer.encode_plus(x['product_title'], x['product_desc'], add_special_tokens=True), axis=1).values

        # Store category to code mapping
        if self.cat_map is None:
            self.cat_map = dict(enumerate(self.dataframe['esci'].astype('category').cat.categories))

        labels = np.array(self.dataframe['esci'].astype('category').cat.codes)
        dataset = self.tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=self.max_length, return_tensors='tf')
        dataset = tf.data.Dataset.from_tensor_slices((dataset, labels))
        return dataset.batch(16)