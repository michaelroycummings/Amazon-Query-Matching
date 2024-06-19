import tensorflow as tf
from transformers import TFBertModel

class ModelBuilder(tf.keras.Model):
    def __init__(self, input_type='combined'):
        super().__init__()
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(4, activation='softmax')
        self.input_type = input_type

    def build(self, inputs):
        if self.input_type == 'combined':
            output = self.bert(inputs)
        elif self.input_type == 'separate':
            output = self.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        pooled_output = output.pooler_output
        dropout_output = self.dropout(pooled_output)
        return self.classifier(dropout_output)
