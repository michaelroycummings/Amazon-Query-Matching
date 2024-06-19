import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from task_2.data_preprocessor import DataPreprocessor
from task_2.model_preprocessor import ModelPreprocessor
from task_2.hyperparameter_tuner import HyperparameterTuner

"""
Task 2 - Multi-class Product Classification

Given a query and a result list of products retrieved for this query, the goal
of this task is to classify each product as being an Exact, Substitute,
Complement, or Irrelevant match for the query.
"""

# Load and Preprocess Data
data_preprocessor = DataPreprocessor()
df_train = data_preprocessor.main("train")
model_preprocessor_train = ModelPreprocessor(df_train) # Prep data for Transformer Model
cat_map = model_preprocessor_train.cat_map

## Train Transformer with various hyperparameters and select best model
tuner = HyperparameterTuner(model_preprocessor_train)
best_model = tuner.main()

## Evaluate best model on test set (ONLY ONCE!! NO PEAKING)
df_test = data_preprocessor.main("test")
df_test_prepped = ModelPreprocessor(df_test).main(input_type='combined')
predictions = best_model.predict(df_test_prepped)

predicted_indices = np.argmax(predictions, axis=1)
predicted_labels = [cat_map[code] for code in predicted_indices]
true_labels = df_test['esci'].values

accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Accuracy on the new dataset: {accuracy}')
print(classification_report(true_labels, predicted_labels))