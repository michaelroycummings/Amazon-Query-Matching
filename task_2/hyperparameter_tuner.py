from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import os

from task_2.data_handler import DataHandler
from task_2.model_builder import ModelBuilder

class HyperparameterTuner(DataHandler):
    def __init__(self, dataset_preparation):
        self.dataset_preparation = dataset_preparation

    def tune(self):
        param_grid = {
            'optimizer_name': ['adam', 'sgd', 'rmsprop'],
            'learning_rate': [1e-2, 1e-4, 1e-6],
            'input_type': ['combined', 'separate'],
            'batch_size': [16, 32]
        }

        model = KerasClassifier(build_fn=lambda: ModelBuilder(), epochs=3, verbose=0)
        dataset = self.dataset_preparation.prepare_dataset('combined')
        X, y = np.array(dataset[0]), np.array(dataset[1])

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
        grid_result = grid.fit(X, y)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        self.save_results(grid_result)
        return grid_result.best_estimator_

    def save_results(self, grid_result):
        filepath_results = os.path.join(self.path_results, 'hyperparameter_tuning.pkl')
        filepath_models = os.path.join(self.path_models, 'best_model.pkl')

        results_df = pd.DataFrame(grid_result.cv_results_)
        results_df.to_pickle(filepath_results)

        best_model = grid_result.best_estimator_.model
        best_model.save(filepath_models)

    def main(self):
        best_model = self.tune()
        return best_model
