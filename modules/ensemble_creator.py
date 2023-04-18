from typing import Dict
from sklearn import metrics
import pandas as pd
from modules.model_classes import Model
from modules.data_preparator import DataPreparator


class EnsembleModeler:

    def __init__(self, model_specs: Dict, data_preparator: DataPreparator):
        self.model_specs = model_specs
        self.model_names = self.model_specs.keys()
        self.data_preparator = data_preparator
        self.models = self._create_ensemble()

    def test_ensemble(self):
        train_res = []
        test_res = []
        test_predictions = []
        for model_name, model_class in self.models.items():
            model = model_class(data_preparator=self.data_preparator)
            model_kwargs = self.model_specs[model_name]
            model.fit(**model_kwargs)
            train_res.append(model.predict(data_type='train'))
            test_res.append(model.predict(data_type='test'))
            test_predictions.append(model.predictions)
        test_predictions = pd.DataFrame(test_predictions, index=list(self.models.keys())).T
        # get the ensemble average values
        ensemble_predictions = test_predictions.mean(axis=1)
        mae = metrics.mean_absolute_error(self.data_preparator.y_test, ensemble_predictions)
        mse = metrics.mean_squared_error(self.data_preparator.y_test, ensemble_predictions)
        r2 = metrics.r2_score(self.data_preparator.y_test, ensemble_predictions)
        test_res.append({'MAE': mae, "MSE": mse, 'r2': r2})
        train_res = pd.DataFrame(train_res, columns=['MAE', 'MSE', 'r2'], index=list(self.model_specs.keys()))
        test_res = pd.DataFrame(test_res, columns=['MAE', 'MSE', 'r2'], index=list(self.model_specs.keys()) + ['ensemble'])
        return train_res, test_res

    def _create_ensemble(self):
        available_models = [model for model in Model.__subclasses__()]
        models = {}
        for model_name, model_kwargs in self.model_specs.items():
            # Dynamic model selection using abstract factory
            if model_name in [model_type.__name__.lower() for model_type in available_models]:
                model_class = [model_type for model_type in Model.__subclasses__() if model_type.__name__.lower() == model_name]
            else:
                raise ValueError(f"model name [{model_name}] is not available. "
                                   f"Models must be one of: lr, svr, rf, knn")
            models[model_name] = model_class[0]
        return models

