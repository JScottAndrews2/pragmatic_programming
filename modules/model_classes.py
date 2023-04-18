from abc import ABC, abstractmethod
from typing import Union
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR as svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from modules.data_preparator import DataPreparator


def standard_predict_method(data_preparator, model, data_type: str = 'test'):
    if data_type == 'train':
        predictions = model.predict(data_preparator.x_train)
        true_scores = data_preparator.y_train
    elif data_type == 'val':
        predictions = model.predict(data_preparator.x_val)
        true_scores = data_preparator.y_val
    elif data_type == 'test':
        predictions = model.predict(data_preparator.x_test)
        true_scores = data_preparator.y_test
    else:
        raise ValueError("type parameter must be one of: train, val, or test")

    mae = metrics.mean_absolute_error(true_scores, predictions)
    mse = metrics.mean_squared_error(true_scores, predictions)
    r2 = metrics.r2_score(true_scores, predictions)
    return [predictions, {'MAE': mae, "MSE": mse, 'r2': r2}]


class Model(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, data_type: str = 'train'):
        pass


class LR(Model):
    def __init__(self, data_preparator: DataPreparator):
        super().__init__()
        self.data_preparator = data_preparator

    def fit(self, fit_intercept: bool = True):
        self.model = LinearRegression(fit_intercept=fit_intercept)
        self.model.fit(self.data_preparator.x_train, self.data_preparator.y_train)

    def predict(self, data_type: str = 'test'):
        out = standard_predict_method(data_preparator=self.data_preparator, model=self.model, data_type=data_type)
        self.predictions = out[0]
        return out[1]


class SVR(Model):
    def __init__(self, data_preparator: DataPreparator):
        super().__init__()
        self.data_preparator = data_preparator

    def fit(self, kernel: str = 'rbf', C: float = 1.0, epsilon: float = 0.1):
        self.model = make_pipeline(StandardScaler(), svm(kernel=kernel, C=C, epsilon=epsilon))
        self.model.fit(self.data_preparator.x_train, self.data_preparator.y_train)

    def predict(self, data_type: str = 'test'):
        out = standard_predict_method(data_preparator=self.data_preparator, model=self.model, data_type=data_type)
        self.predictions = out[0]
        return out[1]


class KNN(Model):
    def __init__(self, data_preparator: DataPreparator):
        super().__init__()
        self.data_preparator = data_preparator

    def fit(self, n_neighbors: int = 5, weights: str = 'uniform', leaf_size: int = 30):
        self.model = KNeighborsRegressor(n_neighbors=5, weights=weights, leaf_size=leaf_size)
        self.model.fit(self.data_preparator.x_train, self.data_preparator.y_train)

    def predict(self, data_type: str = 'test'):
        out = standard_predict_method(data_preparator=self.data_preparator, model=self.model, data_type=data_type)
        self.predictions = out[0]
        return out[1]


class RF(Model):
    def __init__(self, data_preparator: DataPreparator):
        super().__init__()
        self.data_preparator = data_preparator

    def fit(self, n_estimators: int = 100, max_depth: Union[int, None] = None, min_samples_split: int = 2,
            random_state: int = 123):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_split=min_samples_split,
                                           random_state=random_state)
        self.model.fit(self.data_preparator.x_train, self.data_preparator.y_train)

    def predict(self, data_type: str = 'test'):
        out = standard_predict_method(data_preparator=self.data_preparator, model=self.model)
        self.predictions = out[0]
        return out[1]
