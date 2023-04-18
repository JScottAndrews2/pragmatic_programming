import pandas as pd
from typing import List, Union
from sklearn.model_selection import train_test_split


class DataPreparator:

    def __init__(self, data: Union[str, pd.DataFrame], features: List[str], dep_var: str = 'target',
                 min_miss: Union[int, None] = None):
        """
        :Parameters:
            raw_data: pd.DataFrame
                the raw data as a Pandas dataframe

            features: List[str]
                A list of strings containing feature names

            dep_var: str
                The name of the independent variable column
        """
        raw_data = self._load_data(data)
        self.features = features
        self.dep_var = dep_var
        self.data = self._remove_missing(raw_data[features + [dep_var]], min_miss=min_miss)
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def split_data(self, val_set: bool = False, test_size: float = .30, val_size: float = .30, random_state: int = 123,
                   shuffle: bool = True, stratify: Union[str, None] = None):
        """
        :Parameters:
            random_state: int
                A start seed for random number generation
            val_set: bool
                If true, create a validation set
            test_size: float
                A float for the percentage of data to be held as the test_set (drawn from the total sample)
            val_size: float
                A float for the percentage of data to be held as the val_set (drawn from the training sample after
                removing the test set)
            shuffle: bool
                If true, shuffle data before random split
            stratify: str or None
                If string, stratify the sample using the string as column name
        """

        self._split_error_checking(val_set=val_set, test_size=test_size, val_size=val_size, shuffle=shuffle,
                                   stratify=stratify)
        x = self.data[self.features]
        y = self.data[self.dep_var]
        if stratify is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                    random_state=random_state,
                                                                                    shuffle=shuffle, stratify=stratify)
            if val_set is not None:
                self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                                      test_size=test_size,
                                                                                      random_state=random_state,
                                                                                      shuffle=shuffle,
                                                                                      stratify=stratify)
        else:
            stratify_var = self.data[stratify]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                    random_state=random_state,
                                                                                    shuffle=shuffle,
                                                                                    stratify=stratify_var)
            if val_set is not None:
                self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                                      test_size=test_size,
                                                                                      random_state=random_state,
                                                                                      shuffle=shuffle,
                                                                                      stratify=stratify)

    def _load_data(self, data):
        if isinstance(data, str):
            raw_data = pd.read_csv(data)
            return raw_data
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError("raw_data parameter must be either a path to a csv file or a pandas dataframe}")

    def _remove_missing(self, raw_data, min_miss):
        if min_miss is None:
            cleaned_data = raw_data.dropna().reset_index(drop=True)
        elif isinstance(min_miss, int):
            # Keep only variables with missing counts lower than 100
            # get the missing value counts
            na_counts = raw_data.isna().sum()
            keep_vars = [col for col in na_counts.index if na_counts[col] < 100]
            cleaned_data = raw_data[keep_vars].dropna().reset_index(drop=True)
            # Need to change the features list now
            dropped_vars = set(raw_data.columns.values) - set(cleaned_data.columns.values)
            print(f"The following variables contain fewer than the min_miss cutoff of [{min_miss}]"
                  f" and were dropped from the dataset: {dropped_vars}")
            self.features = list(set(cleaned_data.columns.values) - set([self.dep_var]))
        else:
            raise ValueError(f"min_miss parameter must be either an integer value or None, but {min_miss} was provided")
        return cleaned_data

    def _split_error_checking(self, val_set, test_size, val_size, shuffle, stratify):
        if not isinstance(val_set, bool):
            raise ValueError(f"val_set parameter must be a True or False, but {type(val_set)} was provided")
        if not isinstance(test_size, float):
            raise ValueError(f"test_size parameter must be a float value, but {type(test_size)} was provided")
        if not isinstance(val_size, float):
            raise ValueError(f"val_size parameter must be a float value, but {type(val_size)} was provided")
        if not isinstance(shuffle, bool):
            raise ValueError(f"shuffle parameter must be a boolean, but {type(shuffle)} was provided")
        if not (isinstance(stratify, str)) and (stratify is not None):
            raise ValueError(f"stratify parameter must be a string value or None, but {type(stratify)} was provided")