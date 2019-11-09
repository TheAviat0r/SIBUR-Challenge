import pandas as pd
import pathlib

class Dataset:
    def __init__(self, data_dir, data_type="activity", remove_nulls=True):
        assert data_type in {"activity", "atactic"}
        self.data_path = pathlib.Path(data_dir)
        self.data_type = data_type
        self.remove_nulls = remove_nulls
        self.train_df = None
        self.test_df = None
        
    def _extract_datasets(self):
        train_data = pd.read_csv(
            self.data_path.joinpath("activity_train.csv.zip"),
            parse_dates=["date"], index_col="date",
            compression="zip"
        )
        test_data = pd.read_csv(
            self.data_path.joinpath("activity_test.csv.zip"),
            parse_dates=["date"], index_col="date",
            compression="zip"
        )
        
        data = pd.concat([train_data[test_data.columns], test_data])
        data.drop("f28", axis=1, inplace=True)

        if self.data_type == "activity":
            test_target_file = "activity_test_timestamps.csv"
            target_cols = ["activity"]
        else:
            test_target_file = "atactic_test_timestamps.csv"
            target_cols = ["atactic_1", "atactic_2", "atactic_3"]
            
        test_target = pd.read_csv(
            self.data_path.joinpath(test_target_file),
            index_col="date",
            parse_dates=["date"]
        )
        
        self.train_df = train_data[target_cols].join(data.shift(6, freq="H"))
        if self.remove_nulls:
            self.train_df = self.train[self.train.notnull().all(axis=1)]
            
        self.test_df = test_target.join(data.shift(6, freq="H")).ffill()
    
    @property
    def train(self):
        if self.train_df is None:
            self._extract_datasets()
            
        return self.train_df
    
    @property
    def test(self):
        if self.test_df is None:
            self._extract_datasets()
            
        return self.test_df
    
    @property
    def folds(self):
        dates = ["2018-08", "2018-09", "2018-10", "2018-11", "2018-12"]
        if self.data_type == "activity":
            target_cols = ["activity"]
        else:
            target_cols = ["atactic_1", "atactic_2", "atactic_3"]
            
        X = self.train_df.drop(target_cols, axis=1)
        y = self.train_df[target_cols]
    
        for dt in dates:
            train_idx, val_idx = (set(self.train_df[:dt].index), set(self.train_df[dt:].index))
            
            X_train, y_train = X[X.index.isin(train_idx)], y[y.index.isin(train_idx)]
            X_val, y_val = X[X.index.isin(val_idx)], y[y.index.isin(val_idx)]
            
            yield (X_train, y_train), (X_val, y_val)
