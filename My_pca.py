import pandas as pd

class My_pca:
    """Нормализация и сжатие малозначимых признаков"""

    def __init__(self, last_col, out_col):
        self.scaler = None
        self.out_col = out_col
        self.last_col = last_col
        self.pca = None

    def my_scaler_fit(self, df):
        """Обучаю MinMaxScaler для стандартизации нумерованных фичей для   упаковки в  PCA"""
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(df)

    def my_scaler_transform(self, df):
        """Преобразую MinMaxScaler_ом  фичи для упаковки в  PCA"""
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=df.columns.to_list())

        return df_scaled

    def pca_fit_transform(self, train):
        """Функция сжимающая малозначимые признаки методом PCA и возвращающая кроме
        обработанного датасета натренированную модель PCA"""
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=2, random_state=42)
        df_small = train[self.out_col]
        train_pca = pd.concat([pd.DataFrame(self.pca.fit_transform(df_small), \
                                            columns=['component_1', 'component_2']), train[self.last_col]], axis=1)

        return train_pca

    def pca_transform(self, test):
        """Функция сжимающая малозначимые признаки методом PCA на тестовой выборке"""
        df_small = test[self.out_col]
        test_pca = pd.concat([pd.DataFrame(self.pca.transform(df_small), \
                                           columns=['component_1', 'component_2']), test[self.last_col]], axis=1)

        return test_pca

    def fit_transform(self, train):
        self.my_scaler_fit(train)
        train = self.my_scaler_transform(train)
        train_pca = self.pca_fit_transform(train)

        return train_pca

    def transform(self, test):
        test = self.my_scaler_transform(test)
        test_pca = self.pca_transform(test)

        return test_pca