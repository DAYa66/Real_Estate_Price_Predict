from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

class DataPreprocessing:
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.medians = None
        self.kitchen_square_quantile = None
        self.list1 = list(range(1, 400, 5))
        self.list_Kitch = []
        self.list_Kitch2 = []
        self.list_Floor = []
        self.final_model = None
        self.final_model_ls = None

    def fit(self, X):
        """Сохранение статистик"""
        # Расчет медиан
        #self.medians = X.median()
        self.kitchen_square_quantile = X['KitchenSquare'].quantile(.975)

        for i in range(len(self.list1) - 1):
            mdn = X.KitchenSquare[(X.Square >= self.list1[i - 1]) & (X.Square < \
                                                                     self.list1[i]) & (X.KitchenSquare > 5) & (
                                              X.KitchenSquare <= 22)].median()
            self.list_Kitch.append(mdn)

        for i in range(len(self.list1) - 1):
            mdn2 = X['KitchenSquare'][(X.Square >= self.list1[i - 1]) & (X.Square < self.list1[i]) & \
                                      (X.KitchenSquare > 5) & (X.KitchenSquare <= 22)].median()
            self.list_Kitch2.append(mdn2)

        for j in range(50):
            list_r = []
            for i in range(j, 50):
                mdn3 = X['HouseFloor'][(X.Floor == i) & (X.HouseFloor > j) & (X.HouseFloor < 97)].median()
                list_r.append(mdn3)
            self.list_Floor.append(list_r)

        # Healthcare_1
        X['Healthcare_1_nan'] = 0
        X.loc[X['Healthcare_1'].isna(), 'Healthcare_1_nan'] = 1
        Z = X[['Rooms', 'Square', 'KitchenSquare', 'Floor', 'HouseFloor', 'HouseYear', 'Ecology_1'
              , 'Social_1', 'Social_2', 'Social_3', 'Helthcare_2', 'Shops_1']][X.Healthcare_1_nan == 0]
        w = X['Healthcare_1'][X.Healthcare_1_nan == 0]
        Z_train, Z_test, w_train, w_test = train_test_split(Z, w, shuffle=True, test_size=0.25)
        self.final_model = GradientBoostingRegressor(criterion='squared_error',
                                                max_depth=7,
                                                min_samples_leaf=3,
                                                random_state=42,
                                                n_estimators=400)
        self.final_model.fit(Z_train, w_train)

    def transform(self, X):
        """Трансформация данных"""

        binary_to_numbers = {'A': 1, 'B': 0}
        X['Ecology_2'] = X['Ecology_2'].map(binary_to_numbers)  # self.binary_to_numbers = {'A': 0, 'B': 1}
        X['Ecology_3'] = X['Ecology_3'].map(binary_to_numbers)
        X['Shops_2'] = X['Shops_2'].map(binary_to_numbers)

        #  Square
        X['Square_outlier'] = 0
        X.loc[X['Square'] <= 10, 'Square_outlier'] = 1
        X.loc[X['Square'] <= 10, 'Square'] = 999999
        X.Square.fillna(999999, inplace=True)

        # Rooms
        X['Rooms_outlier'] = 0
        X.loc[(X['Rooms'] == 0) | (X['Rooms'] >= 6), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 2) & (X['Square'] < 32), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 3) & (X['Square'] < 32), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 4) & (X['Square'] < 32), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 5) & (X['Square'] < 32), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 1) & (X['Square'] > 60), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 2) & (X['Square'] > 100), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 3) & (X['Square'] < 43), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 3) & (X['Square'] > 120), 'Rooms_outlier'] = 1
        X.loc[(X['Rooms'] == 4) & (X['Square'] < 61), 'Rooms_outlier'] = 1

        X.loc[X['Rooms'] == 0, 'Rooms'] = 1
        X.loc[X['Rooms'] >= 6, 'Rooms'] = 999999

        X.loc[(X['Rooms'] == 2) & (X['Square'] < 32), 'Rooms'] = 1
        X.loc[(X['Rooms'] == 3) & (X['Square'] < 32), 'Rooms'] = 1
        X.loc[(X['Rooms'] == 4) & (X['Square'] < 32), 'Rooms'] = 1
        X.loc[(X['Rooms'] == 5) & (X['Square'] < 32), 'Rooms'] = 1
        X.loc[(X['Rooms'] == 1) & (X['Square'] > 60), 'Rooms'] = 2
        X.loc[(X['Rooms'] == 2) & (X['Square'] > 100), 'Rooms'] = 3
        X.loc[(X['Rooms'] == 3) & (X['Square'] < 43), 'Rooms'] = 2
        X.loc[(X['Rooms'] == 3) & (X['Square'] > 120), 'Rooms'] = 4
        X.loc[(X['Rooms'] == 4) & (X['Square'] < 61), 'Rooms'] = 3

        # KitchenSquare
        for i in range(len(self.list1) - 1):
            X.loc[(X.Square >= self.list1[i - 1]) & (X.Square < self.list1[i]) & (X.KitchenSquare > 0) & \
                  (X.KitchenSquare <= 5), 'KitchenSquare'] = self.list_Kitch[i]

        X.loc[(X.KitchenSquare <= 5) & (X.KitchenSquare > 0), 'KitchenSquare'] = 999999

        for i in range(len(self.list1) - 1):
            X.loc[(X.Square >= self.list1[i - 1]) & (X.Square < self.list1[i]) & (X.KitchenSquare > 22), \
                  'KitchenSquare'] = self.list_Kitch2[i]

        if X.KitchenSquare[(X.KitchenSquare > 22)].count():
            X.loc[(X.KitchenSquare > 22), 'KitchenSquare'] = 999999

        X.KitchenSquare.fillna(999999, inplace=True)

        # HouseFloor, Floor
        X['HouseFloor_outlier'] = 0
        X.loc[X['HouseFloor'] == 0, 'HouseFloor_outlier'] = 1
        X.loc[X['Floor'] > X['HouseFloor'], 'HouseFloor_outlier'] = 1

        X.HouseFloor = X.HouseFloor.fillna(1)

        for j in range(50):
            for i in range(0, 50 - j):
                X.loc[(X.Floor == i) & (X.HouseFloor == j) & (i > j), \
                      'HouseFloor'] = self.list_Floor[j][i]

        X.HouseFloor = X.HouseFloor.fillna(22)
        X.loc[(X['Floor'] > X['HouseFloor']), \
              'Floor'] = X.loc[(X['Floor'] > X['HouseFloor'])]['HouseFloor']
        X.loc[X.HouseFloor > 96, 'HouseFloor'] = 96

        X.HouseFloor.fillna(999999, inplace=True)

        # HouseYear
        current_year = datetime.now().year

        X['HouseYear_outlier'] = 0
        X.loc[X['HouseYear'] > current_year, 'HouseYear_outlier'] = 1

        X.loc[X['HouseYear'] > current_year, 'HouseYear'] = current_year

        # LifeSquare
        X['LifeSquare_outlier'] = 0
        X['LifeSquare_outlier'] = X['LifeSquare'].isna() * 1
        condition = (X['LifeSquare'].isna()) & \
                    (~X['Square'].isna()) & \
                    (~X['KitchenSquare'].isna())

        X.loc[condition, 'LifeSquare'] = X.loc[condition, 'Square'] - X.loc[condition, 'KitchenSquare'] - 3


        #Healthcare_1
        X['Healthcare_1_nan'] = 0

        # LifeSquare
        # LifeSquare
        X.loc[X['LifeSquare'] < 10, 'LifeSquare_outlier'] = 1
        # X.loc[(X.Square - X.KitchenSquare - X.LifeSquare > \
        # 33) & (X.Square < 100), 'LifeSquare_outlier'] = 1
        X.loc[((X['LifeSquare'] + X['KitchenSquare'] + \
                3) / X['Square'] > 1), 'LifeSquare_outlier'] = 1
        X.loc[((X['LifeSquare'] + X['KitchenSquare'] + \
                3) / X['Square'] > 1), 'LifeSquare_outlier'] = 1
        X['LifeSquare'][X.LifeSquare_outlier == 1] = 999999

        X.fillna(999999, inplace=True)

        return X