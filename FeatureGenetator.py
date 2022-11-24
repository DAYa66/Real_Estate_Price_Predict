import numpy as np

class FeatureGenetator():
    """Генерация новых фич"""

    def __init__(self):
        self.DistrictId_counts = None
        self.binary_to_numbers = None
        self.med_price_by_district = None
        self.med_price_by_floor_year = None

    def fit(self, X, y=None):

        X = X.copy()

        if y is not None:

            # Идея создать поле отражающее значение средней цены квадратного метра в районе.

            X['Price'] = y.values
            X['item_m2_price'] = X['Price' ] /X['Square']
            mark = X.groupby(['DistrictId'], as_index=False).agg({'item_m2_price' :'median'})
            self.DistrictId_mark = dict(mark.values)


            X = X.drop('item_m2_price', axis=1)
            X = X.drop('Price', axis=1)



        # DistrictID
        district = X['DistrictId'].value_counts()
        district = district[district > 50]
        self.DistrictId_counts = dict(district)

        # Binary features
        self.binary_to_numbers = {'A': 0, 'B': 1}

        ## District
        df = X.copy()

        if y is not None:
            df['Price'] = y.values
            df['item_m2_price'] = df['Price' ] /df['Square']
            df['DistrictId_popular'] = df['DistrictId'].copy()
            df.loc[~df['DistrictId_popular'].isin(district.keys().tolist())] = 0

            self.med_price_by_district = df.groupby(['DistrictId_popular', 'Rooms'], as_index=False).agg \
                ({'Price' :'median'}). \
                rename(columns={'Price' :'MedPriceByDistrict',
                                'DistrictId_popular': 'DistrictId'})
        """
        ## floor, year
        if y is not None:
            df['Price'] = y.values
            df = self.floor_to_cat(df)
            df = self.year_to_cat(df)
            self.med_price_by_floor_year = df.groupby(['year_cat', 'floor_cat'], as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceByFloorYear'})
        """

        ## Цена в зависимости от общей площади и от площади кухни
        if y is not None:
            df['Price'] = y.values
            df = self.LSqare_to_cat(df)
            df = self.Kitchen_to_cat(df)
            self.med_price_by_Kitcen_Lsqare = df.groupby(['kitch_cat', 'LS_cat'], as_index=False).agg \
                ({'Price' :'median'}). \
                rename(columns={'Price' :'MedPriceByKitchenLS'})

        ## Цена в зависимости от этажности дома
        if y is not None:
            df['Price'] = y.values
            df = self.Bfloor_to_cat(df)
            self.med_price_by_BFF = df.groupby(['B_floor_cat'], as_index=False).agg({'Price' :'median'}). \
                rename(columns={'Price' :'MedPriceByBFF'})

        ## Цена в зависимости от Ecol1_cat
        if y is not None:
            df['Price'] = y.values
            df = self.Ecol1_to_cat(df)
            self.med_price_by_Eco1 = df.groupby(['Ecol1_cat'], as_index=False).agg({'Price' :'median'}). \
                rename(columns={'Price' :'MedPriceByEcol1'})

        ## Цена в зависимости от Social1_cat
        if y is not None:
            df['Price'] = y.values
            df = self.Social1_to_cat(df)
            self.med_price_by_Social1 = df.groupby(['Social1_cat'], as_index=False).agg({'Price' :'median'}). \
                rename(columns={'Price' :'MedPriceBySocial1'})

        ## Цена в зависимости от Social1_cat
        if y is not None:
            df['Price'] = y.values
            df = self.Social2_to_cat(df)
            self.med_price_by_Social2 = df.groupby(['Social2_cat'], as_index=False).agg({'Price' :'median'}). \
                rename(columns={'Price' :'MedPriceBySocial2'})

        ## Цена в зависимости от Shop_cat
        if y is not None:
            df['Price'] = y.values
            df = self.Shop_to_cat(df)
            self.med_price_by_Shop = df.groupby(['Shop_cat'], as_index=False).agg({'Price' :'median'}). \
                rename(columns={'Price' :'MedPriceByShop'})



        E1_mark = X.groupby(['DistrictId'], as_index=False).agg({'Ecology_1' :'median'})
        self.DistrictId_E1_mark = dict(E1_mark.values)

        Social2_mark = X.groupby(['DistrictId'], as_index=False).agg({'Social_2' :'median'})
        self.DistrictId_Social2_mark = dict(Social2_mark.values)

        Shops_1_mark = X.groupby(['DistrictId'], as_index=False).agg({'Shops_1' :'median'})
        self.DistrictId_Shops_1_mark = dict(Shops_1_mark.values)

        Healthcare_2_mark = X.groupby(['DistrictId'], as_index=False).agg({'Helthcare_2' :'median'})
        self.DistrictId_Healthcare_2_mark = dict(Healthcare_2_mark.values)

        # обнуление переменных для районов где статистика ничтожна
        for item in self.DistrictId_counts:
            if self.DistrictId_counts[item] == 0:
                self.DistrictId_Healthcare_2_mark[item] = 0
                self.DistrictId_Social2_mark[item] = 0
                self.DistrictId_E1_mark[item] = 0
                self.med_price_by_Kitcen_Lsqare[item] = 0
                self.med_price_by_BFF[item] = 0
                self.med_price_by_Eco1[item] = 0
                self.med_price_by_Social1[item] = 0
                self.DistrictId_Shops_1_mark[item] =0
                self.med_price_by_district[item] = 0
                # self.med_price_by_floor_year[item] =0
                self.med_price_by_Social2[item] = 0
                self.med_price_by_Shop[item] = 0

    def transform(self, X):

        X['DistrictId_mark'] = X['DistrictId'].copy()
        X['DistrictId_mark'] = X['DistrictId'].map(self.DistrictId_mark)

        # DistrictId
        X['DistrictId_count'] = X['DistrictId'].map(self.DistrictId_counts)  # self.DistrictId_counts = {'id': value}
        X['DistrictId_E1_mark'] = X['DistrictId'].map(self.DistrictId_E1_mark)
        X['DistrictId_Social2_mark'] = X['DistrictId'].map(self.DistrictId_Social2_mark)
        X['DistrictId_Shops_1_mark'] = X['DistrictId'].map(self.DistrictId_Shops_1_mark)
        X['DistrictId_Healthcare_2_mark'] = X['DistrictId'].map(self.DistrictId_Healthcare_2_mark)

        X['new_district'] = 0
        X.loc[X['DistrictId_count'].isna(), 'new_district'] = 1
        X['DistrictId_count'].fillna(5, inplace=True)

        # Binary features
        X['Ecology_2'] = X['Ecology_2'].map(self.binary_to_numbers)
        X['Ecology_2'].fillna(9999, inplace=True)
        X['Ecology_3'] = X['Ecology_3'].map(self.binary_to_numbers)
        X['Ecology_3'].fillna(9999, inplace=True)
        X['Shops_2'] = X['Shops_2'].map(self.binary_to_numbers)
        X['Shops_2'].fillna(9999, inplace=True)

        # More categorical features
        X = self.floor_to_cat(X)  # + столбец flooar_cat
        X = self.year_to_cat(X)  # + столбец year_cat
        X = self.Kitchen_to_cat(X)  # + столбец Kitchen_to_cat
        X = self.LSqare_to_cat(X)  # + столбец LSqare_to_cat
        X = self.Bfloor_to_cat(X)  # + столбец Bfloor_to_cat
        X = self.Ecol1_to_cat(X)  # + столбец Ecol1_to_cat
        X = self.Social1_to_cat(X)  # + столбец Social1_to_cat
        X = self.Social2_to_cat(X)  # + столбец Social2_to_cat
        X = self.Shop_to_cat(X)  # + столбец Shop_to_cat
        X['Shop_cat'].fillna(9999, inplace=True)

        X['Square_2'] = X['Square'] ** 2
        X['LifeSquare_2'] = X['LifeSquare'] ** 2

        # Target encoding
        if self.med_price_by_district is not None:
            X = X.merge(self.med_price_by_district, on=['DistrictId', 'Rooms'], how='left')
        if self.med_price_by_floor_year is not None:
            X = X.merge(self.med_price_by_floor_year, on=['year_cat', 'floor_cat'], how='left')
        if self.med_price_by_Kitcen_Lsqare is not None:
            X = X.merge(self.med_price_by_Kitcen_Lsqare, on=['kitch_cat', 'LS_cat'], how='left')

        if self.med_price_by_BFF is not None:
            X = X.merge(self.med_price_by_BFF, on=['B_floor_cat'], how='left')
        if self.med_price_by_Eco1 is not None:
            X = X.merge(self.med_price_by_Eco1, on=['Ecol1_cat'], how='left')
        if self.med_price_by_Social1 is not None:
            X = X.merge(self.med_price_by_Social1, on=['Social1_cat'], how='left')
        if self.med_price_by_Social2 is not None:
            X = X.merge(self.med_price_by_Social2, on=['Social2_cat'], how='left')
        if self.med_price_by_Shop is not None:
            X = X.merge(self.med_price_by_Shop, on=['Shop_cat'], how='left')

        # переменные характеризующие районы по медианному значению
        X.loc[X['MedPriceByDistrict'].isna(), 'MedPriceByDistrict'] = X['MedPriceByDistrict'].median()
        # X.loc[X['MedPriceByFloorYear'].isna(), 'MedPriceByFloorYear'] = X['MedPriceByFloorYear'].median()

        X.loc[X['DistrictId_mark'].isna(), 'DistrictId_mark'] = X['DistrictId_mark'].median()
        X.loc[X['DistrictId_E1_mark'].isna(), 'DistrictId_E1_mark'] = X['DistrictId_E1_mark'].median()
        X.loc[X['DistrictId_Social2_mark'].isna(), 'DistrictId_Social2_mark'] = X['DistrictId_Social2_mark'].median()
        X.loc[X['DistrictId_Healthcare_2_mark'].isna(), 'DistrictId_Healthcare_2_mark'] = X[
            'DistrictId_Healthcare_2_mark'].median()

        X.loc[X['MedPriceByKitchenLS'].isna(), 'MedPriceByKitchenLS'] = X['MedPriceByKitchenLS'].median()
        X.loc[X['MedPriceByShop'].isna(), 'MedPriceByShop'] = 0
        X = X.drop(columns=['Id'])

        return X

    @staticmethod
    def floor_to_cat(X):

        X['floor_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['Floor'] < 3, 'floor_cat'] = 1
        X.loc[(X['Floor'] >= 3) & (X['Floor'] <= 5), 'floor_cat'] = 2
        X.loc[(X['Floor'] > 5) & (X['Floor'] <= 9), 'floor_cat'] = 3
        X.loc[(X['Floor'] > 9) & (X['Floor'] <= 15), 'floor_cat'] = 4
        X.loc[X['Floor'] > 15, 'floor_cat'] = 5

        return X

    @staticmethod
    def Bfloor_to_cat(X):

        X['B_floor_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['HouseFloor'] < 5, 'B_floor_cat'] = 1
        X.loc[(X['HouseFloor'] >= 5) & (X['HouseFloor'] < 9), 'B_floor_cat'] = 2
        X.loc[(X['HouseFloor'] >= 9) & (X['HouseFloor'] <= 13), 'B_floor_cat'] = 3
        X.loc[(X['HouseFloor'] > 13) & (X['HouseFloor'] <= 16), 'B_floor_cat'] = 4
        X.loc[(X['HouseFloor'] > 16) & (X['HouseFloor'] <= 21), 'B_floor_cat'] = 5
        X.loc[X['HouseFloor'] > 21, 'B_floor_cat'] = 6

        return X

    @staticmethod
    def Social1_to_cat(X):

        X['Social1_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['Social_1'] < 11.5, 'Social1_cat'] = 1
        X.loc[(X['Social_1'] >= 11.5) & (X['Social_1'] < 28.4), 'Social1_cat'] = 2
        X.loc[(X['Social_1'] >= 26.4) & (X['Social_1'] <= 40), 'Social1_cat'] = 3
        X.loc[(X['Social_1'] > 40) & (X['Social_1'] <= 55), 'Social1_cat'] = 4
        X.loc[X['Social_1'] > 55, 'Social1_cat'] = 5

        return X

    @staticmethod
    def Social2_to_cat(X):

        X['Social2_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['Social_2'] < 4200, 'Social2_cat'] = 1
        X.loc[(X['Social_2'] >= 4200) & (X['Social_2'] < 9000), 'Social2_cat'] = 2
        X.loc[X['Social_2'] > 9000, 'Social2_cat'] = 3

        return X

    @staticmethod
    def Shop_to_cat(X):

        X['Shop_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['Shops_1'] < 8, 'Shop_cat'] = 1
        X.loc[X['Shops_1'] > 8, 'Shop_cat'] = 2

        return X

    @staticmethod
    def Ecol1_to_cat(X):

        X['Ecol1_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['Ecology_1'] < 0.1, 'Ecol1_cat'] = 1
        X.loc[X['Ecology_1'] > 0.1, 'Ecol1_cat'] = 2

        return X

    @staticmethod
    def year_to_cat(X):

        X['year_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['HouseYear'] < 1941, 'year_cat'] = 1
        X.loc[(X['HouseYear'] >= 1941) & (X['HouseYear'] <= 1945), 'year_cat'] = 2
        X.loc[(X['HouseYear'] > 1945) & (X['HouseYear'] <= 1980), 'year_cat'] = 3
        X.loc[(X['HouseYear'] > 1980) & (X['HouseYear'] <= 2000), 'year_cat'] = 4
        X.loc[(X['HouseYear'] > 2000) & (X['HouseYear'] <= 2010), 'year_cat'] = 5
        X.loc[(X['HouseYear'] > 2010), 'year_cat'] = 6

        return X

    @staticmethod
    def Kitchen_to_cat(X):

        X['kitch_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['KitchenSquare'] <= 4.35, 'kitch_cat'] = 1
        X.loc[(X['KitchenSquare'] > 4.35) & (X['KitchenSquare'] <= 5.8), 'kitch_cat'] = 2
        X.loc[(X['KitchenSquare'] > 5.8) & (X['KitchenSquare'] <= 6.7), 'kitch_cat'] = 3
        X.loc[(X['KitchenSquare'] > 6.7) & (X['KitchenSquare'] <= 7.3), 'kitch_cat'] = 4
        X.loc[(X['KitchenSquare'] > 7.3) & (X['KitchenSquare'] <= 8.5), 'kitch_cat'] = 5
        X.loc[(X['KitchenSquare'] > 8.5) & (X['KitchenSquare'] <= 9.5), 'kitch_cat'] = 6
        X.loc[(X['KitchenSquare'] > 9.5) & (X['KitchenSquare'] <= 10.8), 'kitch_cat'] = 7
        X.loc[(X['KitchenSquare'] > 10.8) & (X['KitchenSquare'] <= 11.6), 'kitch_cat'] = 8
        X.loc[(X['KitchenSquare'] > 11.6) & (X['KitchenSquare'] <= 12.5), 'kitch_cat'] = 9
        X.loc[(X['KitchenSquare'] > 12.5), 'kitch_cat'] = 10

        return X

    @staticmethod
    def LSqare_to_cat(X):

        X['LS_cat'] = np.nan
        # Пороговые значения выбирались при изучении распределения целевой величины на гистограмме
        X.loc[X['Square'] < 25, 'LS_cat'] = 1
        X.loc[(X['Square'] >= 25) & (X['Square'] <= 50), 'LS_cat'] = 2
        X.loc[(X['Square'] > 50) & (X['Square'] <= 58.5), 'LS_cat'] = 3
        X.loc[(X['Square'] > 58.5) & (X['Square'] <= 70.5), 'LS_cat'] = 4
        X.loc[(X['Square'] > 70.5) & (X['Square'] <= 94), 'LS_cat'] = 5
        X.loc[(X['Square'] >= 94) & (X['Square'] <= 112), 'LS_cat'] = 6
        X.loc[(X['Square'] > 112), 'LS_cat'] = 7

        return X
