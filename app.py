from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
# В виду того, что во входных признаках отсутствуют NAN изменил DataPreprocessing
from DataPreprocessing import DataPreprocessing
from FeatureGenetator import FeatureGenetator
from My_pca import My_pca

import logging
from logging.handlers import RotatingFileHandler
from time import strftime

X_TRAIN_DATASET_PATH = 'data/X_concat.csv'
Y_TRAIN_DATASET_PATH = 'data/y_concat.csv'
BEST_MODEL_PATH = 'models/catb_model.pkl'

app = Flask(__name__)
#best_model = pickle.load(open(BEST_MODEL_PATH, 'rb'))

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    to_predict_df = request.form.to_dict()
    to_predict_df = pd.DataFrame(list(to_predict_df.values())).T.\
        rename(columns=dict(zip(range(18), to_predict_df.keys())))
    to_predict_df = pd.concat([pd.DataFrame({'Id': [1]}), to_predict_df], axis=1)
    to_num = ['DistrictId', 'Floor', 'HouseYear', 'Social_1', 'Social_2', 'Social_3',
              'Helthcare_2', 'Shops_1']
    to_float = ['Rooms', 'Square', 'LifeSquare', 'KitchenSquare', 'Floor', 'HouseFloor', 'Ecology_1',
                'Healthcare_1']
    try:
        to_predict_df[to_num] = to_predict_df[to_num].astype("int8")
    except ValueError as e:
        logger.warning(f'{dt} Exception: {str(e)}')
        result = str(e)
        return render_template('index.html',
                               result=f'Mistake: ${result}')
    try:
        to_predict_df[to_float] = to_predict_df[to_float].astype("float32")
    except ValueError as e:
        logger.warning(f'{dt} Exception: {str(e)}')
        result = str(e)
        return render_template('index.html',
                               result=f'Mistake: ${result}')

    X_train = pd.read_csv(X_TRAIN_DATASET_PATH)
    y_train = pd.read_csv(Y_TRAIN_DATASET_PATH)

    preprocessor = DataPreprocessing()
    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    try:
        to_predict_df = preprocessor.transform(to_predict_df)
    except Exception as e:
        logger.warning(f'{dt} Exception: {str(e)}')
        result = str(e)
        return render_template('index.html',
                               result=f'Mistake: ${result}')

    features_gen = FeatureGenetator()
    features_gen.fit(X_train, y_train)
    X_train = features_gen.transform(X_train)
    to_predict_df = features_gen.transform(to_predict_df)

    last_col = [
        'DistrictId', 'Rooms', 'Square', 'LifeSquare', 'KitchenSquare', 'Floor', 'HouseFloor', 'HouseYear',
        'Ecology_1', 'Social_1', 'Social_2', 'Social_3', 'Healthcare_1', 'Helthcare_2', 'Shops_1', 'Square_outlier',
        'Rooms_outlier', 'HouseFloor_outlier', 'LifeSquare_outlier', 'Healthcare_1_nan', 'Square_2', 'LifeSquare_2',
        'DistrictId_mark', 'DistrictId_count', 'DistrictId_E1_mark', 'DistrictId_Social2_mark',
        'DistrictId_Shops_1_mark',
        'DistrictId_Healthcare_2_mark', 'floor_cat', 'year_cat', 'kitch_cat', 'B_floor_cat', 'Ecol1_cat',
        'MedPriceByDistrict', 'MedPriceByKitchenLS', 'MedPriceByBFF', 'MedPriceByEcol1',
        'MedPriceBySocial1', 'MedPriceBySocial2', 'MedPriceByShop']

    out_col = ['Social2_cat', 'LS_cat', 'Shop_cat', 'Social1_cat', 'Ecology_3', 'Shops_2',
               'HouseYear_outlier', 'Ecology_2', 'new_district']
    step_2 = My_pca(last_col, out_col)
    X_train = step_2.fit_transform(X_train)
    to_predict_df = step_2.transform(to_predict_df)

    with open(BEST_MODEL_PATH, 'rb') as model:
        best_model = pickle.load(model)

    try:
        result = best_model.predict(to_predict_df)
        result = round(result[0], 2)
    except AttributeError as e:
        logger.warning(f'{dt} Exception: {str(e)}')
        result = list(str(e))
        return render_template('index.html',
                               result=f'Mistake: ${result}')

    return render_template('index.html', result=f'A house with your parameters has a value of ${result}')

if __name__ == "__main__":
    app.run()