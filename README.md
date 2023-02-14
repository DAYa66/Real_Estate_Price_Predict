# Real_Estate_Price_Predict

Предсказание цен московской недвижимости

Добился самой высокой целевой метрики в этом соревновании.

Стек:

ML: sklearn, pandas, numpy, scipy, catboost, xgboost, lightgbm, hyperopt, tensorflow,

tensorflow-addons, keras

API: Flask

Здравствуйте!

Знакомство с проектом прошу начинать с Presentation.pdf

Папка /notebooks содержит ноутбуки:
   - Real_Estate_Price_Prediction_EDA.ipynb содержит exploratory data analysis.
   - real_es_price_msk_MLP.ipynb содержит поиск оптимальной архитектуры MLP.
   - real-es-price-msk-predict-END.ipynb содержит пайплайн нахождения наилучшего предсказания.

Папка /model содержит catb_model.pkl сжатый estimator для предсказательной модели.

В папке /data содержатся все .csv-файлы с данными

Файл app.py используется для генерации предсказания.
Этот файл использует классы, прописанные в файлах DataPreprocessing.py, 
FeatureGenetator.py и My_pca.py.



