import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def inititiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Splitting training and test set')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'AdaBoost Regressor' : AdaBoostRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Linear Regression' : LinearRegression(),
                'KNN Regressor' : KNeighborsRegressor(),
                'Catboost Regressor' : CatBoostRegressor(),
                'XGBoost Regressor' : XGBRegressor(),
            }

            params = {
                'Random Forest' : {
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Decision Tree' : {
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                },
                'AdaBoost Regressor' : {
                    'learning_rate':[0.1,0.05,0.01,],
                    'loss' : ['linear', 'square', 'exponential'],
                    'n_estimators': [1,6,8,16,25,36]
                },
                'Linear Regression': {
                    'fit_intercept': [True, False]
                },
                'Gradient Boosting':{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'KNN Regressor': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                'Catboost Regressor': {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                'XGBoost Regressor': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            best_model_score = max(model_report.values())

            best_model_name = max(model_report,key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f"Best model found on both training and testing dataset")

            save_object(self.model_trainer_config.trained_model_path,best_model)

            predicted = best_model.predict(X_test)

            r2_Score = r2_score(y_test,predicted)
            return best_model_name,r2_Score
        
        except Exception as e:
            raise CustomException(e,sys)