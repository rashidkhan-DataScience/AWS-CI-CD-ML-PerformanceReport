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
                # 'Catboost Regressor' : CatBoostRegressor(),
                'XGBoost Regressor' : XGBRegressor(),
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f"Best model found on both training and testing dataset")

            save_object(self.model_trainer_config.trained_model_path,best_model)

            predicted = best_model.predict(X_test)

            r2_Score = r2_score(y_test,predicted)
            return r2_Score
        
        except Exception as e:
            raise CustomException(e,sys)