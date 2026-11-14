import os
import sys
import dill
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score


def save_object(file_path:str,obj):
    try:
        logging.info('Saving the pickle file in the artifacts folder')
        
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)    
        
        logging.info('Retrieved the file path and reading the object')
        
        with open(file_path,mode='wb') as file:
            dill.dump(obj,file)                
            logging.info(f'Pickel file save in {os.path.dirname(file_path)}')
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        logging.info('Model evaluation started')

        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        logging.info('Model evaluation completed')
        return report

    except Exception as e:
        raise CustomException(e,sys)