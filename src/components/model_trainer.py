import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
import sys
from dataclasses import dataclass
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                    'LogisticRegression':LogisticRegression(),

}
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n===========================================================================================')
            logging.info(f'Model Report : {model_report}')

            # to get the best model score from the dictionary
            best_model_score=max(sorted(model_report.values()))

            # best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            print(f'Best model found, model name : {best_model_name}, Accuracy_score : {best_model_score} ')
            print('\n========================================================================================')
            logging.info(f"Best model found, model name : {best_model_name}, accuracy_score : {best_model_score}")

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info("Exception occoured at model training.")
            raise CustomException(e,sys)