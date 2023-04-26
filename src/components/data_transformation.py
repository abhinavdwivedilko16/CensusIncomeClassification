import os  # os is used to save the path of test data and train data
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Initiated")
            #Define which feature is to be oridinal encoded and which has to be scaleed
            categorical_cols=['workclass', 'education', 'marital_status', 'occupation','relationship', 'race', 'sex', 'native_country']
            numerical_cols=['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss','hours_per_week']
            
            #define custom ranking for each ordinal variables
            workclass_categories=[' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov',' Local-gov', ' ?',
                                ' Self-emp-inc', ' Without-pay',' Never-worked']
            education_categories=[' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th',' Some-college', ' Assoc-acdm', 
                                ' Assoc-voc', ' 7th-8th',' Doctorate', ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th',' Preschool', ' 12th']
            marital_categories=[' Never-married', ' Married-civ-spouse', ' Divorced',' Married-spouse-absent', 
                                ' Separated', ' Married-AF-spouse',' Widowed']
            occupation_categories=[' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',' Prof-specialty', 
                                ' Other-service', ' Sales', ' Craft-repair',' Transport-moving', ' Farming-fishing', 
                                ' Machine-op-inspct',' Tech-support', ' ?', ' Protective-serv', ' Armed-Forces',' Priv-house-serv']
            relationship_categories=[' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',' Other-relative']
            race_categories=[' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo',' Other']
            sex_categories=[' Male', ' Female']
            native_categories=[' United-States', ' Cuba', ' Jamaica', ' India', ' ?', 
                            ' Mexico',' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada',
                            ' Germany', ' Iran', ' Philippines', ' Italy', ' Poland',' Columbia', 
                            ' Cambodia', ' Thailand', ' Ecuador', ' Laos',' Taiwan', ' Haiti', ' Portugal', 
                            ' Dominican-Republic',' El-Salvador', ' France', ' Guatemala', ' China', ' Japan', 
                            ' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland',
                            ' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland', 
                            ' Hungary', ' Holand-Netherlands']

            logging.info('Pipeline Initiated')
            #Numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler()),

                ]
            )

            #Categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[workclass_categories,education_categories,marital_categories,occupation_categories,relationship_categories,race_categories,sex_categories,native_categories])),
                ('scalar',StandardScaler())
                ]
            )
            
            preprocessor=ColumnTransformer([

                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            return preprocessor
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #Reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(train_df.shape)
            logging.info(test_df.shape)

            logging.info("Reading Train and Test Data completed.")
            logging.info(f'Train DataFrame head: \n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame head: \n{test_df.head().to_string()}')

            logging.info("obtaining preprocessing object")
            preprocessing_obj= self.get_data_transformation_object()

            target_column_name= 'income'
            drop_columns=[target_column_name]

            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=train_df[target_column_name]

            #Transforming using preprocessor object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on a training and testing datasets")

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor.pkl saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Error occored in Data Transformation')
            raise CustomException(e,sys)