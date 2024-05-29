import click
import pickle
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from typing import Tuple
import logging

log_dir = './Log_Info'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'preprocess_data.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def dump_pickle(obj, filename: str) -> None:
    try:
        with open(filename, "wb") as f_out:
            pickle.dump(obj, f_out)
        logging.info(f"Successfully saved object to {filename}")
    except Exception as e:
        logging.error(f"Failed to save object to {filename}: {e}")


def read_data(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        logging.info(f"Successfully read data from {filename}")
        return df
    except FileNotFoundError:
        logging.error(f"The file {filename} does not exist.")
        raise
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        return None


def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    if df.isna().sum().sum() == 0:
        logging.info("No missing values found in the dataset.")
        return df
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    if not categorical_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        logging.info(f"Imputed missing categorical values in columns: {list(categorical_cols)}")
    
    if not numerical_cols.empty:
        num_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        logging.info(f"Imputed missing numerical values in columns: {list(numerical_cols)}")
    
    return df


def transform_data(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False) -> Tuple[csr_matrix, DictVectorizer]:
    categorical = ['Summary', 'Precip Type']
    numerical = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
                 'Wind Bearing (degrees)', 'Visibility (km)', 'Loud Cover', 'Pressure (millibars)']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
        logging.info("Fitted and transformed data using DictVectorizer.")
    else:
        X = dv.transform(dicts)
        logging.info("Transformed data using already fitted DictVectorizer.")
    return X, dv


@click.command()
@click.option("--raw_data_path", help="Path to the raw CSV data file", required=True)
@click.option("--dest_path", help="Path to save the preprocessed data", required=True)
def preprocess_data(raw_data_path: str, dest_path: str):
    try:
        df = read_data(raw_data_path)
        if df is None:
            logging.error("DataFrame is None. Exiting preprocessing.")
            return
        
        df = impute_values(df)
        target = 'Temperature (C)'
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        logging.info("Successfully split data into training and testing sets.")
        
        y_train = df_train[target].values
        y_test = df_test[target].values
        
        # Initialize and fit the DictVectorizer
        dv = DictVectorizer()
        X_train, dv = transform_data(df_train, dv, fit_dv=True)
        X_test, _ = transform_data(df_test, dv, fit_dv=False)
        
        # Create dest_path folder unless it already exists
        os.makedirs(dest_path, exist_ok=True)
        
        # Save DictVectorizer and datasets
        dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
        dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
        dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
        logging.info(f"Preprocessed data saved to {dest_path}")
    
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")

if __name__ == "__main__":
    preprocess_data()
