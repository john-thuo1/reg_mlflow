import os
import pickle
import subprocess
import click
import mlflow
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
import logging
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

log_dir = './Log_Info'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(filename=os.path.join(log_dir, 'train_model.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_pickle(filename: str) -> any:
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def setup_mlflow() -> None:
    try:
        subprocess.Popen(["mlflow", "ui", "--backend-store-uri", os.getenv("MLFLOW_TRACKING_URI")])
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        experiment = mlflow.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME"))
        if experiment is None:
            mlflow.create_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
        else:
            logging.info("Experiment already exists.")
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError) as e:
        logging.error(f"Error occurred while setting up MLflow: {e}")


@click.command()
@click.option(
    "--data_path",
    default="./Output",
    help="Folder where data was saved"
)
def train_model(data_path: str) -> None:

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run(run_name="model/regression_models"):
        mlflow.autolog(extra_tags={"developer": "@johnthuo"})
        mlflow.log_param("train-data-path", "./Output/train.pkl")
        mlflow.log_param("test-data-path", "./Output/test.pkl")

        # Lasso Regression
        with mlflow.start_run(nested=True, run_name="Lasso"):
            alpha = 0.1
            lasso_model = Lasso(alpha)
            lasso_model.fit(X_train, y_train)
            y_pred_lasso = lasso_model.predict(X_test)
            rmse_lasso = root_mean_squared_error(y_test, y_pred_lasso)
            mlflow.log_metric("RMSE", rmse_lasso)
        
        # Ridge Regression Model
        with mlflow.start_run(nested=True, run_name="Ridge"):
            alpha_ridge = 0.5
            ridge_model = Ridge(alpha_ridge)
            ridge_model.fit(X_train, y_train)
            y_pred_ridge = ridge_model.predict(X_test)
            rmse_ridge = root_mean_squared_error(y_test, y_pred_ridge)
            mlflow.log_metric("RMSE", rmse_ridge)

if __name__ == '__main__':
    setup_mlflow()
    train_model()
