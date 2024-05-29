import os
import numpy as np
import pickle
import subprocess
import click
import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from train import setup_mlflow
import logging
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

log_dir = './Log_Info'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'optimize_model.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_pickle(filename: str) -> any:
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def register_best_model(client: MlflowClient, top_n: int) -> None:
    # Retrieve the top_n model runs based on RMSE
    experiment = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME"))
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.RMSE ASC"]
    )
    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/model"

    # Register the best model
    mlflow.register_model(model_uri=best_model_uri, name="Best Regression Model")
    logging.info(f"Registered model {best_model_uri} as 'Best Regression Model'")



@click.command()
@click.option(
    "--data_path",
    default="./Output",
    help="Folder where data was saved"
)
@click.option(
    "--num_trials",
    default=10,
    help="The number of parameter evaluations for the optimizer to explore"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def optimize_models(data_path: str, num_trials: int, top_n: int) -> None:
    client = MlflowClient()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    def objective(params):
        model_name = params['model']
        del params['model']
        if model_name == 'Lasso':
            model = Lasso(**params)
        elif model_name == 'Ridge':
            model = Ridge(**params)
        else:
            raise ValueError("Invalid model name")
        
        with mlflow.start_run(run_name=f'{model_name} Optimization', nested=True):
            mlflow.log_params(params)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse_optimized = root_mean_squared_error(y_test, y_pred)
            mlflow.log_metric("RMSE", rmse_optimized)
            mlflow.sklearn.log_model(model, "model")

            return {'loss': rmse_optimized, 'status': STATUS_OK}

    search_space = {
        'model': hp.choice('model', ['Lasso', 'Ridge']),
        'alpha': hp.uniform('alpha', 0.01, 1.0),
        'max_iter': scope.int(hp.quniform('max_iter', 100, 1000, 1)),
        'random_state': 42
    }

    with mlflow.start_run(run_name="Regression Models Optimization"):
        rstate = np.random.default_rng(42)
        trials = Trials()
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=trials,
            rstate=rstate
        )

        # Register the best model
        register_best_model(client, top_n)


if __name__ == "__main__":
    setup_mlflow()
    optimize_models()