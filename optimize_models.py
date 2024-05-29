import os
import numpy as np
import pickle
import click
import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from preprocess_data import Logger
from train import setup_mlflow
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()


MODEL_CLASS_MAP = {
    'Lasso': Lasso,
    'Ridge': Ridge
}


def load_pickle(filename: str) -> any:
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def register_best_model(client: MlflowClient, top_n: int) -> None:
    Logger.info("Registering best model...")
    experiment = client.get_experiment_by_name(os.getenv("MLFLOW_EXPERIMENT_NAME"))
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.RMSE ASC"]
    )
    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=best_model_uri, name="Best Regression Model")
    Logger.info(f"Registered model {best_model_uri} as 'Best Regression Model'")


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
    Logger.info("Starting model optimization...")
    client = MlflowClient()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    def objective(params):
        model_name = params.pop('model')
        model_class = MODEL_CLASS_MAP.get(model_name)
        
        if model_class is None:
            error_message = f"Invalid model name: {model_name}"
            Logger.error(error_message)
            raise ValueError(error_message)
        
        model = model_class(**params)
        
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
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=Trials(),
            rstate=np.random.default_rng(42)
        )

        # Register the best model
        register_best_model(client, top_n)
        Logger.info("Optimization completed successfully.")

if __name__ == "__main__":
    setup_mlflow()
    optimize_models()
