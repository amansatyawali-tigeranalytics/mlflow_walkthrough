import warnings
import argparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


parser = argparse.ArgumentParser()

parser.add_argument(
    "--alpha",
    "-a",
    default=0.5
)
parser.add_argument(
    "--l-ratio",
    "-l",
    default=0.5
)

args = parser.parse_args()
arg_alpha = float(args.alpha)
args_l1_ratio = float(args.l_ratio)


data_path = "data/wine-quality.csv"
data = pd.read_csv(data_path)

data.sample(10)


# mlflow server --backend-store-uri mlruns/
# --default-artifact-root mlruns/
# --host 0.0.0.0 --port 5000
remote_server_uri = "http://0.0.0.0:5000"
# set to your server URI
mlflow.set_tracking_uri(remote_server_uri)
# or set the MLFLOW_TRACKING_URI in the env

mlflow.tracking.get_tracking_uri()

exp_name = "ElasticNet_wine"
mlflow.set_experiment(exp_name)


def eval_metrics(actual, pred):
    # compute relevant metrics
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def load_data(data_path):
    data = pd.read_csv(data_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    return train_x, train_y, test_x, test_y


def train(alpha=0.5, l1_ratio=0.5):
    # train a model with given parameters
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running
    # this from the root of MLflow!)
    data_path = "data/wine-quality.csv"
    train_x, train_y, test_x, test_y = load_data(data_path)

    # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run():
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param(key="alpha", value=alpha)
        mlflow.log_param(key="l1_ratio", value=l1_ratio)
        mlflow.log_metric(key="rmse", value=rmse)
        mlflow.log_metrics({"mae": mae, "r2": r2})
        mlflow.log_artifact(data_path)
        print("Save to: {}".format(mlflow.get_artifact_uri()))

        mlflow.sklearn.log_model(lr, "model")


train(arg_alpha, args_l1_ratio)
