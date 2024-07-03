import prefect
from prefect import task, Flow, Parameter
from prefect.schedules import IntervalSchedule
from datetime import timedelta
from lr_model import model_experiment_with_mlflow
from data_prepare import data_split,read_data


@Flow

def main_flow():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("house-pricing-experiment")

    data = read_data("Housing.csv")

    X_train,y_train,X_test,y_test = data_split(data)

