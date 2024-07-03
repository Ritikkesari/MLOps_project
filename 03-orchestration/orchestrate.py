from prefect import flow
from datetime import timedelta
from lr_model import model_experiment_with_mlflow
from data_prepare import data_split,read_data
import mlflow


@flow
def main_flow():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("house-pricing-experiment")

    data = read_data("Housing.csv")

    X_train,y_train,X_test,y_test = data_split(data)

    model_experiment_with_mlflow(X_train,y_train,X_test,y_test)

if __name__ == "__main__":
    main_flow()

