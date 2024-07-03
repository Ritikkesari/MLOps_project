from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from prefect import task, Flow


@task(log_print = True, name = "Train the model")
def model_experiment_with_mlflow(X_train,y_train,X_test,y_test):

    with mlflow.start_run():

        mlflow.set_tag("developer","Ritik")

        mlflow.log_param("train-data-path", "./data/housing_train_data.csv")
        mlflow.log_param("validation-data-path", "./data/housing_test_data-01.csv")

        
        model = LinearRegression()
        model.fit(X_train,y_train)
        
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test,y_pred)
        mlflow.log_metric("mse",mse)

        rmse = np.sqrt(mse)
        mlflow.log_metric("rmse",rmse)

        r2 = r2_score(y_test,y_pred)
        mlflow.log_metric("r2_score",r2)

        mlflow.sklearn.log_model(model, "linear_regression_model")


if __name__ == "__main__":
    model_experiment_with_mlflow(X_train,y_train,X_test,y_test)
