# MLOps_project
This project is about end to end developing and deploying of ML model.

1. Using linear regression algo developed a ML model which is responsible to predict the house price based on the input provided.

2. After developing the model, Experiments have been performed with the help of MLflow. Using xgboost library, tried to find the best parameter for getting the lowest rmse value.

3. Registered the model which had lowest rmse value in the mlflow which later has been used to predict the house pricing.

4. Using fastapi created a microservice in which we need to provide certain input to get the house price.

# steps to perform to run the project.

- we can run app.py file in terminal we command : python app.py
- or we can build docker image with Dockerfile which can later be run in any platform wit command : docker run -p 8000:8000 <image name>

--> After running above steps go to url : http://127.0.0.1:8000/docs



