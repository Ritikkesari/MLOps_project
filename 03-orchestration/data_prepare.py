import pandas as pd
import numpy as np
import prefect
from prefect import task, flow
from sklearn.model_selection import train_test_split

@task(name = "Data Read")
def read_data(path):
    df = pd.read_csv(path)

    df.dropna(inplace = True)

    return df

@task(name = "Data Split")
def data_split(df):

    X = df.drop(["price","airconditioning","hotwaterheating","guestroom","prefarea","basement","mainroad","furnishingstatus"],axis = 1)
    y = df["price"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


    #preparing train data for training
    train_data = X_train.join(y_train)

    #making train data normalisation 

    train_data['bedrooms'] = np.log(train_data['bedrooms']+1)
    train_data['bathrooms'] = np.log(train_data['bedrooms']+1)
    train_data['stories'] = np.log(train_data['bedrooms']+1)
    train_data['parking'] = np.log(train_data['bedrooms']+1)

    #preparing train data for training
    test_data = X_test.join(y_test)

    #making test data normalisation 
    test_data['bedrooms'] = np.log(test_data['bedrooms']+1)
    test_data['bathrooms'] = np.log(test_data['bedrooms']+1)
    test_data['stories'] = np.log(test_data['bedrooms']+1)
    test_data['parking'] = np.log(test_data['bedrooms']+1)

    X_test = test_data.drop("price",axis = 1)

    y_test = test_data["price"]

    return X_train,y_train,X_test,y_test

if __name__=="__main__":
    df = read_data("Housing.csv")
    data_split(df)



