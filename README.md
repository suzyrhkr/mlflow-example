# mlflow-example
This repo is about simple classification model(focused on using MLflow).

## Description
- train_model.py: train model/storage meta data(metrics, parameters, models) on mlflow server.
    - dataset: diabetes dataset provided by scikit-learn
    - network: ElasticNet - Linear Regression + L1 Regularization + L2 Regularization
        - parameter: alpha(Regularization coefficient), l1_ratio(ratio of  L1 Regularization and L2 Regularization)

## About MLflow
- You can check model's meta data info and download model using mlflow ui.
- You can download csv file(including all info below).

![Screenshot from 2022-01-03 17-08-34](https://user-images.githubusercontent.com/48341349/147910876-d29a31f7-0dfd-47f5-b963-b1e89b0165cf.png)

- For specific information, click each model.
  You can check version of python, tools, time-created and so on.
  
![Screenshot from 2022-01-03 17-08-57](https://user-images.githubusercontent.com/48341349/147911073-6a73d8b4-5351-46a9-b3d9-91ecc93dc5c2.png)
