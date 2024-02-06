import logging
import io
import json
import pickle
import numpy as np
import pandas as pd

from datetime import timedelta, datetime
from typing import NoReturn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.preprocessing import StandardScaler

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.aws.hooks.postgres import PostgresHook
from airflow.operators.python_operator import PythonOperator
from airflow.models import DAG
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner" : "Nickolay Yakovchuk",
    "email" : "example@gmail.com",
    "email_on_failure" : True,
    "email_on_retry" : False,
    "retry" : 3,
    "retry_delay" : timedelta(minutes=1)
}

dag = DAG(
    dag_id = "mlopls_dag_1",
    schedule_interval = "0 1 * * *",
    start_date = days_ago(2),
    catchup = False,
    tags = ["mlops"],
    default_args = DEFAULT_ARGS
)

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = "yanp-mlops"
DATA_PATH = "datasets/california_housing.pkl"

def init() -> NoReturn:
    _LOG.info("Train pipeline started.")

def get_data_from_postgres() -> NoReturn:
    
    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()
    # Читаем дату из бдшки
    data = pd.read_sql_query("SELECT * FROM california_housing", con)
    # Использую ранее созданный конекшн
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1") 
    resource = session.resource("s3")
    # Сохраняю в формате пикл на S3
    pickle_byte_obj = pickle.dumbs(data)
    resource.Object(BUCKET, DATA_PATH).put(Body=pickle_byte_obj)

    _LOG.info("Data download finished")


def prepare_data() -> NoReturn:

    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key = DATA_PATH, bucket_name=BUCKET)
    data = pd.read_pickle(file)

    X, y = data[Features], data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitter = scaler.transform(X_test)

    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    for name, data in zip(['X_train', 'X_test', 'y_train', 'y_test'],
                         [X_train_fitted, X_test_fitter, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET, f"dataset/{name}.pkl").put(Body=pickle_byte_obj)

    _LOG.info("Data preparation finished")


def train_model() -> NoReturn:

    s3_hook = S3Hook("s3_connection")
    data = {}
    for name in ['X_train', 'X_test', 'y_train', 'y_test']:
        file = s3_hook.download_file(key = f"dataset/{name}.pkl", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    model = RandomForestRegressor()
    model.fit(data['X_train'], data['y_train'])
    prediction = model.predict(data['X_test'])

    result = {}
    result['r2_score'] = r2_score(data['y_test'], prediction)
    result['mse'] = mean_squared_error(data['y_test'], prediction)
    result['mae'] = median_absolute_error(data['y_test'], prediction)

    date = datetime.now().strftime("%Y_%m_%d_%H")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")
    json_byte_object = json.dumps(result)
    resource.Object(BUCKET, f"result/{date}.json").put(Body=json_byte_object)

    _LOG.info("Model training finished")


def save_results() -> NoReturn:
    _LOG.info('GJ')


task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id="get_data", python_callable=get_data_from_postgres, dag=dag)

task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag)

task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)

task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag)


task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results