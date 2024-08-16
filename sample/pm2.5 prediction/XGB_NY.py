import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os
from datalayer import *
from metrics import evaluate
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import *


def relpace_nan(data):
    # return KNNImputer().fit_transform(data)
    return IterativeImputer().fit_transform(data)
    # return SimpleImputer().fit_transform(data)


def create_df():
    pm = pickle.load(open("data/NewYorkState_pm2.5_imputedS.pickle", "rb"))
    pm.print_info()
    nightlight = pickle.load(
        open("data/NewYorkState_light_imputedS.pickle", "rb")
    )
    nightlight.print_info()
    population = pickle.load(
        open("data/NewYorkState_pop_imputedS.pickle", "rb")
    )
    population.print_info()
    intersection_dens = pickle.load(
        open("data/NewYorkState_inter.pickle", "rb")
    )
    intersection_dens.print_info()
    # built.data = relpace_nan(built.data)
    # pm.data = relpace_nan(pm.data)
    # nightlight.data = relpace_nan(nightlight.data)
    # population.data = relpace_nan(population.data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    pm.data = scaler.fit_transform(pm.data)
    nightlight.data = scaler.fit_transform(nightlight.data)
    population.data = scaler.fit_transform(population.data)
    intersection_dens.data = scaler.fit_transform(intersection_dens.data)

    df = pd.DataFrame()
    # df["pickup"] = pickup.data.flatten()
    # df["dropoff"] = dropoff.data.flatten()
    # df["built"] = built.data.flatten()
    df["pm2.5"] = pm.data.flatten()
    df["nightlight"] = nightlight.data.flatten()
    df["population"] = population.data.flatten()
    df["intersection_dens"] = intersection_dens.data.flatten()
    return df


def split_data(df, label: str, *args, random_seed=42):
    features = [arg for arg in args]
    X = df[features]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9, test_size=0.1, random_state=random_seed
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_estimators": 200,
        "learning_rate": 0.1,
        "n_jobs": -1,
        "random_state": 42,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    return model


def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = format(np.sqrt(mean_squared_error(y_test, y_pred)), ".4f")
    print("RMSE:", rmse)
    return rmse

def Main(XY, name):
    X_train, X_test, y_train, y_test = XY
    model = train_model(X_train, X_test, y_train, y_test)
    y_pred = model.predict(X_test)

    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    # plt.axis("equal")
    # plt.axis("square")
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.title(name)
    plt.savefig(name+".pdf")
    plt.clf()
    # plt.show()

    mae_score, rmse_score, r_2_score = evaluate(y_test, y_pred)
    print(
        # "mae_score: %f, rmse_score: %f, r_2_score: %f"
        "mae_score, rmse_score, r_2_score: %.3f & %.3f & %.3f"
        % (mae_score, rmse_score, r_2_score)
    )

    rmse = test_model(model, X_test, y_test)
    model.save_model('model/xgboost/'+name+'.model')
    print('Model saved.')

if __name__ == "__main__":
    df = create_df()
    Main(split_data(
        df,
        "pm2.5",
        # "pickup",
        # "dropoff",
        # "population",
        # "built",
        # "nightlight",
        "intersection_dens",
    ), "XGB_NY_inter")

    Main(split_data(
        df,
        "pm2.5",
        # "pickup",
        # "dropoff",
        # "population",
        # "built",
        "nightlight",
        # "intersection_dens",
    ), "XGB_NY_light")

    Main(split_data(
        df,
        "pm2.5",
        # "pickup",
        # "dropoff",
        "population",
        # "built",
        # "nightlight",
        # "intersection_dens",
    ), "XGB_NY_pop")

    Main(split_data(
        df,
        "pm2.5",
        # "pickup",
        # "dropoff",
        # "population",
        # "built",
        "nightlight",
        "intersection_dens",
    ), "XGB_NY_inter+light")

    Main(split_data(
        df,
        "pm2.5",
        # "pickup",
        # "dropoff",
        "population",
        # "built",
        # "nightlight",
        "intersection_dens",
    ), "XGB_NY_inter+pop")

    Main(split_data(
        df,
        "pm2.5",
        # "pickup",
        # "dropoff",
        "population",
        # "built",
        "nightlight",
        # "intersection_dens",
    ), "XGB_NY_night+pop")

    Main(split_data(
        df,
        "pm2.5",
        # "pickup",
        # "dropoff",
        "population",
        # "built",
        "nightlight",
        "intersection_dens",
    ), "XGB_NY_inter+night+pop")