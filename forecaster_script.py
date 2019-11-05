import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from alphamethods import *

"""
This is incredibly hacky and should be refactored.
I could not get the exec to work correctly within a class.
"""
data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/')

training_data = pd.read_csv(f"{data}training_data.csv")
testing_data = pd.read_csv(f"{data}testing_data.csv")

methods = {
    "catboost": ["BaseModel"],
    "gpflow": ["GPMC", "GPRMinimizer", "VGP"],
    "h2o": ["GBR", "GLM", "NaiveBayes", "XGBoost"],
    "keras": ["MultiLayerPerceptron", "SequentialKeras"],
    "pyflux": ["GARCH"],
    "sklearn": [
        "Lasso", "LinearRegression", "MLPRegressor", "PolySVR",
        "RFEKNN", "RandomForestCV", "SGDRegression", "SVR",
        "VarDecisionTreeRegressor", "VarGBR", "VarGPR", "VarVotingRegressor"
    ],
    "statsmodels": ["GLS"]
}

ar_lags = 5

methods_list = []
for k, v in methods.items():
    for i in v:
        methods_list.append(f"{k}.{i}")

assert type(training_data) == pd.DataFrame
assert len(training_data) > 5, "insufficient data"
assert type(testing_data) == pd.DataFrame
assert len(testing_data) > 1, "no prediction data"
assert type(methods_list) == list
assert type(ar_lags) == int

# create the ar lags
all_data = training_data.append(testing_data)
for i in range(ar_lags):
    all_data[f"lag_{i}"] = all_data['y'].shift(periods=i)

# split the data
tr = all_data[:len(training_data)].dropna().set_index('ds')
te = all_data[len(training_data):].dropna().set_index('ds')

insample = {}
oos = {}

# generate all of the forecasts
for i in methods_list:
    try:
        exec(f"method = {i}(tr, 'y')")
        model, train_predictions = method.train()
        test_predictions = method.predict(te)
        insample.update({i: train_predictions['predict_y']})
        oos.update({i: test_predictions['predict_y']})
    except Exception as e:
        print(f"method {i} failed with exception {e}")

# store the results
training = pd.concat([tr, pd.DataFrame(insample)], axis=1)
testing = pd.concat([te, pd.DataFrame(oos)], axis=1)
# out_df = training.append(testing).reset_index()
out_df = testing.reset_index()

# drop ar cols
for i in range(ar_lags):
    out_df = out_df.drop(f"lag_{i}", axis=1)
    
out_df.to_csv(f"{data}results.csv", index=False)

