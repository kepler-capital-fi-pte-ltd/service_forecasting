import os
import pandas as pd
import subprocess

from multiprocessing import freeze_support

from flask import Flask
from flask import request

cwd = os.getcwd()

data = f"{cwd}/data/"
script = f"{cwd}/forecaster_script.py"

app = Flask(__name__)


@app.route('/ml_forecasts', methods=['POST'])
def get_data():
    try:
        req_data = request.json

        # clear the local data
        try:
            os.remove(f"{data}/training_data.csv")
            os.remove(f"{data}/testing_data.csv")
            os.remove(f"{data}/results.csv")
        except:
            print("no data")

        # get the API request and store the data locally
        training_data = pd.DataFrame(req_data['train'])
        training_data.to_csv(f"{data}training_data.csv", index=False)
        testing_data = pd.DataFrame(req_data['test'])
        testing_data.to_csv(f"{data}testing_data.csv", index=False)

        # run the forecast script to generate the result
        await_result = subprocess.check_output(
            f"conda activate alphamethods | python {script}", shell=True
        )
        result = pd.read_csv(f"{data}results.csv")

        # return the result to user in same format as existing api
        methods = list(result)
        methods.remove("y")
        output = {}
        for i in methods:
            output.update(
                {i: result[[i, "ds"]].rename(columns={i: "yhat"}).to_dict(orient="records")}
            )
        return output

    except Exception as e:
        print(e)
        return "error, could not create forecast"


if __name__ == '__main__':
    freeze_support()
    app.run(port=5001)
