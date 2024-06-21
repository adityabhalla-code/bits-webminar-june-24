
import json
import os
import pathlib
import pickle as pkl
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime as dt
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":   
    
    # All paths are local for the processing container
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_x_path = "/opt/ml/processing/test/test_x.csv"
    test_y_path = "/opt/ml/processing/test/test_y.csv"
    output_dir = "/opt/ml/processing/evaluation"
    output_prediction_path = "/opt/ml/processing/output/"
        
    # Read model tar file
    with tarfile.open(model_path, "r:gz") as t:
        t.extractall(path=".")
    
    # Load model
    model = xgb.Booster()
    model.load_model("xgboost-model")
    
    # Read test data
    X_test = pd.read_csv(test_x_path)
    x_test = xgb.DMatrix(X_test.values)
    Y_test = pd.read_csv(test_y_path)
    y_test = Y_test.to_numpy()

    # Run predictions
    predictions = np.array(model.predict(x_test), dtype=float).squeeze()

    # Evaluate predictions
    test_results = pd.concat([pd.Series(predictions, name="y_pred", index=X_test.index),X_test,],axis=1,)
    test_results.head()
    
    test_rmse = mean_squared_error(y_test, test_results["y_pred"])
    report_dict = {"regression_metric":{"test_rmse":{"value":test_rmse}}}
    print(f"Test-rmse: {test_rmse:.2f}")


    # Save evaluation report
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))
    
    # Save prediction baseline file - we need it later for the model quality monitoring
    test_results.to_csv(os.path.join(output_prediction_path, 'prediction_baseline/prediction_baseline.csv'), index=False, header=True)
