import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# import predictors
predictors = pd.read_csv("TrainingPredictors.csv")

# clean data
predictors['DELIVERY_START'] = pd.to_datetime(predictors['DELIVERY_START'], utc=True)
tf_floats = predictors.dtypes[predictors.dtypes == 'float'].index
diff = (predictors[tf_floats].fillna(0) - predictors[tf_floats].fillna(0).astype(int)).sum()
predictors[diff[diff == 0].index] = predictors[diff[diff == 0].index].fillna(0).astype(int)

target = pd.read_csv("TrainingTarget.csv")

target['DELIVERY_START'] = pd.to_datetime(target['DELIVERY_START'], utc=True)

training = pd.merge(predictors, target, on='DELIVERY_START', how='inner')

training['non_renewable_need'] = training['load_forecast'] - training[
    ['nucelear_power_available',
     'wind_power_forecasts_average',
     'solar_power_forecasts_average']
].sum(axis=1)

training = training[['non_renewable_need', 'coal_power_available', 'gas_power_available', 'spot_id_delta']]

# remove outliers
training = training[abs(stats.zscore(training['spot_id_delta'])) <= 3]

X_train, X_test, y_train, y_test = train_test_split(
    training[['non_renewable_need', 'coal_power_available', 'gas_power_available']],
    training['spot_id_delta'],
    test_size=0.2)


