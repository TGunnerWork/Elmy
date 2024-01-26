import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from keras import Input, Sequential
from keras.layers import Dense
from keras.utils import to_categorical

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def convert_date(df):

    df['DELIVERY_START'] = pd.to_datetime(df['DELIVERY_START'], utc=True)

    return df


def coal_levels(coal_output):
    if coal_output < 2500:
        return "Low"
    elif coal_output < 3000:
        return "Medium"
    else:
        return "High"


def gas_levels(gas_output):
    if gas_output < 10900:
        return "Low"
    elif gas_output < 11200:
        return "Medium"
    elif gas_output < 11700:
        return "High"
    else:
        return "Max"


def clean_df(df):

    floats = df.dtypes[df.dtypes == 'float'].index

    diff = (df[floats].fillna(0) - df[floats].fillna(0).astype(int)).sum()

    df[diff[diff == 0].index] = df[diff[diff == 0].index].fillna(0).astype(int)

    df[diff[diff == 1].index] = df[diff[diff == 1].index].fillna(0)

    df = df[~(df[['load_forecast', 'coal_power_available', 'gas_power_available',
                  'nucelear_power_available']] == 0).any(axis=1)]

    df['coal_power_level'] = df['coal_power_available'].apply(coal_levels)
    df['gas_power_level'] = df['gas_power_available'].apply(gas_levels)

    df['coal_power_level'] = pd.Categorical(
        df['coal_power_level'],
        categories=df['coal_power_level'].unique()
    )

    df['gas_power_level'] = pd.Categorical(
        df['gas_power_level'],
        categories=df['gas_power_level'].unique()
    )

    df['coal_power_available'] = df['coal_power_level']
    df['gas_power_available'] = df['gas_power_level']

    return df.drop(['predicted_spot_price', 'coal_power_level', 'gas_power_level'], axis=1).dropna()


def delta_size(spot_id_delta):
    message = []
    if abs(spot_id_delta) < 23:
        message += ['low']
    else:
        message += ['high']
    if spot_id_delta >= 0:
        message += ['pos']
    else:
        message += ['neg']
    return ' '.join(message)


target = convert_date(pd.read_csv("TrainingTarget.csv"))
target['spot_id_direction'] = (target['spot_id_delta'] > 0).astype(int)
target['spot_id_vec'] = target['spot_id_delta'].apply(delta_size)
target['spot_id_vec'] = pd.Categorical(
    target['spot_id_vec'],
    categories=['high neg', 'low neg', 'low pos', 'high pos'],
    ordered=True)

training = pd.merge(
    convert_date(clean_df(pd.read_csv("TrainingPredictors.csv"))),
    target,
    on='DELIVERY_START',
    how='inner').drop('DELIVERY_START', axis=1)

training = training[abs(zscore(training['spot_id_delta'])) < 3]

# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    training.iloc[:, :-(target.shape[1]-1)],
    training.iloc[:, -(target.shape[1]-1):],
    test_size=0.2)

X_train_enc = pd.get_dummies(X_train, columns=['coal_power_available', 'gas_power_available'], drop_first=True)
X_test_enc = pd.get_dummies(X_test, columns=['coal_power_available', 'gas_power_available'], drop_first=True)

# Normalize predictor values

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_enc)
X_test_scaled = scaler.transform(X_test_enc)

# Binary Classifier
############################################################

params = {
    'colsample_bytree': 0.8,
    'gamma': 0.2,
    'learning_rate': 0.025,
    'max_depth': 9,
    'min_child_weight': 1,
    'n_estimators': 2350,
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,
    'subsample': 0.8
}

# Parameter Tuning

# grid = GridSearchCV(xgb.XGBClassifier(), param_grid=params, cv=10, scoring='accuracy', verbose=2)
# grid.fit(X_train_scaled, y_train['spot_id_delta_flag'])
# print(grid.best_params_)
# params = grid.best_params_

xgb_model_flag = xgb.XGBClassifier(**params)
xgb_model_flag.fit(X_train_scaled, y_train['spot_id_direction'])

accuracy_score(y_test['spot_id_direction'], xgb_model_flag.predict(X_test_scaled))

# 4 Category Classifier
#############################################

params = {
    'colsample_bytree': 0.8,
    'gamma': 0.2,
    'learning_rate': 0.025,
    'max_depth': 9,
    'min_child_weight': 1,
    'n_estimators': 2350,
    'objective': 'binary:logistic',
    'subsample': 0.8
}

# label encoding

xgb_labels = LabelEncoder()

xgb_y = xgb_labels.fit_transform(y_train['spot_id_vec'])

# param tuning

r = np.arange(-1, 2)

grid_params = {
    'colsample_bytree': [0.8],
    'gamma': 0.2+0.1*r,
    'learning_rate': 0.02+0.01*r,
    'max_depth': 9+r,
    'min_child_weight': [1],
    'n_estimators': 2000+300*r,
    'objective': ['binary:logistic'],
    'subsample': [0.8]
}

grid = GridSearchCV(xgb.XGBClassifier(), param_grid=grid_params, cv=10, scoring='accuracy', verbose=2)
grid.fit(X_train_scaled, xgb_y)
print(grid.best_params_)
params = grid.best_params_

xgb_model_cats = xgb.XGBClassifier(**params)
xgb_model_cats.fit(X_train_scaled, xgb_y)

# True Accuracy
accuracy_score(xgb_labels.transform(y_test['spot_id_vec']), xgb_model_cats.predict(X_test_scaled))

# Sign Accuracy (both positive, both negative)
accuracy_score(xgb_labels.transform(y_test['spot_id_vec']) % 2, xgb_model_cats.predict(X_test_scaled) % 2)

# Neural Network - 4 Cat
###########################################

nn_labels = LabelEncoder()
nn_y = nn_labels.fit_transform(y_train['spot_id_vec'])
y_one_hot = to_categorical(nn_y, num_classes=4)

nn_y_test = to_categorical(nn_labels.transform(y_test['spot_id_vec']))

# NN model
nn_model_cats = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(1024, activation='relu'),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

nn_model_cats.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['precision', 'recall', 'accuracy']
)

history = nn_model_cats.fit(
    X_train_scaled,
    y_one_hot,
    epochs=150,
    validation_data=(
        X_test_scaled,
        nn_y_test
    )
)

sns.heatmap(
    confusion_matrix(
        y_test['spot_id_vec'],
        nn_labels.inverse_transform(
            np.argmax(nn_model_cats.predict(X_test_scaled), axis=1)
        ),
        labels=['high neg', 'low neg', 'low pos', 'high pos']),
    annot=True,
    xticklabels=['high neg', 'low neg', 'low pos', 'high pos'],
    yticklabels=['high neg', 'low neg', 'low pos', 'high pos']
)
