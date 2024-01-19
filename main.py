import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def clean_df(df):

    floats = df.dtypes[df.dtypes == 'float'].index

    df['DELIVERY_START'] = pd.to_datetime(df['DELIVERY_START'], utc=True)

    diff = (df[floats].fillna(0) - df[floats].fillna(0).astype(int)).sum()

    df[diff[diff == 0].index] = df[diff[diff == 0].index].fillna(0).astype(int)

    df[diff[diff == 1].index] = df[diff[diff == 1].index].fillna(0)

    return df


training = pd.merge(
    clean_df(pd.read_csv("TrainingPredictors.csv")),
    clean_df(pd.read_csv("TrainingTarget.csv")),
    on='DELIVERY_START',
    how='inner').drop(['DELIVERY_START', 'predicted_spot_price'], axis=1).fillna(0)

training['spot_id_delta'] = (training['spot_id_delta'] > 0).astype(int)

# remove outlier days, where no load was needed or no service was available

training = training[
    ~(training[
          ['load_forecast',
           'coal_power_available',
           'gas_power_available',
           'nucelear_power_available']
      ] == 0).any(axis=1)
]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    training.iloc[:, :-1],
    training.iloc[:, -1],
    test_size=0.2)

# Normalize predictor values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential([
    Dense(64, input_shape=(X_train_scaled.shape[1],)),
    Dense(256, activation='relu'),
    Dense(512, activation='selu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

nn_model.fit(
    X_train_scaled,
    y_train,
    epochs=1000,
    validation_split=0.2
)

y_pred = (nn_model.predict(X_test_scaled) > 0.5).astype(int)

accuracy_score(y_test, y_pred)
