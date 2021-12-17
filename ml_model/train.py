import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import category_encoders as ce
from category_encoders.hashing import HashingEncoder
from joblib import dump
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)


PROJECT_DIR = os.environ.get('HOME')
SAVED_MODEL_DIR = os.path.join(PROJECT_DIR, 'dnn_model')
HASH_ENCODER_DIR = os.path.join(PROJECT_DIR, 'hash_encoder.joblib')
ONE_HOT_ENCODER_DIR = os.path.join(PROJECT_DIR, 'one_hot_encoder.joblib')
METADATA_DIR = os.path.join(PROJECT_DIR, 'metadata.json')

# BigQuery connection info
BIGQUERY_CREDENTIALS = os.environ.get('BIGQUERY_CREDENTIALS')
PROJECT_ID = 'mystical-accord-330011'
DATASET_NAME = 'london_house_prices'
TABLE_NAME = 'london_house_prices'
TABLE_COLUMNS = ['address', 'type', 'bedrooms', 'latitude',
                 'longitude', 'area', 'price', 'tenure', 'is_newbuild', 'date']


def main():
    # reading data
    db = create_engine('bigquery://', credentials_path=BIGQUERY_CREDENTIALS)
    query = f"""
            select
            {','.join(TABLE_COLUMNS)}
            from `{PROJECT_ID}`.`{DATASET_NAME}`.`{TABLE_NAME}`
            """
    df = pd.read_sql(query, db)
    # Drop missing values and drop duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # cleaning data
    df['address'] = df['address'].apply(lambda s: s.lower())
    df['type'] = df['type'].apply(lambda s: s.lower())
    df['area'] = df['area'].apply(lambda s: s.lower())
    df['tenure'] = df['tenure'].apply(lambda s: s.lower())

    # feature extraction
    # extract feature from address
    df['street_name'] = df['address'].apply(lambda s: s.split(',')[-3].strip())
    # extract feature from date
    df['age_in_days'] = df['date'].apply(lambda d: (datetime.today(
    ) - datetime.strptime(d, '%Y-%m-%d %H:%M:%S%z').replace(tzinfo=None)).days)
    df.drop(['address', 'date'], axis=1, inplace=True)

    # Split the data into training and test sets
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('price')
    test_labels = test_features.pop('price')

    # Handle categoricals
    # hash encoder for steat name with high cardinality
    selected_columns_for_HashEncode = ['street_name']
    hash_encoder = HashingEncoder(
        cols=selected_columns_for_HashEncode, n_components=14, return_df=True, verbose=0)
    train_features = hash_encoder.fit_transform(train_features)
    # one hot encoder for columns with lower cardinality
    selected_columns_for_OneHot = ['type', 'tenure', 'area']
    one_hot_encoder = ce.OneHotEncoder(
        cols=selected_columns_for_OneHot, return_df=True)
    train_features = one_hot_encoder.fit_transform(train_features)

    def ecode_input(df):
        ''' encode the input dataframe based on defined encoders'''
        df = hash_encoder.transform(df)
        df = one_hot_encoder.transform(df)
        return df

    test_features = ecode_input(test_features)

    # train model
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    print(len((normalizer.mean.numpy())[0]))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=5,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    )

    def build_and_compile_model(norm):
        model = keras.Sequential([
            norm,
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    dnn_model = build_and_compile_model(normalizer)
    dnn_model.summary()

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        callbacks=[callback],
        verbose=1, epochs=100)

    # save model
    dnn_model.save(
        SAVED_MODEL_DIR,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )

    dump(hash_encoder, HASH_ENCODER_DIR)
    dump(one_hot_encoder, ONE_HOT_ENCODER_DIR)

    metadata = {}
    metadata['valid_values'] = {
        'area': sorted(df['area'].unique().tolist()),
        'type': sorted(df['type'].unique().tolist()),
        'tenure': sorted(df['tenure'].unique().tolist()),
    }
    metadata['columns'] = df.columns.tolist()
    metadata['train_date'] = str(datetime.today())[:10]

    with open(METADATA_DIR, 'w') as fp:
        json.dump(metadata, fp)


if __name__ == '__main__':
    main()
