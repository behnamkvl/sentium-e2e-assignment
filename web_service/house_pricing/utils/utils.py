import os
from datetime import datetime
from joblib import load
from tensorflow.keras.models import load_model
import json
import os


ML_PROJECT_DIR = os.environ.get('ML_MODEL_DIR')

SAVED_MODEL_DIR = os.path.join(ML_PROJECT_DIR, 'dnn_model')
HASH_ENCODER_DIR = os.path.join(ML_PROJECT_DIR, 'hash_encoder.joblib')
ONE_HOT_ENCODER_DIR = os.path.join(ML_PROJECT_DIR, 'one_hot_encoder.joblib')
METADATA_DIR = os.path.join(ML_PROJECT_DIR, 'metadata.json')

hash_encoder = load(HASH_ENCODER_DIR) 
one_hot_encoder = load(ONE_HOT_ENCODER_DIR)

def ecode_input(df):
  df = hash_encoder.transform(df)
  df = one_hot_encoder.transform(df)
  return df

with open(METADATA_DIR, 'r') as f:
    metadata = json.load(f)
TRAIN_DATE = datetime.strptime(metadata['train_date'], '%Y-%m-%d')

reloaded_model = load_model(SAVED_MODEL_DIR)
