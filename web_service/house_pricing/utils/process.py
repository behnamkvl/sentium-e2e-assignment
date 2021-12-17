from rest_framework import status
import pandas as pd
import logging
from datetime import datetime

from .utils import (
    metadata,
    TRAIN_DATE,
    ecode_input,
    reloaded_model,
)
logger = logging.getLogger(__name__)

def predict_house_price(query_params):
    try:
        address = query_params.get('address').lower().strip()
        _type = query_params.get('type').lower().strip()
        bedrooms = query_params.get('bedrooms').lower().strip()
        latitude = query_params.get('latitude').lower().strip()
        longitude = query_params.get('longitude').lower().strip()
        area = query_params.get('area').lower().strip()
        tenure = query_params.get('tenure').lower().strip()
        is_newbuild = query_params.get('is_newbuild').lower().strip()
        date = query_params.get('date').lower().strip()
    
    
        try:
            if _type not in metadata['valid_values']['type']:
                return 'type parameter with value: {} is not valid, valid values are: {}'.format(_type, metadata['valid_values']['type']), status.HTTP_400_BAD_REQUEST
            if area not in metadata['valid_values']['area']:
                return 'area parameter with value: {} is not valid, valid values are: {}'.format(area, metadata['valid_values']['area']), status.HTTP_400_BAD_REQUEST
                
            age_in_days = (TRAIN_DATE - datetime.strptime(date[:10], '%Y-%m-%d')).days
            
            input_dict = {
                'street_name': address.split(',')[-3].strip(),
                'type': _type,
                'bedrooms': int(bedrooms),
                'latitude': float(latitude),
                'longitude': float(longitude),
                'area': area,
                'tenure': tenure,
                'is_newbuild': int(is_newbuild),
                'age_in_days': age_in_days,
            }
            input = pd.DataFrame([input_dict])
            logger.info(f'input: {input}')
            input_transformed = ecode_input(input)
            
            logger.info(f'input_transformed: {input_transformed}')

            result = reloaded_model.predict(input_transformed)

            return {'estimated_price': int(result[0][0])}, status.HTTP_202_ACCEPTED
        except Exception as e:
            logger.error('error, {}'.format(str(e)))
            return 'error', status.HTTP_500_INTERNAL_SERVER_ERROR
    except Exception as e:
        logger.error('Error in predict_house_price, {}'.format(str(e)))
        return 'required parameters are not provided', status.HTTP_400_BAD_REQUEST
