from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework import permissions
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .utils.process import (
    predict_house_price
)


@swagger_auto_schema(methods=['GET'], manual_parameters=[
    openapi.Parameter('address',     openapi.IN_QUERY, description="House address", type=openapi.TYPE_INTEGER,  required=True),
    openapi.Parameter('type',        openapi.IN_QUERY, description="type",          type=openapi.TYPE_STRING,   required=True),
    openapi.Parameter('bedrooms',    openapi.IN_QUERY, description="bedrooms",      type=openapi.TYPE_NUMBER,   required=True),
    openapi.Parameter('latitude',    openapi.IN_QUERY, description="latitude",      type=openapi.TYPE_NUMBER,   required=True),
    openapi.Parameter('longitude',   openapi.IN_QUERY, description="longitude",     type=openapi.TYPE_NUMBER,   required=True),
    openapi.Parameter('area',        openapi.IN_QUERY, description="area",          type=openapi.TYPE_STRING,   required=True),
    openapi.Parameter('tenure',      openapi.IN_QUERY, description="tenure",        type=openapi.TYPE_STRING,   required=True),
    openapi.Parameter('is_newbuild', openapi.IN_QUERY, description="is_newbuild",   type=openapi.TYPE_INTEGER,  required=True),
    openapi.Parameter('date',        openapi.IN_QUERY, description="date",          type=openapi.TYPE_STRING,   required=True),
], tags=['Predict House Price'])
@api_view(['GET'])
def predict(request):
    query_params = request.query_params
    print('request.query_params:', query_params)
    result, res_status = predict_house_price(query_params)
    return Response({
        'result': result,
    }, status=res_status)
