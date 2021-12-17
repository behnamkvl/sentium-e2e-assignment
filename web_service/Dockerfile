# pull official base image
FROM python:3.8.8

# set work directory
ENV HOME=/usr/app
RUN mkdir $HOME
WORKDIR $HOME

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install -r requirements.txt

# download the ML-model
RUN apt-get update && apt-get install -y unzip
RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1kq4BcMvjl8InNFD11TniuYT6cUQzajMV
RUN unzip predict_house_price_model.zip
ENV ML_MODEL_DIR=$HOME/predict_house_price_model

EXPOSE 8000

# copy project
COPY ./ ./

# run webserver
# RUN python manage.py runserver
