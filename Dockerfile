# set base image (host OS)
FROM python:3.8

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip3 install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ .

# Precompile python code for performance
RUN python3 -m compileall .

ENV MODELS_DIR=/home/chuck/folder/data_mining/ml_model/trained_models

ENV MONGO_URI=mongodb://localhost/tzbackend

EXPOSE 5000

# command to run on container start
CMD [ "python3", "./app.py" ]
