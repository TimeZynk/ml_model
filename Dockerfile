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

RUN mkdir /app/trained_models

# ENV MONGO_URI="mongodb://192.168.1.20/tzbackend"
# ENV MODELS_DIR="/app/trained_models"
# ENV HOUR=17
# ENV MINUTE=54
# ENV SECOND=0

# command to run on container start
CMD [ "python3", "./app.py" ]
