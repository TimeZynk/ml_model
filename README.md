# machine learning model (training)
How to use: docker build and run
docker build -t ml_train:latest .
docker run -v "/absolute/path/of/the/desired/directory/on/local:/app/trained_models"  ml_train
