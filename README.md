# machine learning model (training)
To use it, just docker build and run in the project directory:

    docker build -t ml_train:latest .
    docker run -v "/absolute/path/of/the/desired/directory/on/local:/app/trained_models"  ml_train

Note: The package train and serialize models using sklearn, as included in requirements.txt file, and is supposed to be deserialized and read by the recommendation engine. Therefore, it's crucial to keep the versions of sklearn in the two packages the same. A Mismatch have caused failure to read models in development previously.  
