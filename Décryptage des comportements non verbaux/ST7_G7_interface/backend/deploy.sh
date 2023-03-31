apt-get install libsndfile1
pip install -r requirements.txt
gdown https://drive.google.com/uc?id=1-1Zn60-udyh_JIA4TALjvNCfFin8jAnu
mkdir data
mkdir data/split_audio
gunicorn --bind=0.0.0.0 --timeout 600 app:app
