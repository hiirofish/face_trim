import urllib.request
import bz2
import os

def download_landmarks_model():
    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    file_name = "shape_predictor_68_face_landmarks.dat.bz2"
    
    print("Downloading shape_predictor_68_face_landmarks.dat.bz2...")
    urllib.request.urlretrieve(url, file_name)
    
    print("Extracting...")
    with bz2.BZ2File(file_name) as fr, open("shape_predictor_68_face_landmarks.dat", "wb") as fw:
        fw.write(fr.read())
    
    os.remove(file_name)
    print("Download and extraction complete.")

# スクリプトを実行
download_landmarks_model()
