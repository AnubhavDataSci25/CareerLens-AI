import os
import sys
import joblib

from src.execption import CustomException

def save_objects(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)