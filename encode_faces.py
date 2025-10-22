import face_recognition
import os
import numpy as np
from sklearn.neighbors import KDTree
import pickle
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# ממיר תמונה לקידוד הפנים שלה
def encode_known_faces(criminals_faces_dir, output_file="encoded_faces.pkl"):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(criminals_faces_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(criminals_faces_dir, filename)
            try:
                image = face_recognition.load_image_file(filepath)
                face_encodings_list = face_recognition.face_encodings(image)
                if face_encodings_list:
                    face_encoding = face_encodings_list[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(os.path.splitext(filename)[0])
                    print(f"קידוד פנים: {os.path.splitext(filename)[0]}")
                else:
                    print(f"לא נמצאו פנים בתמונה: {filename}")
            except Exception as e:
                print(f"שגיאה בקידוד פנים: {filename} - {e}")

    encoded_data = {"encodings": np.array(known_face_encodings), "names": np.array(known_face_names)}

    with open(output_file, 'wb') as f:
        pickle.dump(encoded_data, f)

    print(f"\nקידודים נשמרו בקובץ: {output_file}")

if __name__ == '__main__':
    criminals_faces_dir = r"C:\Users\moiam\Documents\Project\dataSet\פושעים" # שנה לנתיב שלך
    output_file = "encoded_faces.pkl"
    encode_known_faces(criminals_faces_dir, output_file)