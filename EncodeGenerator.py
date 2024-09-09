import cv2
from deepface import DeepFace
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://school-attendance-dc527-default-rtdb.firebaseio.com/",
     'storageBucket': "school-attendance-dc527.appspot.com"
})


#import the mode images
folderPath = 'Images'
PathList = os.listdir(folderPath)
print(PathList)
imgList = []
studentIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0])
    #print(path)
   # print(os.path.splitext(path)[0])

fileName = f'{folderPath}/{path}'
#fileName = os.path.join(folderPath, path)
bucket = storage.bucket()
blob = bucket.blob(fileName)
blob.upload_from_filename(fileName)

print(studentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Extract facial embeddings using DeepFace's represent function
        encode = DeepFace.represent(img, model_name='Facenet')[0]["embedding"]
        encodeList.append(encode)
    return encodeList

print("Encoding started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithids = [encodeListKnown, studentIds]
print("Encoding Complete.")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithids, file)
file.close()
print("File Saved")
