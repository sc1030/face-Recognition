import os
import pickle
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cvzone
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime

# Firebase initialization
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://school-attendance-dc527-default-rtdb.firebaseio.com/",
    'storageBucket': "school-attendance-dc527.appspot.com"
})

bucket = storage.bucket()

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera or adjust as needed
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

# Initialize MTCNN and InceptionResnetV1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load known faces
print("Loading Encode File ...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Load mode images
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

modeType = 0
counter = 0
id = -1
imgStudent = []

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Check your camera or video source.")
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and get face embeddings
    faces = mtcnn.detect(imgS)
    encodeCurFrame = []

    if faces[0] is not None:
        boxes = faces[0]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face = imgS[y1:y2, x1:x2]

            # Ensure face has proper dimensions for InceptionResnetV1
            if face.shape[0] >= 160 and face.shape[1] >= 160:
                face = cv2.resize(face, (160, 160))  # Resize face to 160x160 as required by InceptionResnetV1
                face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

                with torch.no_grad():
                    encoding = model(face_tensor).cpu().numpy()[0]
                encodeCurFrame.append(encoding)

        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        if encodeCurFrame:
            for encodeFace in encodeCurFrame:
                matches = []
                faceDis = []

                for knownEncode in encodeListKnown:
                    distance = np.linalg.norm(encodeFace - knownEncode)
                    faceDis.append(distance)
                    matches.append(distance < 0.6)

                if matches:
                    matchIndex = np.argmin(faceDis)
                    if matches[matchIndex]:
                        id = studentIds[matchIndex]
                        y1, x2, y2, x1 = boxes[matchIndex]
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                        imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                        if counter == 0:
                            counter = 1
                            modeType = 1

                        if counter == 1:
                            studentInfo = db.reference(f'Students/{id}').get()
                            if studentInfo:
                                print(studentInfo)
                            else:
                                print(f"Student ID {id} not found in the database.")
                                counter = 0
                                continue

                            counter += 1

                            if counter == 2:
                                modeType = 2
                                blob = bucket.get_blob(f'Images/{id}.png')
                                array = np.frombuffer(blob.download_as_string(), np.uint8)
                                imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
                                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()

                                if secondsElapsed > 30:
                                    ref = db.reference(f'Students/{id}')
                                    studentInfo['total_attendance'] += 1
                                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                else:
                                    modeType = 3
                                    counter = 0
                                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                            if modeType != 3:
                                if 10 < counter < 20:
                                    modeType = 2

                                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]

                                if counter <= 10:
                                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                                    cv2.putText(imgBackground, str(id), (1006, 493),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                                    offset = (414 - w) // 2
                                    cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                                counter += 1

                                if counter >= 20:
                                    counter = 0
                                    modeType = 0
                                    studentInfo = []
                                    imgStudent = []
                                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]
        else:
            modeType = 0
            counter = 0

        cv2.imshow("Face Attendance", imgBackground)
        cv2.waitKey(1)
