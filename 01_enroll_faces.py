import cv2 #test
import face_recognition
import pickle
import os

PERSON_NAME = input("Enter your name in English: ")
PERSON_PREFERENCE = input(f"Enter {PERSON_NAME}'s favorite drink: ")
NUM_SAMPLES = 20  
DATABASE_FILE = "face_database.pkl" 

output_folder = f"dataset/{PERSON_NAME}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("\n[INFO] Preparing camera...")
cap = cv2.VideoCapture(0)
print(f"[INFO] Look at camera, we will collect {NUM_SAMPLES} photos of your face.")
print("change your expression, look left/right, smile, etc. to get diverse samples.")

count = 0
while count < NUM_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] could not read from camera. Please check your camera connection.")
        break

    cv2.putText(frame, f"Collecting sample {count+1}/{NUM_SAMPLES}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Face Enrollment", frame)

    boxes = face_recognition.face_locations(frame, model='hog') # cnn is more accurate but need dlib with gpu support

    if len(boxes) == 1: #ensure we only capture one face
        file_path = os.path.join(output_folder, f"{PERSON_NAME}_{count}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"已保存: {file_path}")
        count += 1
        cv2.waitKey(200) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n[INFO] finished collecting samples.")
cap.release()
cv2.destroyAllWindows()


print("[INFO] Processing collected images to create face embeddings...")

known_encodings = []
image_paths = [os.path.join(output_folder, f) for f in os.listdir(output_folder)]

for image_path in image_paths:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)

if known_encodings:
    prototype_encoding = sum(known_encodings) / len(known_encodings)

    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'rb') as f:
            database = pickle.load(f)
    else:
        database = {}

    database[PERSON_NAME] = {
        "embedding": prototype_encoding,
        "preference": PERSON_PREFERENCE
    }

    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(database, f)

    print(f"[SUCCESS] {PERSON_NAME} has been enrolled successfully with preference: {PERSON_PREFERENCE}.")
else:
    print("[ERROR] No valid face encodings found. Please try again with clearer images.")