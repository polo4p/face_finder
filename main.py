import dlib
import PIL.Image
import numpy as np
from pathlib import Path
import argparse
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--folder_path", type=str, default = 'processed',
    help="Path to the folder of export")
ap.add_argument("-m", "--model", type=str, default="SVM",
	help="CNN or SVM")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())

print('[INFO] Importing pretrained models...')
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
if args["model"] == 'CNN':
    face_detector = dlib.cnn_face_detection_model_v1("pretrained_model/mmod_human_face_detector.dat") 
else :
    face_detector = dlib.get_frontal_face_detector()
print('[INFO] Finished models import')

def image_importer(path:Path):
    """Import all png and jpg images from a folder."""
    list_imgs = [img for img in path.rglob('*.jpg')]
    for image in path.rglob('*.png'):
        list_imgs.append(image)
    if len(list_imgs)==0:
        raise Exception(f"No image detected at {path}")
    return(list_imgs)

def encode_face(image):
    """Encode the face data as an array for similarity measure."""
    # ML to know where the faces are
    face_locations = face_detector(image, args["upsample"])
    face_encodings_list = []
    for face_location in face_locations:
        # ML to encode whose face it is
        if args["model"] == 'CNN':
            face_location = face_location.rect
        shape = pose_predictor_68_point(image, face_location)
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))
    return face_encodings_list

def sort_by_face(frame, known_face_encodings, known_face_names):
    """Fill the folder with the images sorted by faces."""
    # Encoding faces
    frame = PIL.Image.open(picture)
    frame = np.array(frame)
    face_encodings_list = encode_face(frame)
    for face_encoding in face_encodings_list:
        # Check distances between the detected and known faces
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        faces_detected=vectors<= tolerance
        if faces_detected.any():
            name = known_face_names[np.argmin(vectors)]
            shutil.copy(picture,Path(args["folder_path"]+'/'+name))

if __name__=='__main__':
    print('[INFO] Importing faces')
    # Importing data from dir
    list_faces = image_importer(Path('faces'))
    list_names = [os.path.splitext(os.path.basename(face_name))[0] for face_name in list_faces]
    list_pictures = image_importer(Path('to_process'))
    picture_names = [os.path.basename(picture_name) for picture_name in list_pictures]
    # Encoding faces data
    faces_encodings = []
    for face in list_faces:
        image = PIL.Image.open(face)
        image = np.array(image)
        face_encoded = encode_face(image)[0]
        faces_encodings.append(face_encoded)
    print(f'[INFO] {len(faces_encodings)} faces imported') 
    # Creating the folders
    os.makedirs(args["folder_path"], exist_ok=True)
    for name in list_names:
        os.makedirs(Path(args["folder_path"] +'/' + name), exist_ok=True)
    # Outputting the pictures
    pic_amount = len(list_pictures)
    processed_amount = 0
    print('[INFO] Beginning sorting')
    for picture in list_pictures: 
        sort_by_face(picture, faces_encodings, list_names)
        processed_amount += 1
        print('\033[A                                                    \033[A')
        print(f'Processed {processed_amount} / {pic_amount} pictures')
        
            
    
