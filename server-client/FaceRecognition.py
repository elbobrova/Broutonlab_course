import face_recognition
from sklearn import svm
import os
import cv2


class FaceRecognition(object):

    def __init__(self):
        """Constructor"""

    def train(self, image):
        # Training the SVC classifier
        # The training data would be all the face encodings from all the known images and the labels are their names
        encodings = []
        names = []

        # Training directory
        train_dir = os.listdir('dataset/')

        # Loop through each person in the training directory
        for person in train_dir:
            pix = os.listdir("dataset/" + person)

            # Loop through each training image for the current person
            for person_img in pix:
                # Get the face encodings for the face in each image file
                face = face_recognition.load_image_file("dataset/" + person + "/" + person_img)
                face_bounding_boxes = face_recognition.face_locations(face)
                # If training image contains exactly one face
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
                    # Add face encoding for current image with corresponding label (name) to the training data
                    encodings.append(face_enc)
                    names.append(person)
                else:
                    print(person + "/" + person_img + " was skipped and can't be used for training")
        clf = svm.SVC(gamma='scale')
        clf.fit(encodings, names)
        face_locations = face_recognition.face_locations(image)
        no = len(face_locations)
        test_image_enc = face_recognition.face_encodings(image)[0]
        name = clf.predict([test_image_enc])
        return (name)

