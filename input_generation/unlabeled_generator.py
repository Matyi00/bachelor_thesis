import numpy as np
import cv2
import mediapipe as mp
import pickle
import dlib
from imutils import face_utils
import imutils


##
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


##
def generate_dataset(image_size, image_list, dlib_shape_predictor_path, save_path):
    input_images = []
    count = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_shape_predictor_path)
    mp_face_mesh = mp.solutions.face_mesh
    for file in image_list:
        original_image = cv2.imread(file)
        image = original_image
        image = imutils.resize(image, width=500)

        with mp_face_mesh.FaceMesh(
                max_num_faces=1,
                # refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            mp_face_mesh = mp.solutions.face_mesh
            shape_y, shape_x = image.shape[:2]
            landmark_scaling = np.array([shape_x, shape_y, shape_x])

            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:
                    landmarks = [[l.x, l.y, l.z] for l in face.landmark]

                landmarks = np.array(landmarks) * landmark_scaling

                left_eye = (landmarks[33] + landmarks[246] + landmarks[161] + landmarks[160] + landmarks[159] +
                            landmarks[
                                158] + landmarks[157] + landmarks[173] + landmarks[133] + landmarks[155] + landmarks[
                                154] + landmarks[
                                153] + landmarks[145] + landmarks[144] + landmarks[163] + landmarks[7]) // 16
                right_eye = (landmarks[362] + landmarks[398] + landmarks[384] + landmarks[385] + landmarks[386] +
                             landmarks[
                                 387] + landmarks[388] + landmarks[466] + landmarks[263] + landmarks[249] + landmarks[
                                 390] + landmarks[
                                 373] + landmarks[374] + landmarks[380] + landmarks[381] + landmarks[382]) // 16

            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)

                for (i, rect) in enumerate(rects):
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    left_eye = (shape[36] + shape[37] + shape[38] + shape[39] + shape[40] + shape[41]) // 6
                    right_eye = (shape[42] + shape[43] + shape[44] + shape[45] + shape[46] + shape[47]) // 6

        image = original_image
        image = imutils.resize(image, width=500)
        left_eye_desired_position = (150, 150)
        height, width = image.shape[:2]
        T = np.float32(
            [[1, 0, left_eye_desired_position[0] - left_eye[0]], [0, 1, left_eye_desired_position[1] - left_eye[1]]])
        img_translation = cv2.warpAffine(image, T, (width, height))
        angle = angle_between((1, 0), (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])) * 180 / 3.141592
        if left_eye[1] > right_eye[1]:
            angle = -angle

        rotate_matrix = cv2.getRotationMatrix2D(center=left_eye_desired_position, angle=angle,
                                                scale=100 / np.linalg.norm(right_eye - left_eye))
        rotated_image = cv2.warpAffine(src=img_translation, M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))

        rectangle = ((75, 80), (325, 330))
        cropped_image = rotated_image[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]]

        downsampled_image = cv2.resize(cropped_image, (image_size, image_size))
        input_images.append(downsampled_image)

        count = count + 1
        #if count % 50 == 0:
        print(str(count) + "/" + str(len(image_list)))

    file = open(save_path + "unlabeled_data_" + str(image_size) + '.pkl', 'wb')
    pickle.dump(input_images, file, protocol=4)
    file.close()

