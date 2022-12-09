# pkill -KILL -u polyamatyas
# watch -n 0 nvidia-smi
##
import numpy as np
import cv2
import pandas as pd
import os
import pickle


##
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


##
def generate_dataset(image_size, amfed_path, save_path):
    #amfed_path = r"/home/polyamatyas/projects/mosoly/AMFEDPLUS_Distribution/"
    dir_list = os.listdir(amfed_path + "Landmark Points (labeled videos)")
    files = [text.split(".")[0] for text in dir_list if (os.path.getsize(
        amfed_path + "Landmark Points (labeled videos)/" + text) != 0)]

    ##

    file_inputs = dict()
    file_negative_positive = dict()

    count = 0
    for file in files:
        label_csv = pd.read_csv(amfed_path + "AU Labels/" + file + "-label.csv")
        len_label = label_csv.shape[0]
        label_idx = 0

        input_images = []
        input_labels = []

        landmarks = pd.read_csv(amfed_path + "Landmark Points (labeled videos)/" + file + ".csv")
        cap = cv2.VideoCapture(amfed_path + "Videos - FLV (labeled)/" + file + ".flv")

        frame_number = 0
        number_of_positives = 0
        number_of_negatives = 0
        while cap.isOpened():
            frame_exists, image = cap.read()
            if not frame_exists:
                break
            try:
                left_eye = (int((np.float64(landmarks.pt_affdex_tracker_34[frame_number]) + np.float64(
                    landmarks.pt_affdex_tracker_32[frame_number]) +
                                 np.float64(landmarks.pt_affdex_tracker_60[frame_number]) + np.float64(
                            landmarks.pt_affdex_tracker_62[frame_number])) / 4),
                            int((np.float64(landmarks.pt_affdex_tracker_35[frame_number]) + np.float64(
                                landmarks.pt_affdex_tracker_33[frame_number]) +
                                 np.float64(landmarks.pt_affdex_tracker_61[frame_number]) + np.float64(
                                        landmarks.pt_affdex_tracker_63[frame_number])) / 4))

                right_eye = (int((np.float64(landmarks.pt_affdex_tracker_36[frame_number]) + np.float64(
                    landmarks.pt_affdex_tracker_64[frame_number]) +
                                  np.float64(landmarks.pt_affdex_tracker_38[frame_number]) + np.float64(
                            landmarks.pt_affdex_tracker_66[frame_number])) / 4),
                             int((np.float64(landmarks.pt_affdex_tracker_37[frame_number]) + np.float64(
                                 landmarks.pt_affdex_tracker_65[frame_number]) +
                                  np.float64(landmarks.pt_affdex_tracker_39[frame_number]) + np.float64(
                                         landmarks.pt_affdex_tracker_67[frame_number])) / 4))

                angle = angle_between((1, 0), (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])) * 180 / 3.141592
                if left_eye[1] > right_eye[1]:
                    angle = -angle
                distance = int(np.linalg.norm((right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])))

                rotate_matrix = cv2.getRotationMatrix2D(center=left_eye, angle=angle, scale=1)
                rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))

                ratio = distance / 100
                rectangle = ((int(left_eye[0] - 75 * ratio), int(left_eye[1] - 70 * ratio)),
                             ((int(left_eye[0] + 175 * ratio), int(left_eye[1] + 180 * ratio))))

                cropped_image = rotated_image[rectangle[0][1]:rectangle[1][1], rectangle[0][0]:rectangle[1][0]]
                downsampled_image = cv2.resize(cropped_image, (image_size, image_size))

                if (len_label > label_idx + 1 and label_csv.iloc[label_idx + 1, 0] * 1000 < landmarks.iloc[
                    frame_number, 0]):  # TimeStamp(msec) csak nem jeleníti meg valamiért
                    label_idx = label_idx + 1

                if (np.float64(label_csv.iloc[label_idx, 1]) == 0):
                    input_labels.append(0)
                    number_of_negatives += 1
                else:
                    input_labels.append(1)
                    number_of_positives += 1
                input_images.append(downsampled_image)

            except Exception as e:
                # print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
                pass

            frame_number = frame_number + 1
        file_negative_positive[file] = (number_of_negatives, number_of_positives)
        file_inputs[file] = (input_images, input_labels)

        count = count + 1
        print(str(count) + "/" + str(len(files)))
    # if count == 2:
    # 	break

    file = open(save_path + 'amfed_data_' + str(image_size) + '.pkl', 'wb')
    pickle.dump(file_inputs, file, protocol=4)
    file.close()

    file = open(save_path + 'amfed_negative_positive_ratio.pkl', 'wb')
    pickle.dump(file_negative_positive, file)
    file.close()
