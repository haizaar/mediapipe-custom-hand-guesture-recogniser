import csv
import copy
import argparse
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
import mediapipe_gesture_classifier as mgc

import uberlogging
import structlog

logger = structlog.get_logger(__name__)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument('--use_static_image_mode', action='store_true', default=False)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.6)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    uberlogging.configure()
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    classifier = mgc.GestureClassifier()

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)


    training_data_file = open(mgc.training_data_path, 'a', newline="")
    training_data_writer = csv.writer(training_data_file)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            training_data_file.close()
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Put e.g. "right" on the right
        # FIXME: Why to create a copy?
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                landmark_list = classifier.calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = classifier.pre_process_landmark(landmark_list)

                if mode == 1:
                    logging_csv(training_data_writer, number, pre_processed_landmark_list)

                gesture, confidence = classifier.classify(pre_processed_landmark_list)

                brect_array = classifier.calc_bounding_rect(debug_image, hand_landmarks)
                x, y, w, h = cv.boundingRect(brect_array)
                brect = [x, y, x + w, y + h]
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                mp.solutions.drawing_utils.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    gesture,
                    confidence,
                )

        debug_image = draw_info(debug_image, fps, mode, number)

        # 画面反映 #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def logging_csv(writer, number, landmark_list):
    if 0 <= number <= 9:
        writer.writerow([number, *landmark_list])


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # bounding rectangle
        cv.rectangle(image, (int(brect[0]*.95), int(brect[1]*.95)), (int(brect[2]*1.05), int(brect[3]*1.05)),
                     (255, 0, 255), 3)

    return image


def draw_info_text(image, brect, handedness, gesture: mgc.Gesture, confidence: mgc.Confidence):
    cv.rectangle(image, (int(brect[0]*.95)-2, int(brect[1]*.95)), (int(brect[2]*1.05)+2, int((brect[1]-22)*.95) - 30),
                 (255, 0, 255), -1)

    info_text = handedness.classification[0].label[0:][0]
    if gesture:
        info_text +=  f": {gesture.name} {confidence:.2f}"
        cv.putText(image, info_text, (int(brect[0]*.95) + 5, int(brect[1]*.95) - 8),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
