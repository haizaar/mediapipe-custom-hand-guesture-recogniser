from typing import Optional
from enum import Enum
from pathlib import Path
import numpy as np
import tensorflow as tf

import structlog


logger = structlog.get_logger(__name__)
print(Path(__file__))
model_path: Path = Path(__file__).with_suffix(".tflite")
model_hdf5_path: Path = Path(__file__).with_suffix(".hdf5")
training_data_path: Path = Path(__file__).with_name("training_data.csv")


Confidence = float


class Gesture(Enum):
    rock = 0
    paper = 1
    scissors = 2
    thumbs_up = 3


class GestureClassifier(object):
    def __init__(self, model_path: Path = model_path, num_threads: int = 1):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path),
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def classify(self, landmark_list) -> tuple[Optional[Gesture], Confidence]:
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        weights = np.squeeze(result)
        logger.debug(weights=weights)

        result_index = np.argmax(weights)

        return Gesture(result_index), weights[result_index]
