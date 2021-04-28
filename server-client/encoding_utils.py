import numpy as np
import base64
import cv2


def image_to_string(image: np.ndarray) -> str:
    return base64.b64encode(
        cv2.imencode('.jpg', image)[1]

    ).decode()


def string_to_image(s: str) -> np.ndarray:
    return cv2.imdecode(
        np.frombuffer(base64.b64decode(s), dtype=np.uint8),
        flags=1
    )
