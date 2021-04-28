from argparse import ArgumentParser
import cv2
import requests
from encoding_utils import image_to_string
from PIL import Image
import numpy as np

def parse_args():
    parser = ArgumentParser(description='Client')
    parser.add_argument('--image', required=True, type=str)
    parser.add_argument('--ip', required=False, type=str, default='0.0.0.0')
    parser.add_argument('--port', required=False, type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        print('Can\'t open image')
        return

    request_json = {
        'image': image_to_string(image)
    }
    response = requests.post(
        url='http://{}:{}/api/classification/imagenet'.format(
            args.ip,
            args.port
        ),
        json=request_json
    )
    if response.status_code != 200:
        print('Bad request')
    print('{}'.format(response.json()))


if __name__ == '__main__':
    main()
