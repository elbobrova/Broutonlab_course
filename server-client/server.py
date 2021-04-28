from flask import Flask, jsonify, request
from encoding_utils import string_to_image
from argparse import ArgumentParser
import ImageCaptionBaseline as imgcb
application = Flask(__name__)
@application.route('/api/classification/imagenet', methods=['POST', 'GET'])
def server_inference():
    """
    Request structure:
        'image': image in base64 string format
    Return json with class name:
        'class': class name string
    """
    if request.method == 'POST':
        request_data = request.get_json()
        image = string_to_image(request_data['image'])
        model = imgcb.Imagecaption()
        result = model.__call__(image)
        return jsonify({
            'result': result
        })
    else:
        return jsonify({
            '0': 0
        })

def args_parse():
    parser = ArgumentParser(description='Server')
    parser.add_argument('--ip', required=False, type=str, default='0.0.0.0')
    parser.add_argument('--port', required=False, type=int, default=5000)
    return parser.parse_args()


def main():
    args = args_parse()
    application.run(host=args.ip, debug=False, port=args.port)

if __name__ == '__main__':
    main()

