import argparse
import cv2
import io
from base64 import b64encode
from PIL import Image
from ad_blaster.ad_blaster import AdBlaster


def ask_from_image(image_path):
    ad_blaster = AdBlaster("http://192.168.1.15:11434")
    image_content = image_to_base64(image_path)
    ad_blaster.ask(image_content)


def ask_from_webcam():
    ad_blaster = AdBlaster("http://192.168.1.15:11434")
    fetch_webcam_frame(0)
    frame_contents = image_to_base64("captured_image.jpg")
    ad_blaster.ask(frame_contents)


def calibrate(device_id=0):
    fetch_webcam_frame()


def fetch_webcam_frame(device_id=0):
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    if not cap.isOpened():
        raise Exception(f"could not open camera {device_id}")
    success, frame = cap.read()
    cv2.imwrite("captured_image.jpg", frame)
    print("Image captured and saved as captured_image.jpg")
    if not success:
        raise Exception(f"Failed to capture image from {device_id}")
    cap.release()

    return frame


def image_to_base64(image_path):
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        img_string = b64encode(buffered.getvalue()).decode("utf-8")
    return img_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AdBlaster with an image.")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument("--webcam", action="store_true", help="use webcam mode")
    parser.add_argument("--calibrate", action="store_true", help="calibrate the webcam")
    args = parser.parse_args()

    if not args.image_path and not args.webcam and not args.calibrate:
        print(parser.usage)

    if args.image_path:
        ask_from_image(args.image_path)

    if args.calibrate:
        calibrate()

    if args.webcam:
        ask_from_webcam()
