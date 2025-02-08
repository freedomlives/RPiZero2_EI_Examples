#!/usr/bin/env python
# Using the model trained in Edge Impulse Studio, identify objects in the camera view
# and draw a circle and text above the object
# This is a modified version of the Edge Impulse example code for the Raspberry Pi Zero 2 W
# It was tested on DietPi 9.9.0 with the Edge Impulse SDK for python 1.1.0
# Usage: python classify_cam.py <path_to_model.eim>
# This file is based on the Edge Impulse example code from:
# https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/image/classify-image.py

import cv2
import os
import sys
import getopt
from edge_impulse_linux.image import ImageImpulseRunner
from picamera2 import Picamera2
import time

runner = None

min_confidence = 0.8 # The minimum confidence to classify an object

def help():
    print('python classify_cam.py <path_to_model.eim>')

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) != 1:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    # Initialize camera
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    # Using 640x480, as is the default when training using the Pi Camera in Edge Impulse Studio
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)  # Give camera time to warm up

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            # Reminding of the size of the images used to train the model
            print('Model input shape: %dx%d' % (
                model_info['model_parameters']['image_input_width'],
                model_info['model_parameters']['image_input_height']
            ))
            labels = model_info['model_parameters']['labels']

            # Capture full resolution image
            original_img = picam2.capture_array()
            # Convert to RGB (Picamera2 outputs BGR by default)
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Get features from image (this will resize internally)
            features, cropped = runner.get_features_from_image_auto_studio_setings(img_rgb)

            # Get scaling factors (the coordinates of the circles are in the dimensions of the cropped & scaled image)
            scale_x = original_img.shape[1] / cropped.shape[1]
            scale_y = original_img.shape[0] / cropped.shape[0]

            # Classify features
            res = runner.classify(features)

            if "classification" in res["result"].keys():
                print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                for label in labels:
                    score = res['result']['classification'][label]
                    print('%s: %.2f\t' % (label, score), end='')
                print('', flush=True)

            elif "bounding_boxes" in res["result"].keys():
                print('Found %d objects (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                for bb in res["result"]["bounding_boxes"]:
                    print('\t%s (%.2f): x=%d y=%d' % (bb['label'], bb['value'], bb['x'], bb['y']))
                    if bb['value'] > min_confidence:
                        # Only annotate the image if the confidence is above the minimum

                        # Scale coordinates to match original image size
                        center_x = int(bb['x'] * scale_x)
                        center_y = int(bb['y'] * scale_y)
                        
                        radius = 10 # Radius of the circle indicating the object
                        
                        # Draw circle at center point (red circle)
                        cv2.circle(original_img, (center_x, center_y), radius, (0, 0, 255), 2)
                        
                        # Add label text above circle 
                        label_text = f"{bb['label']} ({bb['value']:.2f})"
                        font_size = 1.0 # The text doesn't need to be larger
                        cv2.putText(original_img, label_text, 
                                (center_x - 40, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)

            # Save the full resolution annotated image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_filename = f"detected_objects_{timestamp}.jpg"
            cv2.imwrite(output_filename, original_img)
            print(f"Saved annotated image as: {output_filename}")

        finally:
            if runner:
                runner.stop()
            picam2.stop()

if __name__ == "__main__":
   main(sys.argv[1:]) 
