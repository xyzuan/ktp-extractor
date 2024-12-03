from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import numpy as np
import os
import sys
import kyc_config as cfg

computervision_client = ComputerVisionClient(
    cfg.azure_endpoint,
    CognitiveServicesCredentials(cfg.azure_subscription_key)
)

def get_text_response_from_path(path):
    try:
        if path.startswith('http'):
            text_response = computervision_client.read(path, raw=True)
        else:
            with open(path, 'rb') as image_file:
                text_response = computervision_client.read_in_stream(image_file, raw=True)

        operation_location = text_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        while True:
            result = computervision_client.get_read_result(operation_id)
            if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break

        if result.status == OperationStatusCodes.succeeded:
            output = [
                {"text": line.text, "bounding_box": line.bounding_box}
                for page in result.analyze_result.read_results
                for line in page.lines
            ]
            return output
        else:
            print("OCR processing failed.")
            return None

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def process_ocr(img_path):
    text_response = get_text_response_from_path(img_path)
    if text_response:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        json_name = os.path.join(cfg.json_loc, f'ocr_{img_name}.npy')
        np.save(json_name, text_response)
        print(f"OCR data saved to {json_name}")
    else:
        print("No valid OCR data to save.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f'OCR processing {img_path}')
        process_ocr(img_path)
    else:
        print('Argument is missing: image path')
