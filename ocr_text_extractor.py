from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import json
import os
import sys
import numpy as np
import kyc_config as cfg

# Azure Cognitive Services configuration
computervision_client = ComputerVisionClient(
    cfg.azure_endpoint,
    CognitiveServicesCredentials(cfg.azure_subscription_key)
)

def get_text_response_from_path(path):
    output = None

    try:
        if path.startswith('http') or path.startswith('gs:'):
            # Handle remote image (URL)
            text_response = computervision_client.read(path, raw=True)
        else:
            # Handle local image file
            with open(path, 'rb') as image_file:
                text_response = computervision_client.read_in_stream(image_file, raw=True)

        # Extract the operation location (URL with operation ID)
        operation_location = text_response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]

        # Wait for the operation to complete
        while True:
            result = computervision_client.get_read_result(operation_id)
            if result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
                break

        if result.status == OperationStatusCodes.succeeded:
            # Extract OCR results
            output = [line.text for page in result.analyze_result.read_results for line in page.lines]
        else:
            output = "OCR processing failed"

    except Exception as e:
        output = f"Error processing file: {e}"
    
    return output

def process_ocr(img_path):
    text_response = get_text_response_from_path(img_path)

    # Save the output file
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    json_name = os.path.join(cfg.json_loc, f'ocr_{img_name}.npy')
    np.save(json_name, text_response)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Input: image path
        img_path = sys.argv[1]
        print(f'OCR processing {img_path}')
        process_ocr(img_path)
    else:
        print('Argument is missing: image path')
