import os
import kyc_config as cfg
import ocr_text_extractor as ocr
import ktp_entity_extractor as extractor

from flask import Flask, request, jsonify
from datetime import datetime


app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the OCR & KTP Extraction API!"})

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        original_filename = file.filename.rsplit('.', 1)
        if len(original_filename) == 2:
            unique_filename = f"{original_filename[0]}_{timestamp}.{original_filename[1]}"
        else:
            unique_filename = f"{file.filename}_{timestamp}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            print(f'OCR processing {file_path}')
            ocr.process_ocr(file_path)

            img_name = file.filename.split('/')[-1].split('.')[0]
            ocr_path = os.path.join(cfg.json_loc, f'ocr_{img_name}.npy')

            print(f'Extracting data from {ocr_path}')
            extracted_data = extractor.process_extract_entities(ocr_path)

            if extracted_data is not None:
                extracted_data_json = extracted_data.to_dict(orient='records')
                return jsonify({
                    "message": "Processing successful",
                    "data": extracted_data_json[0]
                })
            else:
                return jsonify({"message": "No data extracted from the OCR response"}), 204

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unknown error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
