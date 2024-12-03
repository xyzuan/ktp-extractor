import cv2
import numpy as np


image = cv2.imread('img/sample_ktp.png')
data = np.load('ocr/ocr_sample_ktp.npy', allow_pickle=True)

for item in data:
    bbox = np.array(item['bounding_box'], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [bbox], isClosed=True, color=(0, 255, 255), thickness=2)

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
