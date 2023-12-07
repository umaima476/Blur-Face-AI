import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression

# load model
model_path = 'model/face_detection_yolov5s.pt'
device = 'cpu'
model = attempt_load(model_path, device)

# evaluation mode on
model.eval()

# input image
image_path = 'hd.jpg'

# preprocessing
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (640, 640))
rgb_trans_image = resized_image[:, :, ::-1].transpose(2, 0, 1)
input_tensor = torch.as_tensor(rgb_trans_image.astype("float32")).div(255.0).unsqueeze(0)

# detection
with torch.no_grad():
    pred = model(input_tensor)
    print(pred)

# checking and assigning tuple
if isinstance(pred, tuple):
    pred = pred[0]

# post-processing including bounding boxes and blurring
conf_thres = 0.5
iou_thres = 0.5
pred = non_max_suppression(pred, conf_thres, iou_thres)

if len(pred) > 0:
    det = pred[0]
    for *xyxy, conf, cls in det:
        label = f'{int(cls)}: {conf:.2f}'
        xyxy = [int(i) for i in xyxy]

        face_roi = resized_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        # check if coordinates are present
        if face_roi.size == 0:
            print('face region is empty')
        else:
            face_roi = cv2.GaussianBlur(face_roi, (0, 0), 30)

        resized_image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = face_roi

        cv2.rectangle(resized_image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(resized_image, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# save and display
output_path = f'outputs/hd3.jpg'
cv2.imwrite(output_path, resized_image)
cv2.imshow('Blurred Faces', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()