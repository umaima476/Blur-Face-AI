# Blur-Face-AI
It is small project in which faces in images detected then blurred
# Model
The weights of the Pytorch model, YoloV5 pretrained on dataset, available on open source is used.
# Procedure
All code is available in the Face Detection with Pretrained/main.py
1. Model loaded using attempt_load function of models.experiments.py given in yolov5 repository.
2. Preprocessing done on input image included image resize, conversion from BRG to RGB format, transposition from height, width, channel to channel height width and conversion of the input image into Pytorch tensor format.
3. Detection done by applying no_grad in order to temporarily stop gradient tracking.
4. Post processing including bounding box defining and face blurring is applied.
# Data
More data is available in the 'data' folder to test the code.
# Screen shots of Results
Input image
Face Detection with Pretrained/hd.jpg <br>
Output <br>
<img width="398" alt="Capture" src="https://github.com/umaima476/Blur-Face-AI/assets/59387036/1a306972-42a5-49ab-be9b-00b867ad0457">
