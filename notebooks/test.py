# import cv2
# from sympy.printing.pretty.pretty_symbology import annotated
# from ultralytics import YOLO
#
# # Load your YOLOv8 model (replace 'yolov8s.pt' with your model's path)
# model = YOLO('./model')
#
# # Input video path
# video_path = "C:/Users/Efundzz-005/IdeaProjects/mp4_videos/scrap.mp4"
# cap = cv2.VideoCapture(video_path)
#
# # Output video settings
# output_path = 'output_video.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
# # Create the VideoWriter
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Perform object detection on the frame
#     results = model(frame)
#
#     # Annotate the detected objects
#     annotated_frame = results.render()[0]
#
#     # Write the annotated frame to the output video
#     out.write(annotated_frame)
#
# # Release video capture and writer resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()



import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="F6GbctrA5BtvWWkQZrsd")
project = rf.workspace().project("smart-factory-ejbih")
model = project.version(3).model

# # infer on a local image
print(model.predict("steel_scrap.jpg", confidence=40, overlap=30).json())



# visualize your prediction
model.predict("steel_scrap.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("prediction.jpg", hosted=True, confidence=40, overlap=30).json())




#
#
# # Open a video capture (replace 'video_file.mp4' with your video file)
# cap = cv2.VideoCapture('scrap.mp4')
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Perform object detection on the frame
#     # You'll need to convert the frame to an image format that the model.predict method can accept
#     # Draw bounding boxes on the frame based on the model's predictions
#     # Display the frame with bounding boxes
#
#     # Process and visualize results here
#
#     # Break the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()