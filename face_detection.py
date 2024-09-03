
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Load an image containing faces
img = Image.open('im.jpg')

# Detect faces in the image
boxes, _ = mtcnn.detect(img)

# If faces are detected, 'boxes' will contain the bounding box coordinates
if boxes is not None:
    for box in boxes:
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        draw.rectangle(box.tolist(), outline='red', width=3)

# Display or save the image with detected faces
img.show()