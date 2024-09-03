
import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType
import pandas as pd



# Ensure CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Helper function
def encode(img):
    res = resnet(img.to(device))  # Move the image tensor to GPU
    return res

def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces

### Load model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Move the model to GPU
mtcnn = MTCNN(
    image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60, device=device  # Specify the device for MTCNN
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### Get encoded features for all saved images
def embed(all_people_faces={}, saved_vid="./vid/vid1.mp4", id=0):
    cap = cv2.VideoCapture(saved_vid)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames

    with tqdm(total=total_frames, desc="Processing Video") as pbar:  # Initialize tqdm with the total number of frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if no more frames
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                cropped = mtcnn(frame_rgb)
                if cropped is not None:
                    all_people_faces[id] = encode(cropped)[0, :].detach().cpu().numpy()  # Move the tensor back to CPU for numpy conversion
            
            pbar.update(1)  # Update tqdm progress bar
            id += 1

    return all_people_faces

if __name__ == "__main__":
    data = embed()
    df = pd.DataFrame(data)

    file_path = 'data1.csv'
    df.to_csv(file_path, index=False)


'''
# Ensure CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Helper function
def encode(img):
    res = resnet(img.to(device))  # Move the image tensor to GPU
    return res

def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces

### Load model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Move the model to GPU
mtcnn = MTCNN(
    image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60, device=device  # Specify the device for MTCNN
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### Get encoded features for all saved images
def embed(all_people_faces={}, saved_vid="./vid/vid.mp4", id='sepehr'):
    cap = cv2.VideoCapture(saved_vid)

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            cropped = mtcnn(frame_rgb)
            if cropped is not None:
                all_people_faces[id] = encode(cropped)[0, :].detach().cpu().numpy()  # Move the tensor back to CPU for numpy conversion

    return all_people_faces

if __name__ == "__main__":
    data = embed()
    df = pd.DataFrame(data)

    file_path = 'data.csv'
    df.to_csv(file_path, index=False)
'''

'''


### helper function
def encode(img):
    res = resnet(torch.Tensor(img))
    return res

def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces


### load model
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
  image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### get encoded features for all saved images

def embed(all_people_faces = {}, saved_vid = "./vid/vid.mp4", id='sepehr'):
    cap = cv2.VideoCapture(saved_vid)

    while cap.isOpened() :
        ret, frame = cap.read()
        # img = cv2.imread(f'{saved_pictures}/{person_face.split(".")[0]}.jpg')
        if frame is not None:
            cropped = mtcnn(frame)
            if cropped is not None:
                all_people_faces[id] = encode(cropped)[0, :].detach().numpy() #person_face.split('.')[0]

    return all_people_faces



if __name__ == "__main__":
    data = embed()
    df = pd.DataFrame(data)

    file_path = 'data.csv'
    df.to_csv(file_path, index=False)'''
