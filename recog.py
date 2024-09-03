import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### helper function
def encode(img):
    res = resnet(torch.Tensor(img).to(device))
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
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(
  image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60, device=device
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### get encoded features for all saved images
'''
saved_pictures = "./saved/"
all_people_faces = {}
for person_face in os.listdir(saved_pictures):
    img = cv2.imread(f'{saved_pictures}/{person_face.split(".")[0]}.jpg')
    cropped = mtcnn(img)
    if cropped is not None:
        all_people_faces[person_face.split('.')[0]] = encode(cropped)[0, :]'''

file_path = 'data1.csv'

face_id_df = pd.read_csv(file_path)
# print(face_id_df.head())
face_id = {}
for column_name, column_data in face_id_df.items():
    id = torch.tensor(column_data.values)
    face_id[column_name] = id

###########
def detect(cam=0, thres=0.7):
    vdo = cv2.VideoCapture(cam)
    while vdo.grab():
        _, img0 = vdo.retrieve()
        batch_boxes, cropped_images = mtcnn.detect_box(img0)

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(x) for x in box]
                img_embedding = encode(cropped.unsqueeze(0))
                detect_dict = {}
                for k, v in face_id.items():
                    detect_dict[k] = (v.to(device) - img_embedding).norm().item()

                min_key = min(detect_dict, key=detect_dict.get)

                if int(min_key) < 738:
                    name = 'Sepehr'
                if detect_dict[min_key] >= thres:
                    name = 'Undetected'

                
                
                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                  img0, name, (x + 5, y + 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
                 
        ### display
        cv2.imshow("output", img0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    detect(0)