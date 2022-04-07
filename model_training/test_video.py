import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

import config
import utils
from model import EBUnet


dice_bce_loss = lambda x: None

model = EBUnet().to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load(f'{config.OUTPUT_PATH}/EB-UNet/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# capture the webcam
cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('Error while trying to open webcam. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# set up the save file path
save_path = f"{config.OUTPUT_PATH}/vid_keypoint_detection.mp4"
# define codec and create VideoWriter object
# out = cv2.VideoWriter(f"{save_path}",
#                       cv2.VideoWriter_fourcc(*'mp4v'), 20,
#                       (frame_width, frame_height))

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        with torch.no_grad():
            image = frame
            orig_frame = image.copy()
            image = cv2.resize(image, (224, 224))
            image = cv2.bilateralFilter(image, 9, 75, 75)
            orig_h, orig_w, c = orig_frame.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to(config.DEVICE)

            outputs = utils.mask_to_keypoints(model(image))

        outputs = outputs.cpu().detach().numpy()

        outputs = outputs.reshape(-1, 2)
        keypoints = (outputs * np.array([orig_w, orig_h])).astype(np.int)
        orig_frame = utils.draw_keypoints(orig_frame, keypoints)
            # cv2.circle(orig_frame, (int(keypoints[p, 0]), int(keypoints[p, 1])),
            #             1, (0, 0, 255), 2)

        orig_frame = cv2.resize(orig_frame, (frame_width, frame_height))


        cv2.imshow('Video', orig_frame)
        # out.write(orig_frame)

    else:
        break
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()
