import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


import utils
import config
from model import KPR, EBUnet

dice_bce_loss = lambda x: None




# model = KPR(pretrained=False, requires_grad=False).to(config.DEVICE)
model = EBUnet().to(config.DEVICE)
# load the model checkpoint
checkpoint = torch.load(f'{config.OUTPUT_PATH}/EB-UNet/model_test.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()



for i in range(1, 10):
    # get the image
    image = cv2.imread(f'test_images/{i}.jpg')
    #print(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_frame = image.copy()

    # rescale it
    image = cv2.resize(image, (128, 128))
    orig_h, orig_w, c = orig_frame.shape
    image = image / 255.0

    # get the tensor
    x = np.transpose(image, (2, 0, 1))
    x = torch.tensor(x, dtype=torch.float)
    x = x.unsqueeze(0).to(config.DEVICE)

    # predict the keypoints
    outputs = utils.mask_to_keypoints(model(x))
    outputs = outputs.cpu().detach().numpy()
    keypoints = outputs.reshape(-1, 2)

    # plot the batch
    utils.plot_dataset_batch([{
        'image': orig_frame / 255,
        'keypoints': keypoints
    }], n=1, transpose=False)
