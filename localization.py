#this part does not work properly enough to be used in the code
#some mofifications needed as coordinates are not always correct
#logic of this code seems to be correct just some parameters probably need to be changed

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('shufflenetmodel.pth')
model.eval()

import dataloader as dtl
data_dir = "data/train"
#dividing our data into train(80%) and test(20%) data
trainloader, testloader = dtl.load_split_train_test(data_dir, .2)

def to_coord(pred, shape):
    # Convert predictions and shape into bounding box coordinates
    _, _, w, h = shape
    x0 = max(int(pred[0] * w), 0)
    x1 = min(int(pred[1] * w), w)
    y0 = max(int(pred[2] * h), 0)
    y1 = min(int(pred[3] * h), h)
    return [x0, y0, x1 - x0, y1 - y0]

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

import matplotlib.patches as patches

def bounding_box(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    im = input_batch.permute(0, 2, 3, 1)
    predicted = to_coord(output[0], input_batch.shape)
    print("predicted")
    print(predicted)
    fig = plt.imshow(im[0])
    fig.axes.add_patch(bbox_to_rect(predicted, 'blue'))
    plt.show()
    return torch.tensor((predicted))
