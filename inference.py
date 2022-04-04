import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('shufflenetmodel.pth')
model.eval()

import dataloader as dtl
data_dir = "data/train"
trainloader, testloader = dtl.load_split_train_test(data_dir, .2)

def inference(input_image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    # The output has unnormalized scores. To get probabilities, we will run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Show top categories per image
    categories = trainloader.dataset.classes
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    result = {}
    for i in range(top3_prob.size(0)):
        result[categories[top3_catid[i]]] = top3_prob[i].item()
    return result

# Variables for gradio interface
inputs = gr.inputs.Image(type='pil')
outputs = gr.outputs.Label(type="confidences",num_top_classes=3)
title = "SHUFFLENET"
description = "Please upload your headphones, coffee or reiven image."
