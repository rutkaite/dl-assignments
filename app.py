import torch
import os
from openimages.download import download_dataset
from torch import nn
from torch import optim
import torch
import gradio as gr
import matplotlib.pyplot as plt
import dataloader as dtl

# Downloading images from openimages.com
data_dir = "data/train"
classes = ["Headphones", "Coffee", "Raven"]

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print("Downloading is starting...")
    download_dataset(data_dir, classes)

# Creating pretrained Shufflenet model:
device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
model = torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True)

# Load images
# Dividing our data into train(80%) and test(20%) data
trainloader, testloader = dtl.load_split_train_test(data_dir, .2)

# TRAINING PART

# Freezing the part of the model as no changes happen to its parameters
for param in model.parameters():
    param.requires_grad = False

# The final layer of the model is model.fc
model.fc = nn.Sequential(nn.Linear(1024, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 4),
                         nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
# Optimizer is the one which actually updates these values
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

# Training the model
# Number of epochs to train the model
epochs = 5
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

if not os.path.exists("shufflenetmodel.pth"):
    for epoch in range(epochs):
        # monitor training loss
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model?
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            running_loss += loss.item()

            # print results every 10 steps
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                # Put the model into evaluation mode, not training mode
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device),labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))
                # print training statistics
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()
    # Saving our trained model
    torch.save(model, 'shufflenetmodel.pth')
    # Visualisation of training and test losses
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

# From now on we will use our trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('shufflenetmodel.pth')
model.eval()

# Testing the model with gradio interface
import inference as inf
gr.Interface(inf.inference, inf.inputs, inf.outputs, title=inf.title, description=inf.description, analytics_enabled=False).launch(share=True)

# Testing localization
import localization as loc
#print("Predicted localization coordinates:" + loc.bounding_box(imagetest))
