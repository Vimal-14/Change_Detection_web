from flask import Flask, request, jsonify, render_template, send_file
import torch
from torchvision import transforms
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import io
import matplotlib.pyplot as plt
app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UpResNet(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, bilinear=True, dropout_rate=0.1):
        super(UpResNet, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_1, in_channels_1, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels_1 + in_channels_2, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, in_channels, out_channels, resnet_type="resnet18", bilinear=False, dropout_rate = 0.1):
        super(ResNetUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resnet_type = resnet_type
        self.bilinear = bilinear
        self.dropout_rate = dropout_rate
        
        # Define the backbone network
        if self.resnet_type == "resnet18":
            self.backbone_model = torchvision.models.resnet18(weights="DEFAULT")
            self.channel_distribution = [3, 64, 64, 128, 256]
        elif self.resnet_type == "resnet34":
            self.channel_distribution = [3, 64, 64, 128, 256]
            self.backbone_model = torchvision.models.resnet34(weights="DEFAULT")
        elif self.resnet_type == "resnet50":
            self.channel_distribution = [3, 64, 256, 512, 1024]
            self.backbone_model = torchvision.models.resnet50(weights="DEFAULT")
        else:
            print("Resnet type is not recognized. Loading ResNet 18 as backbone model")
            self.channel_distribution = [3, 64, 64, 128, 256]
            self.backbone_model = torchvision.models.resnet34(weights="DEFAULT")
        
        self.backbone_layers = list(self.backbone_model.children())
        
        # Define the ResNetUNet
        self.inc = DoubleConv(in_channels, 64)
        
        self.block1 = nn.Sequential(*self.backbone_layers[0:3])
        self.block2 = nn.Sequential(*self.backbone_layers[3:5])
        self.block3 = nn.Sequential(*self.backbone_layers[5])
        self.block4 = nn.Sequential(*self.backbone_layers[6])
        
        self.up1 = Up(self.channel_distribution[-1], self.channel_distribution[-2], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up2 = Up(self.channel_distribution[-2], self.channel_distribution[-3], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up3 = UpResNet(self.channel_distribution[-3], 64, self.channel_distribution[-4], bilinear=bilinear, dropout_rate = dropout_rate)
        self.up4 = UpResNet(self.channel_distribution[-4], 64, self.channel_distribution[-4], bilinear=bilinear, dropout_rate = dropout_rate)
        
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        y1 = self.up1(x4, x3)
        y2 = self.up2(x3, x2)
        y3 = self.up3(x2, x1)
        y4 = self.up4(x1, x0)

        logits = self.outc(y4)
        
        return logits


model = ResNetUNet(in_channels=3, out_channels=2, resnet_type="resnet34").to(device)
model.load_state_dict(torch.load('Change_Detection_model_best.pth', map_location=device))
model.eval()

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def generate_change_mask(old_mask, new_mask):
    # Compare old and new masks pixel by pixel
    change_mask = (old_mask != new_mask).float()
    
    # Example: Apply post-processing or filtering to refine the change mask
    
    return change_mask

# Function to predict mask and plot examples
# def predict_and_save(model, image, save_path):
def predict_changes_and_save(model, old_image, new_image, save_path):
    model.eval()
    
    with torch.no_grad():
        old_output = model(old_image.unsqueeze(0).to(device)).cpu()
        new_output = model(new_image.unsqueeze(0).to(device)).cpu()
        
    old_mask = torch.argmax(old_output, dim=1)
    new_mask = torch.argmax(new_output, dim=1)
    
    # Compare old and new masks to generate a mask highlighting changes
    change_mask = generate_change_mask(old_mask, new_mask)
    
    # Convert torch tensors to numpy arrays
    old_image_np = old_image.cpu().numpy().transpose(1, 2, 0)
    new_image_np = new_image.cpu().numpy().transpose(1, 2, 0)
    change_mask_np = change_mask.cpu().numpy().squeeze()
    
    # Convert old and new images to grayscale
    old_image_gray = cv2.cvtColor(old_image_np, cv2.COLOR_RGB2GRAY)
    new_image_gray = cv2.cvtColor(new_image_np, cv2.COLOR_RGB2GRAY)
    
    # Create a combined image with old image, new image, and change mask
    combined_image = np.concatenate((old_image_gray[..., np.newaxis], new_image_gray[..., np.newaxis], change_mask_np[..., np.newaxis]), axis=1)
    
    # Convert the combined image to uint8 data type
    combined_image_uint8 = (combined_image * 255).astype(np.uint8)
    
    # Create a PIL Image from the combined image
    combined_image_pil = Image.fromarray(combined_image_uint8)
    
    # Save the combined image
    combined_image_pil.save(save_path)
    
    # Return the saved image file path
    return save_path

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to navigate to upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Update predict function to handle two images and generate change mask
@app.route('/upload', methods=['POST'])
def predict():
    # Receive input images
    if 'oldImageFile' not in request.files or 'newImageFile' not in request.files:
        return jsonify({'error': 'Please upload both old and new images'}), 400

    old_image_file = request.files['oldImageFile']
    new_image_file = request.files['newImageFile']
    
    old_image_bytes = old_image_file.read()
    new_image_bytes = new_image_file.read()
    
    old_image = Image.open(io.BytesIO(old_image_bytes)).convert("RGB")
    new_image = Image.open(io.BytesIO(new_image_bytes)).convert("RGB")

    # Preprocess the input images
    old_input_tensor = image_transform(old_image).unsqueeze(0).to(device)
    new_input_tensor = image_transform(new_image).unsqueeze(0).to(device)
    
    # Predict masks for old and new images
    with torch.no_grad():
        old_output = model(old_input_tensor).cpu()
        new_output = model(new_input_tensor).cpu()

    old_pred_mask = torch.argmax(old_output, dim=1).squeeze()
    new_pred_mask = torch.argmax(new_output, dim=1).squeeze()
    
    # Compute difference mask
    change_mask = (new_pred_mask != old_pred_mask).float()
    
    # Convert to PIL images
    old_image_pil = Image.open(io.BytesIO(old_image_bytes))
    new_image_pil = Image.open(io.BytesIO(new_image_bytes))
    change_mask_pil = transforms.ToPILImage()(change_mask)

    # Combine images and mask
    combined_image = Image.new('RGB', (old_image_pil.width * 3, old_image_pil.height))
    combined_image.paste(old_image_pil, (0, 0))
    combined_image.paste(new_image_pil, (old_image_pil.width, 0))
    combined_image.paste(change_mask_pil, (old_image_pil.width * 2, 0), mask=change_mask_pil)
    
    # Save combined image
    save_path = "predicted_mask.png"
    combined_image.save(save_path)

    # Return the saved image file
    return send_file(save_path, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)