from flask import Flask, request, jsonify, render_template, send_file
import torch
from torchvision import transforms
import torchvision
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
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
model.load_state_dict(torch.load('Change_Detection_model.pth', map_location=device))
model.eval()

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict mask and plot examples
def predict_and_save(model, image, save_path):
    model.eval()
    
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device)).cpu()
        
    pred_mask = torch.argmax(output, dim=1)
    
    # Plot the image and predicted mask
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(transforms.ToPILImage()(image.cpu().squeeze()))  # Convert tensor to PIL image
    plt.axis("off")
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask.squeeze(), cmap='gray')  # Assuming pred_mask is a single channel image
    plt.axis("off")
    plt.title('Predicted Mask')

    plt.tight_layout()
    
    # Save the predicted mask image
    plt.savefig(save_path)
    plt.close()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to navigate to upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def predict():
    # Receive input image
    if 'imageFile' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['imageFile']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess the input image
    input_tensor = image_transform(image).unsqueeze(0).to(device)
    
    # Predict mask and save example
    save_path = "predicted_mask.png"
    predict_and_save(model, input_tensor.squeeze(), save_path)
    
    # Return the saved image file
    return send_file(save_path, mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)