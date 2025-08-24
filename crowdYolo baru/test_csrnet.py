import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
sys.path.append("CSRNet-pytorch")  
from model import CSRNet

# Load pretrained CSRNet
model = CSRNet()
model = model.cuda()
checkpoint = torch.load("checkpoint.pth.tar")  # download dari repo CSRNet
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Transform untuk input gambar
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)  # webcam / video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).cuda()

    # Prediksi density map
    with torch.no_grad():
        density_map = model(img).cpu().numpy()[0,0,:,:]

    count = int(density_map.sum())

    # Tentukan crowded level
    if count < 10:
        status = "LOW"
        color = (0, 255, 0)
    elif count < 30:
        status = "MEDIUM"
        color = (0, 255, 255)
    else:
        status = "HIGH"
        color = (0, 0, 255)

    # Tampilkan teks indikator
    cv2.putText(frame, f"Crowd: {count} ({status})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Density map -> heatmap
    heatmap = (density_map / density_map.max() * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    blended = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    cv2.imshow("Crowd Density CSRNet", blended)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
