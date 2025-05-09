import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io

import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 128 * 128, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
model.load_state_dict(torch.load("simple_nn_model.pth", map_location=torch.device('cpu')))
model.eval()

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def predict(image):
    image_tensor = transform_image(image)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    classes = ['health(건강)', 'rusts(썩음)', 'powdry(질병)']
    return classes[predicted.item()]


st.title("📷 식물 병해 이미지 분류기 📷")
st.write("이미지를 업로드하면 health(건강) / rusts(썩음) / powdry(질병) 중 하나로 예측해줍니다.")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    prediction = predict(image)
    st.success(f"✅ 예측 결과: **{prediction}**")

st.write("---------------------------------------------------------------------")

st.image("예시2.PNG", caption="사용 방법_예시_이미지")
