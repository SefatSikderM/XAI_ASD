import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import dlib


# Device setup
device = torch.device("cpu")

# Streamlit UI setup
st.set_page_config(page_title="ExplainNet ASD Predictor", layout="centered")
st.title("🧠 ExplainNet – Autism Detection with Explanation")

# Load model
@st.cache_resource
def load_model():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load("mobilenetv2_asd.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()
cam = GradCAM(model=model, target_layers=[model.features[-1]])
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=False, device=device)

# File upload
uploaded_file = st.file_uploader("📤 Upload a facial image", type=["jpg", "jpeg", "png"])
def detect_face_combined(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # MTCNN Detection
    mtcnn_box = None
    try:
        box = mtcnn.detect(Image.fromarray(image_np))[0]
        if box is not None:
            mtcnn_box = box[0]  # (x1, y1, x2, y2)
    except:
        pass

    # dlib Detection
    dlib_faces = detector(gray)

    return mtcnn_box, dlib_faces

if uploaded_file:
    # Load original image
    orig_img = Image.open(uploaded_file).convert("RGB")
    st.image(orig_img, caption="📷 Uploaded Image", use_container_width=True)

    # Resize image for processing
    resized_img = orig_img.resize((224, 224))
    np_img = np.array(resized_img).astype(np.float32) / 255.0

    # Apply transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(resized_img).unsqueeze(0).to(device)

    mtcnn_box, dlib_faces = detect_face_combined(np.array(resized_img))

    if mtcnn_box is None and len(dlib_faces) == 0:
        st.warning("⚠️ No clear human face detected. Prediction may be unreliable.")
        explanation = "No face detected. The prediction may not be accurate."
    else:
        explanation = ""



    # Predict
    output = model(input_tensor)
    prob = torch.nn.functional.softmax(output, dim=1)
    confidence, pred_class = torch.max(prob, 1)
    pred_label = "Autistic" if pred_class.item() == 0 else "Non-Autistic"

    # Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class.item())])[0]
    cam_img = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
    st.image(cam_img, caption=f"🔍 Prediction: **{pred_label}** ({confidence.item():.2f})", use_container_width=True)

    # Facial landmarks and region explanation (dlib)
    gray_dlib = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2GRAY)
    faces_dlib = detector(gray_dlib)
    explanation = "No face landmarks detected."

    if faces_dlib:
        shape = predictor(gray_dlib, faces_dlib[0])

        def region_mask(indices):
            points = np.array([(shape.part(i).x, shape.part(i).y) for i in indices])
            mask = np.zeros(gray_dlib.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            return mask

        region_masks = {
            'eyes': region_mask(list(range(36, 48))),
            'nose': region_mask(list(range(27, 36))),
            'mouth': region_mask(list(range(48, 68))),
            'forehead': region_mask([19, 24, 27, 22, 17])
        }

        cam_resized = cv2.resize(grayscale_cam, (224, 224))
        region_scores = {
            name: (cam_resized * mask).sum() / (mask.sum() + 1e-5)
            for name, mask in region_masks.items()
        }

        top = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        explanation = f"The model focused on the **{top[0][0]}** and **{top[1][0]}** regions."

    st.markdown(f"📝 **Explanation**: {explanation}")