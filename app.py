import torch
from torchvision.transforms import v2 as T
from unet_sod_model import UNetSODModel
import gradio as gr
import time
import numpy as np
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

model = UNetSODModel().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=DEVICE))
model.eval()

image_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True)
])

def predict(image):
    original_size = image.size #(width, height)
    image = image.convert("RGB")

    image_tensor = image_transform(image)
    
    # add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        #Start time measurement
        start = time.perf_counter()
        prediction = model(image_tensor)
        #End time measurement
        elapsed_ms = (time.perf_counter() - start) * 1000
        
    pred_binary = (prediction > 0.5).float()

    #This turns the pixels into either 0 or 1
    result = pred_binary.squeeze().cpu().numpy()
    #We multiply by 255 to turn it into either 0(black) or 255(pure white)
    result = (result * 255).astype(np.uint8)
    result = Image.fromarray(result)
    result = result.resize(original_size, Image.NEAREST)

    original_np = np.array(image) / 255.0
    mask_np = np.array(result) / 255.0

    #Adding red glow on salient pixels and overlaying
    overlay = original_np.copy()
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + 0.4 * mask_np, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    overlay = Image.fromarray(overlay)

    return result, overlay, f"{elapsed_ms:.1f} ms"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Image(type="pil"), gr.Textbox()],
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)