import torch
from torchvision import transforms
from PIL import Image

# Define la transformación para la imagen
preprocess = transforms.Compose([
    transforms.ToTensor(),
])

# Carga el modelo preentrenado desde el archivo local
model_path = fr"practice\neuronal_networks\yolov8n.onnx"  # Reemplaza con la ruta a tu modelo
model = torch.load(model_path)
# model.eval()

# Carga la imagen de prueba
image_path = "bus.jpg"  # Reemplaza con la ruta a tu imagen
img = Image.open(image_path)

# Realiza la transformación de la imagen
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

# Realiza la inferencia
with torch.no_grad():
    output = model(input_batch)

# Muestra los resultados o realiza acciones adicionales según tus necesidades
# Puedes acceder a los resultados específicos del modelo preentrenado y procesarlos según sea necesario
