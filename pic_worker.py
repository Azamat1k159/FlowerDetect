from torchvision.models import densenet201
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json


def load_model(model, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError("Файл не найден")

    device = 'cuda:0' if use_gpu and torch.cuda.is_available() else 'cpu'
    d = torch.load(filename, map_location=device)
    model.load_state_dict(d['model'])
    model = model.to(device)
    return d['epoch'], model


def preprocess_image(image_path):
    """ Преобразование изображения для модели """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Добавляем размерность batch


def classify_image(model, image_path, device, species_mapping, class_mapping, topk=5):
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        model.eval()
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.topk(probabilities, topk)
        top_prob = top_prob[0].cpu().numpy()
        top_catid = top_catid[0].cpu().numpy()
        # Маппинг id нейросети в id вида
        mapped_ids = [class_mapping.get(str(cat_id)) for cat_id in top_catid]
        species_names = [species_mapping.get(str(mapped_id), "Неизвестный вид") for mapped_id in mapped_ids]
        return list(zip(species_names, top_prob))


filename = 'densenet201_weights_best_acc.tar'  # путь к предобученной модели
use_gpu = True  # использовать GPU для загрузки весов
model = densenet201(num_classes=1081)  # 1081 класс в Pl@ntNet-300K

epoch, model = load_model(model, filename=filename, use_gpu=use_gpu)

# Загрузка сопоставления видов
with open('plantnet300K_species_id_2_name.json', 'r') as f:
    species_mapping = json.load(f)

# Загрузка промежуточного сопоставления
with open('input2id.json', 'r') as f:
    class_mapping = json.load(f)

# Пример использования
image_path = 'lolr(26).JPG'  # путь к вашему изображению
results = classify_image(model, image_path, "cuda:0", species_mapping, class_mapping)
for species, prob in results:
    print(f"Вид: {species}, Вероятность: {prob:.2%}")
