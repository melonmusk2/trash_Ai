from urllib.request import urlopen
from PIL import Image
import timm
import torch

# replace with actual solution, instea dof static
img = Image.open(urlopen(
    'https://m.media-amazon.com/images/I/A1p+c9bsUhL.jpg'
))

model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

try:
    # this is where the image names come from
    with urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
except Exception as e:
    print(f"Could not load ImageNet labels. Error: {e}")
    labels = None

if labels:
    print("Top 5 predictions:")
    for i in range(top5_probabilities.shape[1]):
        class_index = top5_class_indices[0, i].item()
        probability = top5_probabilities[0, i].item()
        class_label = labels[class_index]
        print(f"- {class_label}: {probability:.2f}%")
else:
    print("Could not display human-readable labels. Here are the top 5 class indices and probabilities:")
    print(f"Top 5 class indices: {top5_class_indices.tolist()}")
    print(f"Top 5 probabilities: {top5_probabilities.tolist()}")