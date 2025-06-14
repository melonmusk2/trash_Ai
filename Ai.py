from urllib.request import urlopen
from PIL import Image
import timm
import torch

# Load image from URL
try:
    img = Image.open(urlopen('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT_p_4ET0w23YzM-cHhuIDdVYDzU1CXavkwgQ&s'))
except Exception as e:
    print(f"Error opening image from URL: {e}")
    exit()

# Load pre-trained model
model = timm.create_model('mobilenetv3_small_100.lamb_in1k', pretrained=True)
model.eval()

# ImageNet transform setup
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Model inference
output = model(transforms(img).unsqueeze(0))
probabilities = torch.nn.functional.softmax(output, dim=1) * 100
top_prob, top_class_index = torch.max(probabilities, dim=1)

# Load ImageNet labels
try:
    with urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
        labels = [line.strip().decode("utf-8") for line in f.readlines()]
except Exception as e:
    print(f"Could not load ImageNet labels. Error: {e}")
    labels = None

# Expanded category mapping
category_mapping = {
    # PLASTIK
    "water bottle": "plastik",
    "plastic bag": "plastik",
    "pop bottle": "plastik",
    "packet": "plastik",
    "sunscreen": "plastik",
    "shampoo": "plastik",
    "soap dispenser": "plastik",
    "detergent": "plastik",
    "toothbrush": "plastik",
    "fork": "plastik",
    "knife": "plastik",
    "spoon": "plastik",
    "container ship": "plastik",
    "shopping basket": "plastik",
    "tray": "plastik",
    "lotion": "plastik",
    "pen": "plastik",
    "highlighter": "plastik",
    "comb": "plastik",
    "screwdriver": "plastik",
    "toys": "plastik",

    # PAPER
    "newspaper": "paper",
    "notebook": "paper",
    "book": "paper",
    "envelope": "paper",
    "magazine": "paper",
    "toilet tissue": "paper",
    "paper towel": "paper",
    "calendar": "paper",
    "folder": "paper",
    "menu": "paper",
    "paper plate": "paper",
    "receipt": "paper",
    "manual": "paper",
    "document": "paper",

    # ORGANIC
    "banana": "organic",
    "apple": "organic",
    "orange": "organic",
    "lemon": "organic",
    "cucumber": "organic",
    "corn": "organic",
    "mushroom": "organic",
    "carrot": "organic",
    "broccoli": "organic",
    "pineapple": "organic",
    "eggplant": "organic",
    "zucchini": "organic",
    "lettuce": "organic",
    "sandwich": "organic",
    "pizza": "organic",
    "hamburger": "organic",
    "hotdog": "organic",
    "bone": "organic",
    "egg": "organic",
    "bread": "organic",
    "cake": "organic",
    "cheeseburger": "organic",
    "meat loaf": "organic",
    "steak": "organic",

    # GLASS
    "wine bottle": "glass",
    "beer bottle": "glass",
    "goblet": "glass",
    "glass": "glass",
    "vase": "glass",
    "perfume": "glass",

    # BATTERIES
    "remote control": "batteries",
    "cellular telephone": "batteries",
    "iPod": "batteries",
    "battery": "batteries",
    "digital watch": "batteries",
    "hand-held computer": "batteries",
    "laptop": "batteries",
}

# Display result
if labels:
    class_index = top_class_index.item()
    probability = top_prob.item()
    class_label = labels[class_index]
    category = category_mapping.get(class_label, "anything else")
    print(f"Top Prediction: {class_label} ({probability:.2f}%) -> Category: {category}")
else:
    print("Could not display human-readable label.")
    print(f"Top class index: {top_class_index.item()} | Probability: {top_prob.item():.2f}%")
