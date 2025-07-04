def train(image):
  import torch, torchvision as tv
  from torchvision import transforms
  import torch.nn as nn
  import torch
  from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
  from torch.utils.data import DataLoader
  from pathlib import Path
  from tqdm import tqdm
  from PIL import Image

  data_dir = Path("data/split")  # instead of "data/garbage-classification"

  img_size = 224  # MobileNetV3 default
  train_tf = transforms.Compose([
    transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(tv.transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
  ])
  val_tf = transforms.Compose([
    transforms.Resize(img_size + 32),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225])
  ])

  train_ds = tv.datasets.ImageFolder(data_dir / "train", transform=train_tf)
  val_ds   = tv.datasets.ImageFolder(data_dir / "valid", transform=val_tf)

  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
  val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

  num_classes = len(train_ds.classes)   # → 12
  #print(train_ds.classes)


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
  model = mobilenet_v3_large(weights=weights).to(device)

# Freeze feature extractor (optional – will speed up training)
  for param in model.features.parameters():
    param.requires_grad = False

# Replace final linear head
  in_features = model.classifier[-1].in_features      # 1280
  model.classifier[-1] = nn.Linear(in_features, num_classes)
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
# Only parameters that require_grad=True are handed to the optimiser
  optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=1e-3, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

  """ epochs = 10
  for epoch in range(epochs):
    # ---- Train ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total

    # ---- Validate ----
    model.eval()
    with torch.no_grad():
        val_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)

        val_loss /= total
        val_acc   = correct / total

    scheduler.step()
    print(f"Epoch {epoch+1}: "
          f"train loss={train_loss:.3f} acc={train_acc:.3%} | "
          f"val loss={val_loss:.3f} acc={val_acc:.3%}")
    
  torch.save({
    "model_state": model.state_dict(),
    "classes": train_ds.classes
  }, "mobilenetv3_garbage12.pt")"""

  ckpt = torch.load("mobilenetv3_garbage12.pt", map_location=device)
  model.load_state_dict(ckpt["model_state"])
  model.eval()


  x = val_tf(image).unsqueeze(0).to(device)
  with torch.no_grad():
    logits = model(x)
  pred_idx = logits.argmax(1).item()
  pred_label = ckpt["classes"][pred_idx]
  return pred_label

if __name__ == "__main__":
    train()




