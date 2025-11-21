"""
Train ResNet-18 classifier on Chihuahua vs Muffin dataset using 3LC.

This script implements the full training pipeline with:
- 3LC Table loading for dataset management
- ResNet-18 model training
- Per-sample metrics collection
- Embeddings collection for visualization
- Automatic experiment tracking

DATASET DOWNLOAD:
Before running, ensure you have AWS CLI installed and download the dataset:
    
    Install AWS CLI (if not installed):
    - Windows: https://awscli.amazonaws.com/AWSCLIV2.msi
    - Mac: brew install awscli
    - Linux: sudo apt install awscli
    
    Download dataset (no AWS account required):
    aws s3 sync s3://3lc-hackathons/muffin-chihuahua/train128 ./train128 --no-sign-request
    aws s3 sync s3://3lc-hackathons/muffin-chihuahua/test128 ./test128 --no-sign-request
    
This will create train128/ and test128/ folders directly in your current directory.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path

# ---------------------------
# Training hyperparameters
# ---------------------------
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
TRAIN_VAL_SPLIT = 0.8  # fraction for train

# Project configuration
PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------------------
# Model
# ---------------------------
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet18Classifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats = self.resnet(x)
        return self.classifier(feats)

# ---------------------------
# Transforms & dataset
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class TableDataset(Dataset):
    def __init__(self, samples, transform):
        """
        samples: list of (image_path, label) tuples
        transform: torchvision transforms to apply
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # defensive — ensure path is str
        img_path = str(img_path)
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_t = self.transform(img)
        return img_t, torch.tensor(label, dtype=torch.long)

# ---------------------------
# Metric helper (optional)
# ---------------------------
def compute_metrics_from_outputs(outputs, labels):
    # outputs: tensor (batch, C), labels: tensor (batch,)
    probs = F.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    conf = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
    acc = (preds == labels).float()
    return preds.cpu().numpy(), conf.cpu().numpy(), acc.cpu().numpy()

# ---------------------------
# Main training flow
# ---------------------------
def train():
    # Register URL alias (keeps old behaviour)
    base_path = Path(__file__).parent
    dataset_path = r"C:/Users/Lavanya Bansal/AppData/Local/3LC/3LC/projects/Chihuahua-Muffin/datasets/chihuahua-muffin/tables"
    tlc.register_project_url_alias(token="MUFFIN_DATA", path=dataset_path, project=PROJECT_NAME)
    print("[OK] Registered data path alias")

    # Load the edited table (this returns an EditedTable object)
    print("\nLoading 3LC tables...")
    table_name = "DeleteX45Set342PropertiesIn342RowsTo3Values"
    edited_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name=table_name
    ).latest()
    print(f"Loaded edited table with {len(edited_table)} samples")

    # Build a list of (image_path, label) pairs from the EditedTable rows.
    # We expect the columns to be 'IMAGE' and 'LABEL' (as you indicated).
    samples = []
    missing = 0
    for row in edited_table:
        # try different key casing just in case
        # prefer 'IMAGE' and 'LABEL' but support 'image'/'label'
        if 'IMAGE' in row:
            img_col = 'IMAGE'
        elif 'image' in row:
            img_col = 'image'
        else:
            img_col = None

        if 'LABEL' in row:
            lbl_col = 'LABEL'
        elif 'label' in row:
            lbl_col = 'label'
        else:
            lbl_col = None

        if img_col is None or lbl_col is None:
            missing += 1
            continue

        img_path = row[img_col]
        label = row[lbl_col]

        # If the table stores tlc URL alias tokens (like MUFFIN_DATA/...), resolve them:
        # Many rows already have absolute paths like "C:/.../train128/..."
        # tlc usually expands tokens automatically when using the table API; here we ensure string.
        if isinstance(img_path, dict) and 'url' in img_path:
            img_path = img_path['url']
        samples.append((img_path, int(label)))

    if missing:
        print(f"Warning: {missing} rows skipped because IMAGE/LABEL columns not found in those rows")

    num_samples = len(samples)
    if num_samples == 0:
        raise RuntimeError("No samples found in the edited table. Check IMAGE/LABEL columns and paths.")

    print(f"Total samples collected from EditedTable: {num_samples}")

    # Optional: you said you have many 'undefined' labels.
    # If you want to *exclude* the 'undefined' class from training (keep only 0/1),
    # uncomment the following filter line:
    # samples = [s for s in samples if s[1] in (0, 1)]
    # num_samples = len(samples)
    # print(f"After filtering undefined, samples: {num_samples}")

    # Create train/val split (random)
    perm = torch.randperm(num_samples).tolist()
    train_count = int(TRAIN_VAL_SPLIT * num_samples)
    train_idx = perm[:train_count]
    val_idx = perm[train_count:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    # Create PyTorch datasets and dataloaders
    train_dataset = TableDataset(train_samples, train_transform)
    val_dataset = TableDataset(val_samples, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Determine classes (we'll assume labels are ints starting at 0)
    # Build class_names from unique labels encountered (sorted by label)
    unique_labels = sorted(set([lbl for _, lbl in samples]))
    class_names = [str(l) for l in unique_labels]  # fallback names are label numbers
    num_classes = max(unique_labels) + 1 if unique_labels else 1
    # If you want human names: use ['chihuahua','muffin','undefined'] and set num_classes=3
    # class_names = ['chihuahua','muffin','undefined']; num_classes = len(class_names)

    print(f"Detected label ids: {unique_labels} -> num_classes = {num_classes}")

    # Initialize model, loss, optimizer
    model = ResNet18Classifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize (optional) 3LC run for logging
    try:
        run = tlc.init(project_name=PROJECT_NAME, description="Finetuning classifier for active learning")
    except Exception:
        run = None

    best_val_accuracy = 0.0
    best_model_state = None

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_avg_loss = val_loss / (len(val_loader) if len(val_loader) > 0 else 1)
        val_accuracy = 100.0 * val_correct / (val_total if val_total > 0 else 1)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss/len(train_loader):.4f}  Val Loss: {val_avg_loss:.4f}  Val Acc: {val_accuracy:.2f}%")

        # log to 3LC if available
        if run is not None:
            try:
                tlc.log({"epoch": epoch, "val_loss": val_avg_loss, "val_accuracy": val_accuracy})
            except Exception:
                pass

        # checkpoint best
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [OK] New best model! Validation accuracy: {best_val_accuracy:.2f}%")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print("=" * 60)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "resnet18_classifier_best.pth")
        print("[OK] Saved best model to resnet18_classifier_best.pth")

    # If you still want to collect metrics into 3LC programmatically you can:
    # - either convert the samples back into a tlc Table (depends on your 3LC version)
    # - or skip and use the tlc API to upload whatever per-sample metrics you'd like.
    # For simplicity and compatibility we omitted tlc.collect_metrics here.

    if run is not None:
        try:
            run.set_status_completed()
        except Exception:
            pass

if __name__ == "__main__":
    train()