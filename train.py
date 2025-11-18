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
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import tlc
from tqdm import tqdm
from pathlib import Path

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# Project configuration
PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class ResNet18Classifier(nn.Module):
    """ResNet-18 model for classification (no pretrained weights)"""
    def __init__(self, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        # Load ResNet-18 without pretrained weights
        self.resnet = models.resnet18(weights=None)
        
        # Get the number of features from ResNet's final layer
        resnet_features = self.resnet.fc.in_features
        
        # Remove the original final layer
        self.resnet.fc = nn.Identity()
        
        # Create new classification head
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Get ResNet features (without final classification layer)
        resnet_features = self.resnet(x)
        
        # Pass through classification head
        return self.classifier(resnet_features)


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def train_fn(sample):
    """Transform function for training data"""
    image = Image.open(sample['image'])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return train_transform(image), sample['label']


def val_fn(sample):
    """Transform function for validation data"""
    image = Image.open(sample['image'])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return val_transform(image), sample['label']


# ============================================================================
# METRICS FUNCTION
# ============================================================================

def metrics_fn(batch, predictor_output: tlc.PredictorOutput):
    """Compute per-sample metrics for 3LC collection"""
    labels = batch[1].to(device)
    predictions = predictor_output.forward
    
    # Softmax for probabilities
    softmax_output = F.softmax(predictions, dim=1)
    predicted_indices = torch.argmax(predictions, dim=1)
    confidence = torch.gather(softmax_output, 1, predicted_indices.unsqueeze(1)).squeeze(1)
    accuracy = (predicted_indices == labels).float()
    
    # Compute loss, set to 1.0 for labels outside valid range
    valid_labels = labels < predictions.shape[1]
    cross_entropy_loss = torch.ones_like(labels, dtype=torch.float32)
    cross_entropy_loss[valid_labels] = nn.CrossEntropyLoss(reduction="none")(
        predictions[valid_labels], labels[valid_labels]
    )
  
    return {
        "loss": cross_entropy_loss.cpu().numpy(),
        "predicted": predicted_indices.cpu().numpy(),
        "accuracy": accuracy.cpu().numpy(),
        "confidence": confidence.cpu().numpy(),
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train():
    """Main training function with full 3LC integration"""
    
    # Register URL alias for portable paths
    base_path = Path(__file__).parent
    tlc.register_project_url_alias(
        token="MUFFIN_DATA", 
        path=str(base_path.absolute()), 
        project=PROJECT_NAME
    )
    print(f"[OK] Registered data path alias")
    
    # Load tables using names (portable across systems)
    print("\nLoading 3LC tables...")
    train_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="train"
    ).latest()
    
    val_table = tlc.Table.from_names(
        project_name=PROJECT_NAME,
        dataset_name=DATASET_NAME,
        table_name="test"
    ).latest()
    
    print(f"Loaded train table with {len(train_table)} samples")
    print(f"Loaded val table with {len(val_table)} samples")
    
    class_names = list(train_table.get_simple_value_map("label").values()) 
    print(f"Classes: {class_names}")
    
    # Apply map functions
    train_table.map(train_fn).map_collect_metrics(val_fn)
    val_table.map(val_fn)
    
    # Create sampler that uses weights and excludes zero-weight samples
    sampler = train_table.create_sampler(exclude_zero_weights=True)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_table, 
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,  # Windows compatibility
    )

    val_dataloader = DataLoader(
        val_table,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows compatibility
    )

    # Initialize model
    model = ResNet18Classifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize 3LC run
    run = tlc.init(
        project_name=train_table.project_name,
        description="Finetuning classifier for active learning"
    )
    
    # Define metric schemas
    metric_schemas = {
        "loss": tlc.Schema(
            description="Cross entropy loss",
            value=tlc.Float32Value(),
        ),
        "predicted": tlc.CategoricalLabelSchema(
            display_name="predicted label",
            classes=class_names,
        ),
        "accuracy": tlc.Schema(
            description="Per-sample accuracy",
            value=tlc.Float32Value(),
        ),
        "confidence": tlc.Schema(
            description="Prediction confidence",
            value=tlc.Float32Value(),
        ),
    }
    
    # Create metrics collector
    classification_metrics_collector = tlc.FunctionalMetricsCollector(
        collection_fn=metrics_fn,
        column_schemas=metric_schemas,
    )
    
    # Find the layer index for embeddings collection
    indices_and_modules = list(enumerate(model.resnet.named_modules()))
    resnet_fc_layer_index = None
    for idx, (name, _) in indices_and_modules:
        if name == 'fc':
            resnet_fc_layer_index = idx
            break
    
    if resnet_fc_layer_index is None:
        resnet_fc_layer_index = len(indices_and_modules) - 1
    
    print(f"Using layer {resnet_fc_layer_index} for embeddings collection")
    
    # Create embeddings collector
    embeddings_metrics_collector = tlc.EmbeddingsMetricsCollector(
        layers=[resnet_fc_layer_index]
    )
    
    # Create predictor
    predictor = tlc.Predictor(model, layers=[resnet_fc_layer_index])
    
    # Variables to track best model
    best_val_accuracy = 0.0
    best_model_state = None
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})

        model.eval()
        # Run validation
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_progress_bar = tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}/{EPOCHS}')
        with torch.no_grad():
            for images, labels in val_progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                val_progress_bar.set_postfix({'val_loss': loss.item()})
            
        # Calculate epoch metrics
        val_avg_loss = val_loss / len(val_dataloader)
        val_accuracy = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            print(f"  [OK] New best model! Validation accuracy: {best_val_accuracy:.2f}%")
        
        # Log to 3LC
        tlc.log({
            "epoch": epoch,
            "val_loss": val_avg_loss,
            "val_accuracy": val_accuracy,
        })
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print("=" * 60)
    
    # Load best model for evaluation and metrics collection
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n[OK] Loaded best model for metrics collection")
    
    # Save best model
    torch.save(model.state_dict(), 'resnet18_classifier_best.pth')
    print("[OK] Best model saved to 'resnet18_classifier_best.pth'")
    
    # Collect metrics with best model
    print("\nCollecting metrics on train set with best model...")
    model.eval()
    tlc.collect_metrics(
        train_table,
        predictor=predictor,
        metrics_collectors=[classification_metrics_collector, embeddings_metrics_collector],
        split="train",
        dataloader_args={"batch_size": BATCH_SIZE, "num_workers": 0},
    )
    
    print("\n[OK] Metrics collection complete!")
    
    # Reduce embeddings using PaCMAP for visualization
    print("\nReducing embeddings using PaCMAP...")
    run.reduce_embeddings_by_foreign_table_url(
        train_table.url,
        method="pacmap",
        n_neighbors=2,
        n_components=3,
    )
    print("Embeddings reduction complete!")

    run.set_status_completed()
    
    print("\n" + "=" * 60)
    print("[OK] All done! View results at https://dashboard.3lc.ai")
    print("=" * 60)


if __name__ == "__main__":
    train()

