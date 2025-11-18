"""
Register Chihuahua vs Muffin dataset in 3LC tables.
Creates separate tables for train and test sets.
Reads from three folders: chihuahua, muffin, and undefined.

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
"""

import tlc
from pathlib import Path

# Define constants
CLASSES = ["chihuahua", "muffin", "undefined"]
PROJECT_NAME = "Chihuahua-Muffin"
DATASET_NAME = "chihuahua-muffin"

# Define schemas for the table
schemas = {
    "id": tlc.Schema(value=tlc.Int32Value(), writable=False),  # Ensure the ID is not writable
    "image": tlc.ImagePath,  # ImagePath stores paths to images. In the 3LC Dashboard, the images will be displayed.
    "label": tlc.CategoricalLabel("label", classes=CLASSES),  # Label is just an integer, but we want to display it as a string
    "weight": tlc.SampleWeightSchema(),  # The weight of the sample, to be used for weighted training
}


def register_dataset_to_table(dataset_path, table_name, split_name):
    """
    Register images from a folder structure to a 3LC table.
    Each folder (chihuahua, muffin, undefined) corresponds to a class.
    
    Args:
        dataset_path: Path to the dataset (e.g., 'train128' or 'test128')
        table_name: Name for the 3LC table
        split_name: Name of the split ('train' or 'test')
    """
    dataset_path = Path(dataset_path)
    
    # Collect all images with their labels
    image_data = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = dataset_path / class_name
        if class_folder.exists():
            # Get all jpg images in the folder
            image_files = sorted(class_folder.glob('*.jpg'))
            print(f"Found {len(image_files)} images in {class_name} folder for {split_name} set")
            
            for img_path in image_files:
                image_data.append({
                    'path': str(img_path.absolute()),
                    'label': class_idx,  # Store as integer index (0=chihuahua, 1=muffin, 2=undefined)
                })
        else:
            print(f"Warning: {class_folder} does not exist")
    
    print(f"\nTotal images for {split_name} set: {len(image_data)}")
    print(f"  - Chihuahua: {sum(1 for x in image_data if x['label'] == 0)}")
    print(f"  - Muffin: {sum(1 for x in image_data if x['label'] == 1)}")
    print(f"  - Undefined: {sum(1 for x in image_data if x['label'] == 2)}")
    
    # Create table writer
    table_writer = tlc.TableWriter(
        table_name=table_name,
        dataset_name=DATASET_NAME,
        project_name=PROJECT_NAME,
        description=f"Chihuahua vs Muffin {split_name} set with {len(image_data)} images",
        column_schemas=schemas,
        if_exists="overwrite",
    )
    
    # Add rows to the table
    # Weight: 1.0 for labeled data (chihuahua, muffin), 0.0 for undefined
    for i, data in enumerate(image_data):
        label = data['label']
        
        # Set weight based on class: 1.0 for labeled, 0.0 for undefined
        weight = 1.0 if label in [0, 1] else 0.0
        
        table_writer.add_row({
            "id": i,
            "image": data['path'],
            "label": label,
            "weight": weight,
        })
    
    # Finalize the table
    table = table_writer.finalize()
    
    print(f"\n[OK] Created 3LC table: '{table_name}' with {len(image_data)} samples")
    
    # Print final statistics
    num_labeled = sum(1 for x in image_data if x['label'] in [0, 1])
    num_undefined = sum(1 for x in image_data if x['label'] == 2)
    
    print(f"\nFinal distribution:")
    print(f"  - Labeled samples (weight=1.0): {num_labeled}")
    print(f"  - Undefined samples (weight=0.0): {num_undefined}")
    
    print(f"  Table URL: {table.url}")
    
    return table

def main():
    """Main function to register both train and test sets."""
    
    # Base path to the Muffin dataset
    base_path = Path(__file__).parent
    
    print("=" * 60)
    print("Registering Chihuahua vs Muffin Dataset in 3LC Tables")
    print("=" * 60)
    
    # Register URL alias for portable paths
    # This allows data to be moved later and ALIASES can be updated - we don't want hardcoded URLs in the tables
    print("\nRegistering URL alias for portable paths...")
    tlc.register_project_url_alias(
        token="MUFFIN_DATA", 
        path=str(base_path.absolute()), 
        project=PROJECT_NAME
    )
    print(f"  [OK] Alias 'MUFFIN_DATA' -> {base_path.absolute()}")
    
    # Register train set
    print("\n[1/2] Registering TRAIN set...")
    print("-" * 60)
    train_path = base_path / 'train128'
    train_table = register_dataset_to_table(
        train_path, 
        table_name='train',
        split_name='train'
    )
    
    # Register test set
    print("\n[2/2] Registering TEST set...")
    print("-" * 60)
    test_path = base_path / 'test128'
    test_table = register_dataset_to_table(
        test_path,
        table_name='test',
        split_name='test'
    )
    
    print("\n" + "=" * 60)
    print("[OK] Successfully registered both tables!")
    print("=" * 60)
    print(f"\nTrain table: {train_table.url}")
    print(f"Test table:  {test_table.url}")
    print("\nYou can now use these tables in your 3LC workflows.")

if __name__ == '__main__':
    main()

