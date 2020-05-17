# Unrestricted Adversarial Examples

This repository contains the Two-Class Unambiguous images required for evaluation of the warmup to the Unrestricted Adversarial Examples contest.


### Installation
```bash
git clone https://github.com/google/unrestricted-adversarial-examples
pip install -e unrestricted-adversarial-examples/bird-or-bicycle

bird-or-bicycle-download
```

### Usage
```python
import bird_or_bicycle 
train_dataset_folder = bird_or_bicycle.get_dataset('train')

# Use a pytorch directory-based dataset loader
import torchvision.datasets
train_dataset = torchvision.datasets.ImageFolder(
    train_dataset_folder)
    
# Or a keras one
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dataset_folder)
```

# Splits

- clean train -> 125 images/class
- clean test -> 125 images/class
- extras -> 13750 images/class
