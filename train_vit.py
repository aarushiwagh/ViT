import os
import torch
import argparse
import data_setup, engine, utils
import vit

from torchvision import transforms

# Setup hyperparameters
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-tr', '--train_dir', help = 'Enter the path of your train directory')
parser.add_argument('-ts', '--test_dir', help = 'Enter the path of your test directory')
parser.add_argument('-e', '--epochs', default = 5, help='No of epochs you want to train for')
parser.add_argument('-b', '--batch_size', default=32)
parser.add_argument('-h', '--hidden_units', default=10, help='No of neurons in the model')
#parser.add_argument('-lr', '--learning_rate', default=0.001)

args = parser.parse_args()

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
#LEARNING_RATE = args.learning_rate

# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = vit.ViT(num_classes = len(class_names)).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                             betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                             weight_decay=0.3)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="vit.pth")