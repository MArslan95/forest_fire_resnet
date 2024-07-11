import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import torch.optim.lr_scheduler as lr_scheduler
from evaluate_visualization import EvaluateVisualization
# from resnet_50 import ResNet50, ResidualBlock
from resnet50_train import ResNetTrainer, EarlyStopping

def main():
    # Data augmentation transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Increased jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset directoriesc
    train_data_dir = "./dataset/Training"
    valid_data_dir = "./dataset/Test"
    test_data_dir = "./dataset/Test"

    # Create datasets and dataloaders
    train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(valid_data_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)

    print(f"Number of training samples: {len(train_dataloader.dataset)}")
    print(f"Number of validation samples: {len(val_dataloader.dataset)}")
    print(f"Number of testing samples: {len(test_dataloader.dataset)}")

    # Create ResNet18 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Freeze the initial layers
    for  param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    # Modify the classifier
    model.fc = nn.Sequential(
        nn.Linear(num_features, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(4096, 2)
    )
    # Ensure only the classifier parameters are trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
   
    # Create dynamic scheduler
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,mode='min', threshold=1e-4,)

    # Training and evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    trainer = ResNetTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=10  # Adjusted early stopping patience
    )

    num_epochs = 100
    trainer.train(num_epochs)
    trainer.evaluate()

if __name__ == "__main__":
    main()

# from evaluate_visualization import EvaluateVisualization
# from resnet_model_101 import ResNet101v2, ResidualBlock
# from resnet_50 import ResNet50, ResidualBlock
# from resnet50_train import ResNetTrainer, EarlyStopping
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms, datasets
# from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
# import torch.optim.lr_scheduler as lr_scheduler

# def main():
#     # Data augmentation transformations
#     train_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     valid_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Dataset directories
#     train_data_dir = "./dataset/Training"
#     valid_data_dir = "./dataset/Test"
#     test_data_dir = "./dataset/Test"

#     # Create datasets and dataloaders
#     train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
#     val_dataset = datasets.ImageFolder(valid_data_dir, transform=valid_transforms)
#     test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)
    
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
#     val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
#     test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)

#     print(f"Number of training samples: {len(train_dataloader.dataset)}")
#     print(f"Number of validation samples: {len(val_dataloader.dataset)}")
#     print(f"Number of testing samples: {len(test_dataloader.dataset)}")

#     # Create ResNet18 model
#     model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#     # Freeze all the layers
#     for name ,param in model.parameters():
#         if"layer4" not in name:
#             param.requires_grad = False

#     num_features = model.fc.in_features
#     # Modify the classifier
#     model.fc = nn.Sequential(
#         nn.Linear(num_features, 4096),
#         nn.ReLU(),
#         nn.Dropout(p=0.6),
#         nn.Linear(4096, 4096),
#         nn.ReLU(),
#         nn.Dropout(p=0.6),
#         nn.Linear(4096, 2)
#     )

#     # Ensure only the classifier parameters are trainable
#     for param in model.fc.parameters():
#         param.requires_grad = True
    
#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)
   
#     # Create scheduler
#     # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
#     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

#     # Training and evaluation
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(device)

#     trainer = ResNetTrainer(
#         model=model,
#         train_dataloader=train_dataloader,
#         val_dataloader=val_dataloader,
#         test_dataloader=test_dataloader,
#         criterion=criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         device=device,
#         patience=10  # Adjusted early stopping patience
#     )

#     num_epochs = 100
#     trainer.train(num_epochs)
#     trainer.evaluate()

# if __name__ == "__main__":
#     main()

# from custom_dataset import CustomDataset
# from evaluate_visualization import EvaluateVisualization
# from resnet_model_101 import ResNet101v2, ResidualBlock
# from resnet_50 import ResNet50, ResidualBlock
# from resnet_trainer import ResNetTrainer
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms,datasets,models
# import torch.optim.lr_scheduler as lr_scheduler

# def main():
#     # Define transformations for data augmentation
#     # transform = transforms.Compose([
#     #     transforms.RandomResizedCrop(224,224),
#     #     transforms.RandomHorizontalFlip(),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     # ])
#     # Data augmentation transformations
#     train_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     valid_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Create custom dataset and DataLoader
#     train_data_dir = "./dataset/train"
#     valid_data_dir = "./dataset/valid"
#     test_data_dir = "./dataset/test"
#     # custom_dataset = CustomDataset(data_dir, transform=transform)
#     # classes = custom_dataset.classes
#     # print("Classes:", classes)
#     # train_size = int(0.8 * len(custom_dataset))
#     # val_size = len(custom_dataset) - train_size
#     train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
#     val_dataset = datasets.ImageFolder(valid_data_dir, transform=valid_transforms)
#     test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)
    
#     train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=8)
#     val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False,num_workers=8)
#     test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False,num_workers=8)
#     # # Create test dataset and DataLoader
#     # test_dataset = CustomDataset("./dataset/Test", transform=transform)
#     print(f"Number of training samples: {len(train_dataloader.dataset)}")
#     print(f"Number of validation samples: {len(val_dataloader.dataset)}")
#     print(f"Number of testing samples: {len(test_dataloader.dataset)}")

#     # Create ResNet101v2 model
#     # models = ResNet101v2(num_classes =2)
#     # Create Resnet 50 Model
#     resnet50_model = ResNet50(num_classes=2)
#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     # Optimizer Adam and SGD for Resnet101v2
#     # optimizer = optim.Adam(resnet50_model.parameters(), lr= 1e-05,weight_decay= 1e-05)
#     # optimizer = optim.SGD(resnet101v2_model.parameters(), lr=0.005,momentum=0.9, weight_decay=0.0001)
    
#     # Optimizer Adam and SGD for Resnet50
#     # optimizer = optim.Adam(resnet50_model.parameters(), lr=0.00001)
#     # optimizer = optim.Adam(resnet50_model.parameters(), lr= 1e-05,weight_decay= 1e-05)
#     optimizer = optim.SGD(resnet50_model.parameters(), lr=1e-01, momentum=0.9, weight_decay=0.0001)
   
#     # Create scheduler
#     # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)   
#     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 

#     # Training and evaluation
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'# Fix typo in 'is_available'
#     print(device)
#     trainer = ResNetTrainer(
#         model=resnet50_model,
#         train_dataloader=train_dataloader,
#         val_dataloader=val_dataloader,
#         test_dataloader=test_dataloader,
#         criterion=criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         device=device
#     )

#     num_epochs = 100
#     trainer.train(num_epochs)
#     trainer.evaluate()

# if __name__ == "__main__":
#     main()
    





# from evaluate_visualization import EvaluateVisualization
# from resnet_model_101 import ResNet101v2, ResidualBlock
# from resnet_50 import ResNet50, ResidualBlock
# from resnet50_train import ResNetTrainer, EarlyStopping
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms, datasets
# from torchvision.models import resnet50,ResNet50_Weights,resnet18,ResNet18_Weights
# import torch.optim.lr_scheduler as lr_scheduler

# def main():
#     # Data augmentation transformations
#     train_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     valid_transforms = transforms.Compose([
#         transforms.Resize(254),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Dataset directories
#     train_data_dir = "./dataset/Training"
#     valid_data_dir = "./dataset/Test"
#     test_data_dir = "./dataset/Test"

#     # Create datasets and dataloaders
#     train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
#     val_dataset = datasets.ImageFolder(valid_data_dir, transform=valid_transforms)
#     test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)
    
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
#     val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
#     test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)

#     print(f"Number of training samples: {len(train_dataloader.dataset)}")
#     print(f"Number of validation samples: {len(val_dataloader.dataset)}")
#     print(f"Number of testing samples: {len(test_dataloader.dataset)}")

#     # Create ResNet50 model
#     # resnet50_model = ResNet50(num_classes=2)
#     model= resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#     # Freeze all the layers
#     for param in model.parameters():
#         param.requires_grad = False

#     num_features =  model.fc.in_features
#     # Modify the classifier
#     model.fc = nn.Sequential(
#         nn.Linear(num_features,4096),
#         # nn.Linear(25088, 4096),
#         nn.ReLU(),
#         nn.Dropout(p=0.5),
#         nn.Linear(4096, 4096),
#         nn.ReLU(),
#         nn.Dropout(p=0.5),
#         nn.Linear(4096, 2)
#     )

#     # Ensure only the classifier parameters are trainable
#     for param in model.fc.parameters():
#         param.requires_grad = True
    

#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
   
#     # Create scheduler
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
#     # Alternative scheduler example (comment out the above scheduler line if you use this)
#     # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

#     # Training and evaluation
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(device)

#     trainer = ResNetTrainer(
#         model=model,
#         train_dataloader=train_dataloader,
#         val_dataloader=val_dataloader,
#         test_dataloader=test_dataloader,
#         criterion=criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         device=device,
#         patience=150 # Early stopping patience
#     )

#     num_epochs = 100
#     trainer.train(num_epochs)
#     trainer.evaluate()

# if __name__ == "__main__":
#     main()
