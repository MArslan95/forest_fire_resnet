import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluate_visualization import EvaluateVisualization
from torch.cuda.amp import GradScaler, autocast
import os
import json


class ResNetTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device='cuda'):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.evaluator = EvaluateVisualization()
        self.model_name = self.model.__class__.__name__
        self.model_save_dir = f'./model_res_{self.model.__class__.__name__}'
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            
        self.results_file = f'{self.model_save_dir}/All_epochs_results_{self.model_name}.json'
        
        self.scalar = GradScaler()

    def _train_one_epoch(self):
        self.model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            self.scalar.scale(loss).backward()
            self.scalar.step(self.optimizer)
            self.scalar.update()

            epoch_train_loss += loss.item() * inputs.size(0)
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        accuracy_train = correct_train / total_train
        return epoch_train_loss / total_train, accuracy_train

    def _validate_one_epoch(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        accuracy_val = correct_val / total_val
        return val_loss / total_val, accuracy_val

    def train(self, num_epochs):
        train_losses = []
        val_losses = []
        test_losses = []
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []
        all_epoch_res = []
        train_curr_acc_list = []
        train_curr_loss_list = []
        val_curr_acc_list = []
        val_curr_loss_list = []
        test_curr_acc_list = []
        test_curr_loss_list = []
        best_accuray = 0
        best_metric_epoch = -1

        for epoch in range(num_epochs):
            # Training
            train_loss, train_accuracy = self._train_one_epoch()

            # Validation
            val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)

            # Test
            test_loss, test_accuracy = self.evaluate()

            # Print progress
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

            # Update the training, validation, and testing loss and accuracy lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            test_accuracies.append(test_accuracy)

            # Calculate current accuracy and loss for training, validation, and testing
            curr_train_accuracy = sum(train_accuracies) / len(train_accuracies)
            curr_train_loss = sum(train_losses) / len(train_losses)
            curr_val_accuracy = sum(val_accuracies) / len(val_accuracies)
            curr_val_loss = sum(val_losses) / len(val_losses)
            curr_test_accuracy = sum(test_accuracies) / len(test_accuracies)
            curr_test_loss = sum(test_losses) / len(test_losses)

            # Store current accuracies and losses
            train_curr_acc_list.append(curr_train_accuracy)
            train_curr_loss_list.append(curr_train_loss)
            val_curr_acc_list.append(curr_val_accuracy)
            val_curr_loss_list.append(curr_val_loss)
            test_curr_acc_list.append(curr_test_accuracy)
            test_curr_loss_list.append(curr_test_loss)

            epoch_data = {
                'epoch': epoch + 1,
                'training_loss': train_loss,
                'training_accuracy': train_accuracy,
                'validation_loss': val_loss,
                'validation_accuracy': val_accuracy,
                'testing_loss': test_loss,
                'testing_accuracy': test_accuracy,
            }
            all_epoch_res.append(epoch_data)

            # Save best model
            if best_accuray < test_accuracy:
                best_accuray = test_accuracy
                best_metric_epoch = epoch
                torch.save(self.model.state_dict(), f"{self.model_save_dir}/{self.model.__class__.__name__}_{epoch + 1}.pth")

        # Save all epoch results to a JSON file
        with open(self.results_file, 'w') as f:
            json.dump(all_epoch_res, f, indent=4)

        # Plotting loss curve and accuracy curve
        self.evaluator.plot_loss_curve(train_losses, val_losses)
        self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies)
        self.evaluator.plot_test_acc_loss_curve(test_losses,test_accuracies)

    def evaluate(self):
        self.model.eval()
        y_true_test = []
        y_pred_test = []
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                _, predicted_test = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()

                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted_test.cpu().numpy())

                # Calculate test loss
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

        accuracy_test = correct_test / total_test
        test_loss /= total_test  # Calculate average test loss

        return test_loss, accuracy_test
    
    
    

# # Training class
# class ResNetTrainer:
#     def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer,scheduler, device='cuda'):
#         self.model = model.to(device)
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.test_dataloader = test_dataloader
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.device = device
#         self.scheduler = scheduler
#         self.evaluator = EvaluateVisualization()

#     def _train_one_epoch(self):
#         self.model.train()
#         epoch_train_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         for inputs, labels in self.train_dataloader:
#             inputs, labels = inputs.to(self.device), labels.to(self.device)

#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, labels)
#             loss.backward()
#             self.optimizer.step()

#             epoch_train_loss += loss.item() * inputs.size(0)
#             _, predicted_train = torch.max(outputs.data, 1)
#             total_train += labels.size(0)
#             correct_train += (predicted_train == labels).sum().item()

#         accuracy_train = correct_train / total_train
#         return epoch_train_loss / total_train, accuracy_train

#     def _validate_one_epoch(self, dataloader):
#         self.model.eval()
#         val_loss = 0.0
#         correct_val = 0
#         total_val = 0

#         with torch.no_grad():
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 val_loss += loss.item() * inputs.size(0)

#                 _, predicted_val = torch.max(outputs.data, 1)
#                 total_val += labels.size(0)
#                 correct_val += (predicted_val == labels).sum().item()

#         accuracy_val = correct_val / total_val
#         return val_loss / total_val, accuracy_val
    
#     def train(self, num_epochs):
#         train_losses = []
#         val_losses = []
#         test_losses = []  # New: To store test losses
#         train_accuracies = []
#         val_accuracies = []
#         test_accuracies = []  # New: To store test accuracies

#         for epoch in range(num_epochs):
#             # Training
#             train_loss, train_accuracy = self._train_one_epoch()

#             # Validation
#             val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)

#             # Test
#             test_loss, test_accuracy = self.evaluate()

#             # Print progress
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

#             # Update the training and validation loss and accuracy lists
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             test_losses.append(test_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)
#             test_accuracies.append(test_accuracy)

#         # Plotting loss curve and accuracy curve
#         self.evaluator.plot_loss_curve(train_losses, val_losses)  # Include test_losses
#         self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies )  # Include test_accuracies
#         self.evaluator.plot_test_acc_loss_curve(test_accuracies,test_losses)


#     def evaluate(self):
#         self.model.eval()
#         y_true_test = []
#         y_pred_test = []
#         correct_test = 0
#         total_test = 0
#         test_loss = 0.0

#         with torch.no_grad():
#             for inputs, labels in self.test_dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)

#                 _, predicted_test = torch.max(outputs.data, 1)
#                 total_test += labels.size(0)
#                 correct_test += (predicted_test == labels).sum().item()

#                 y_true_test.extend(labels.cpu().numpy())
#                 y_pred_test.extend(predicted_test.cpu().numpy())

#                 # Calculate test loss
#                 loss = self.criterion(outputs, labels)
#                 test_loss += loss.item() * inputs.size(0)

#         accuracy_test = correct_test / total_test
#         test_loss /= total_test  # Calculate average test loss

#         return test_loss, accuracy_test