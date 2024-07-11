import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from evaluate_visualization import EvaluateVisualization
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report
import pandas as pd
import os
import json

class ResNetTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device='cuda', patience=10):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.evaluator = EvaluateVisualization()
        self.model_name = self.model.__class__.__name__
        self.model_save_dir = f'./model_res_{self.model_name}'
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.results_file = f'{self.model_save_dir}/All_epochs_results_{self.model_name}.json'
        self.scaler = GradScaler()
        self.early_stopping = EarlyStopping(patience=patience)
        self.best_metric_epoch = -1  # Initialize best_metric_epoch
        self.best_accuracy = 0

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
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
        best_accuracy = 0

        for epoch in range(num_epochs):
            # Training
            train_loss, train_accuracy = self._train_one_epoch()

            # Validation
            val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)

            # Test
            test_loss, test_accuracy = self._validate_one_epoch(self.test_dataloader)

            # Print progress
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

            # Update the training, validation, and testing loss and accuracy lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            test_accuracies.append(test_accuracy)

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
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.best_metric_epoch = epoch
                torch.save(self.model.state_dict(), f"{self.model_save_dir}/{self.model_name}_best.pth")

            # Step with the scheduler based on validation loss
            self.scheduler.step(val_loss)

            # Check early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        # Save all epoch results to a JSON file
        with open(self.results_file, 'w') as f:
            json.dump(all_epoch_res, f, indent=4)

        # Plotting loss curve and accuracy curve
        self.evaluator.plot_loss_curve(train_losses, val_losses)
        self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies)
        self.evaluator.plot_test_acc_loss_curve(test_losses, test_accuracies)

    def evaluate(self):
        self.model.load_state_dict(torch.load(f"{self.model_save_dir}/{self.model_name}_best.pth"))
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

        # Generate classification report
        class_report = classification_report(y_true_test, y_pred_test, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        class_report_df.to_csv(f'{self.model_save_dir}/classification_report_epoch_{self.best_metric_epoch+1}.csv', index=True)

        return test_loss, accuracy_test

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# class ResNetTrainer:
#     def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device='cuda', patience=10):
#         self.model = model.to(device)
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.test_dataloader = test_dataloader
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.device = device
#         self.evaluator = EvaluateVisualization()
#         self.model_name = self.model.__class__.__name__
#         self.model_save_dir = f'./model_res_{self.model_name}'
#         if not os.path.exists(self.model_save_dir):
#             os.makedirs(self.model_save_dir)
            
#         self.results_file = f'{self.model_save_dir}/All_epochs_results_{self.model_name}.json'
        
#         self.scaler = GradScaler()
#         self.early_stopping = EarlyStopping(patience=patience)
#         self.best_metric_epoch = -1  # Initialize best_metric_epoch

#     def _train_one_epoch(self):
#         self.model.train()
#         epoch_train_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         for inputs, labels in self.train_dataloader:
#             inputs, labels = inputs.to(self.device), labels.to(self.device)

#             self.optimizer.zero_grad()
#             with autocast():
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()

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
#         test_losses = []
#         train_accuracies = []
#         val_accuracies = []
#         test_accuracies = []
#         all_epoch_res = []
#         train_curr_acc_list = []
#         train_curr_loss_list = []
#         val_curr_acc_list = []
#         val_curr_loss_list = []
#         test_curr_acc_list = []
#         test_curr_loss_list = []
#         best_accuracy = 0

#         for epoch in range(num_epochs):
#             # Training
#             train_loss, train_accuracy = self._train_one_epoch()

#             # Validation
#             val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)

#             # Test
#             test_loss, test_accuracy = self.evaluate()

#             # Print progress
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

#             # Update the training, validation, and testing loss and accuracy lists
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             test_losses.append(test_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)
#             test_accuracies.append(test_accuracy)

#             # Store current accuracies and losses
#             train_curr_acc_list.append(train_accuracies)
#             train_curr_loss_list.append(train_losses)
#             val_curr_acc_list.append(val_accuracies)
#             val_curr_loss_list.append(val_losses)
#             test_curr_acc_list.append(test_accuracies)
#             test_curr_loss_list.append(test_losses)

#             epoch_data = {
#                 'epoch': epoch + 1,
#                 'training_loss': train_loss,
#                 'training_accuracy': train_accuracy,
#                 'validation_loss': val_loss,
#                 'validation_accuracy': val_accuracy,
#                 'testing_loss': test_loss,
#                 'testing_accuracy': test_accuracy,
#             }
#             all_epoch_res.append(epoch_data)

#             # Save best model
#             if best_accuracy < test_accuracy:
#                 best_accuracy = test_accuracy
#                 self.best_metric_epoch = epoch
#                 torch.save(self.model.state_dict(), f"{self.model_save_dir}/{self.model_name}_{epoch + 1}.pth")

#             # Step with the scheduler based on validation loss
#             self.scheduler.step(val_loss)
#             # self.scheduler.step()

#             # Check early stopping
#             self.early_stopping(val_loss)
#             if self.early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#         # Save all epoch results to a JSON file
#         with open(self.results_file, 'w') as f:
#             json.dump(all_epoch_res, f, indent=4)

#         # Plotting loss curve and accuracy curve
#         self.evaluator.plot_loss_curve(train_losses, val_losses)
#         self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies)
#         self.evaluator.plot_test_acc_loss_curve(test_losses, test_accuracies)

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
#         # Generate classification report
#         class_report = classification_report(y_true_test, y_pred_test, output_dict=True)
#         class_report_df = pd.DataFrame(class_report).transpose()
#         class_report_df.to_csv(f'{self.model_save_dir}/classification_report_epoch_{self.best_metric_epoch+1}.csv', index=True)

#         return test_loss, accuracy_test

# class EarlyStopping:
#     def __init__(self, patience=10, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_loss = None
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True

