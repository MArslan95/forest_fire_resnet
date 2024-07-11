import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation and Visualization class
class EvaluateVisualization:
    @staticmethod
    def plot_loss_curve(train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        # plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0, 1) 
        plt.legend()
        plt.show()
        
    @staticmethod
    def plot_accuracy_curve(train_accuracies, val_accuracies):  # Renamed from plot_acc_curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        # plt.plot(test_accuracies, label='Test Accuracy')
        plt.title('Training and ValidationAccuracy')  # Changed from 'Loss' to 'Accuracy'
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy') 
        plt.ylim(0, 1) # Changed from 'Loss' to 'Accuracy'
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_test_acc_loss_curve(test_accuracies,test_losses):  # Renamed from plot_acc_curve
        plt.figure(figsize=(10, 5))
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.plot(test_losses, label='Test Loss')
        plt.title(' Test Accuracy and Loss ')  # Changed from 'Loss' to 'Accuracy'
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1) # Changed from 'Loss' to 'Accuracy'
        plt.legend()
        plt.show()

    # @staticmethod
    # def plot_confusion_matrix(y_true, y_pred, class_names):
    #     conf_matrix = confusion_matrix(y_true, y_pred)
    #     plt.figure(figsize=(8, 8))
    #     sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    #     plt.title('Confusion Matrix')
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')
    #     plt.show()


