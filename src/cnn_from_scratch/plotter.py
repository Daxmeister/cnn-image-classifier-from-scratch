import matplotlib.pyplot as plt
import numpy as np

class Plotter():

    def __init__(self):
        self.train_loss_per_epoch = []
        self.train_cost_per_epoch = []
        self.train_accuracy_per_epoch = []
        
        self.val_loss_per_epoch = []
        self.val_cost_per_epoch = []
        self.val_accuracy_per_epoch = []
        
        self.update_step_vector = [] #  to plot x axis
    
    def add_update_step(self, loss_train,  cost_train, accuracy_train, loss_val, cost_val, accuracy_val, update_step):
        self.train_loss_per_epoch.append(loss_train)
        self.train_cost_per_epoch.append(cost_train)
        self.train_accuracy_per_epoch.append(accuracy_train)
      
        self.val_loss_per_epoch.append(loss_val)
        self.val_cost_per_epoch.append(cost_val)
        self.val_accuracy_per_epoch.append(accuracy_val)
        
        self.update_step_vector.append(update_step)
    
    def plot(self, titletext, filename):

        if self.train_accuracy_per_epoch != None:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # TO account for update steps
        if self.update_step_vector != None:
            x_axis = self.update_step_vector
        else:
            x_axis = np.arange(len(self.train_loss_per_epoch))
        
        # Loss
        axs[0].plot(x_axis,self.train_loss_per_epoch, label="Train Loss")
        axs[0].plot(x_axis,self.val_loss_per_epoch, label="Test Loss")
        axs[0].set_title("Loss per Epoch ")
        if self.update_step_vector != None:
            axs[0].set_xlabel("Update Step")
        else:
            axs[0].set_xlabel("Epoch")    
        
        axs[0].set_ylabel("Loss")
        axs[0].set_ylim(bottom=0)
        axs[0].set_xlim(left=0)
        axs[0].legend()

        # Cost
        axs[1].plot(x_axis,self.train_cost_per_epoch, label="Train Cost")
        axs[1].plot(x_axis,self.val_cost_per_epoch, label="Test Cost")
        axs[1].set_title("Cost per Epoch ")
        if self.update_step_vector != None:
            axs[1].set_xlabel("Update Step")
        else:
            axs[1].set_xlabel("Epoch")    
        axs[1].set_ylabel("Cost")
        axs[1].set_ylim(bottom=0)
        axs[1].set_xlim(left=0)
        axs[1].legend()
        
        if self.train_accuracy_per_epoch != None:
            axs[2].plot(x_axis,self.train_accuracy_per_epoch, label="Train accuracy")
            axs[2].plot(x_axis,self.val_accuracy_per_epoch, label="Test accuracy")
            axs[2].set_title("Accuracy per Epoch ")
            axs[2].set_xlabel("Update Step")
            axs[2].set_ylabel("Accuracy")
            axs[2].set_ylim(bottom=0)
            axs[2].set_xlim(left=0)
            axs[2].legend()
            
        fig.text(0.02,0.01,s="Settings "+titletext)
        fig.tight_layout()
        fig.savefig(filename)
        print("len x", len(self.update_step_vector))
        plt.show()
