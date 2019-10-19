import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# file_name = sys.argv[1]

if os.path.exists('deepunet_training_data.txt'):
    test_file = open('deepunet_training_data.txt', "r")
    lines = test_file.readlines()
    new_epochs= []
    new_training_loss = []
    new_validation_loss = []
    new_training_accuracy = []
    new_validation_accuracy = []
    for line in lines:
        new_epochs.append(int(line.split(' ')[0]))
        new_training_loss.append(float(line.split(' ')[1]))
        new_validation_loss.append(float(line.split(' ')[2]))
        new_training_accuracy.append(float(line.split(' ')[3]))
        new_validation_accuracy.append(float(line.split(' ')[4]))
    test_file.close()

fig = plt.figure(figsize = (6, 4))
N = len(new_epochs)
print(new_training_loss)
plt.plot(np.arange(0, N), new_training_loss, label="Training Loss")
plt.plot(np.arange(0, N), new_validation_loss, label="Validation Loss")
plt.plot(np.arange(0, N), new_training_accuracy, label="Training Accuracy")
plt.plot(np.arange(0, N), new_validation_accuracy, label="Validation Accuracy")
plt.xlim((0, int(N)))
plt.ylim((0, 1.0))
plt.title("Training Loss and Accuracy", fontsize=13)
plt.xlabel("Epochs", fontsize = 12)
plt.ylabel("Loss/Accuracy", fontsize = 12)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
plt.savefig("deepunet_training.png", bbox_inches='tight')
plt.show()
