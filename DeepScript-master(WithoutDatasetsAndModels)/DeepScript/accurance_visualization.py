#typs and tricks for python http://pythonic.eu/fjfi/posts/matplotlib.html
#typs and tricks for csv https://docs.python.org/3/library/csv.html#csv.writer

import matplotlib.pyplot as plt
import numpy as np
import csv





def acc_vis(file_name, count_epoch, title, label_x, label_y): #visualize the accuracy from concrete file only concrete number of epochs
    acc = [] # array for the accuracy
    with open(file_name) as csvfile: # open CSV file
        reader = csv.reader(csvfile) # read line
        for row in reader:
            roowlist=row[0].split(" ") # split each accuracy
            map(float,roowlist)
            acc.append(roowlist)
    fig = plt.figure()
    x = np.arange(len(acc[count_epoch]))
    y=[]
    y.append('0') # add on the beginning accuracy zero
    for cnt in range(0, count_epoch):
        nbs=str(acc[count_epoch-1][cnt])
        y.append(nbs) # add all needed accuracies to the array y
    print(y)

    # add_axes- how big will be square with the graph
    # left, down, width, height (in relative numbers from 0 to 1)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(x, y)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.set_title(title)
    axes.grid(True) # add grid
    name = title.replace(' ','_') + ".pdf" # make a name for the saving file
    fig.savefig(name) # save the accuracy plot
    return

if __name__ == '__main__':
    acc_vis('Accurancyvgg16.csv', 10,'VGG16 accuracy for 10 epochs', 'Epoch [-]', 'Accuracy [-]')
    acc_vis('Accurancyresnet50.csv', 10, 'ResNet50 accuracy for 10 epochs', 'Epoch [-]', 'Accuracy [-]')
    acc_vis('AccurancyWhitening.csv', 25, 'ResNet50 accuracy for 25 epochs with whitening', 'Epoch [-]', 'Accuracy [-]')