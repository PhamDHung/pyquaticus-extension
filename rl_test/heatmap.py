import matplotlib.pyplot as plt
import time
import sys
import subprocess
import numpy as np
from matplotlib.patches import Rectangle
import argparse
def get_position_heatmap(data_file, image):
    f = open(data_file, 'r')
    distance = []
    distance2 = []
    x = []
    y = []
    time = []
    lines = f.readlines()
    for line in lines:
        s = line.split(',')
        time.append(float(s[0]))
        x.append(float(s[1]))
        y.append(float(s[2]))

    x.append(-10)
    y.append(-44)
    x.append(170)
    y.append(123)
    fig1, ax  = plt.subplots()
    plt.hist2d(x, y, bins=25)
    #plt.xlabel('X (Meters)')
    #plt.ylabel('Y (Meters)')
    plt.xlim([-10, 170])
    plt.ylim([-10, 90])
    plt.axis('equal')

    ax.add_patch(Rectangle((0, 0), 160, 80,edgecolor = 'white',fill=False,lw=1))
    ax.add_patch(Rectangle((0, 0), 80, 80, edgecolor = 'white', fill = False, lw=1))
    #plt.axhline(y = 80, xmin=0, xmax = 160, color='w', linewidth=1)
    #plt.axhline(y=0, xmin = 0, xmax=160, color='w', linewidth=1)
    #plt.axvline(x=0, ymin = 0, ymax=80, color='w', linewidth=1)
    #plt.axvline(x=80, ymin = 0, ymax = 80, color='w',linewidth=1)
    #plt.axvline(x = 160, ymin = 0, ymax=80, color='w', linewidth=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    #print("X: ", x, " y: ",y)
    print("Name: ", image + 'png')
    plt.savefig(image)
    plt.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy a trained policy in a 2v2 PyQuaticus environment')
    parser.add_argument('position', help='Please enter the path to the model you would like to load in')
    parser.add_argument('image', help='Please enter the file name for the saved actions')
    args = parser.parse_args()
    get_position_heatmap(args.position, args.image)
    #action = 41
    #for i in range(20):
     #   val  = action * 500
     #   if val == 0: 
     #       get_position_heatmap("actions_1.txt", 'image_1')
     #   else:
     #       get_position_heatmap("actions_" + str(val)+".txt", "image_"+str(val))
     #   action += 1
