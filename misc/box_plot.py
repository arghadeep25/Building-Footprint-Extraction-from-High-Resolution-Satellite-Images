import matplotlib.pyplot as plt
import numpy as np

fcn = [
       [0.38, 0.39, 0.47, 0.50, 0.53, 0.54, 0.56, 0.58, 0.70, 0.71],#austin
       [0.30, 0.53, 0.59, 0.63, 0.63, 0.63, 0.71, 0.73, 0.75, 0.76],#chicago
       [0.19, 0.26, 0.31, 0.37, 0.40, 0.48, 0.49, 0.51, 0.63, 0.65],#kitsap
       [0.29, 0.31, 0.37, 0.38, 0.44, 0.48, 0.49, 0.51, 0.52, 0.53],#tyrol
       [0.47, 0.53, 0.59, 0.60, 0.67, 0.69, 0.74, 0.75, 0.79, 0.80]#vienna
       ]
segnet = [
          [0.41, 0.48, 0.50, 0.53, 0.55, 0.57, 0.59, 0.63, 0.71, 0.73],
          [0.43, 0.53, 0.59, 0.63, 0.63, 0.65, 0.67, 0.71, 0.71, 0.79],
          [0.31, 0.33, 0.37, 0.45, 0.51, 0.51, 0.57, 0.63, 0.65, 0.68],
          [0.43, 0.44, 0.46, 0.46, 0.55, 0.57, 0.63, 0.65, 0.67, 0.68],
          [0.53, 0.67, 0.70, 0.71, 0.71, 0.73, 0.77, 0.81, 0.81, 0.83]
          ]
unet = [
          [0.49, 0.50, 0.57, 0.63, 0.68, 0.70, 0.77, 0.78, 0.81, 0.83],
          [0.64, 0.67, 0.70, 0.71, 0.78, 0.80, 0.86, 0.87, 0.89, 0.91],
          [0.44, 0.47, 0.50, 0.53, 0.58, 0.60, 0.61, 0.62, 0.74, 0.75],
          [0.60, 0.61, 0.66, 0.67, 0.68, 0.72, 0.75, 0.76, 0.77, 0.79],
          [0.67, 0.70, 0.71, 0.76, 0.78, 0.82, 0.83, 0.89, 0.91, 0.93]
          ]
deepunet = [
            [0.45, 0.51, 0.57, 0.58, 0.62, 0.64, 0.71, 0.73, 0.76, 0.77],
            [0.50, 0.51, 0.57, 0.63, 0.67, 0.69, 0.77, 0.78, 0.83, 0.85],
            [0.49, 0.52, 0.53, 0.55, 0.56, 0.60, 0.63, 0.67, 0.67, 0.69],
            [0.52, 0.57, 0.61, 0.63, 0.68, 0.70, 0.77, 0.78, 0.80, 0.81],
            [0.61, 0.65, 0.68, 0.71, 0.75, 0.79, 0.85, 0.89, 0.91, 0.95]
           ]
# pspnet = [
#           [0.41, 0.48, 0.50, 0.53, 0.55, 0.57, 0.59, 0.63, 0.71, 0.73],
#           [0.43, 0.53, 0.59, 0.63, 0.63, 0.65, 0.67, 0.71, 0.71, 0.79],
#           [0.31, 0.33, 0.37, 0.45, 0.51, 0.51, 0.57, 0.63, 0.65, 0.68],
#           [0.31, 0.37, 0.41, 0.50, 0.53, 0.54, 0.67, 0.71, 0.78, 0.79],
#           [0.31, 0.37, 0.41, 0.50, 0.53, 0.54, 0.67, 0.71, 0.78, 0.79]
#           ]
# mask_rcnn = [
#              [0.41, 0.48, 0.50, 0.53, 0.55, 0.57, 0.59, 0.63, 0.71, 0.73],
#              [0.43, 0.53, 0.59, 0.63, 0.63, 0.65, 0.67, 0.71, 0.71, 0.79],
#              [0.31, 0.33, 0.37, 0.45, 0.51, 0.51, 0.57, 0.63, 0.65, 0.68],
#              [0.31, 0.37, 0.41, 0.50, 0.53, 0.54, 0.67, 0.71, 0.78, 0.79],
#              [0.31, 0.37, 0.41, 0.50, 0.53, 0.54, 0.67, 0.71, 0.78, 0.79]
#             ]
ticks = ['Austin', 'Chicago', 'Kitsap', 'Tyrol', 'Vienna']

def set_box_color(bp, border, color):
    plt.setp(bp['boxes'], color=border)
    plt.setp(bp['fliers'], color=border)
    plt.setp(bp['whiskers'], color=border)
    plt.setp(bp['caps'], color=border)
    plt.setp(bp['medians'], color=border)

    for patch in bp['boxes']:
        patch.set(facecolor=color)


fig = plt.figure(figsize = (8, 4))
fcn_box = plt.boxplot(fcn,
                      positions=np.array(range(len(fcn)))*2.0-0.5,
                      sym='',
                      widths=0.15,
                      patch_artist=True,
                      )
segnet_box = plt.boxplot(segnet,
                         positions=np.array(range(len(segnet)))*2.0-0.2,
                         sym='',
                         widths=0.15,
                         patch_artist=True)
unet_box = plt.boxplot(unet,
                       positions=np.array(range(len(unet)))*2.0+0.1,
                       sym='',
                       widths=0.15,
                       patch_artist=True)
deepunet_box = plt.boxplot(deepunet,
                           positions=np.array(range(len(deepunet)))*2.0+0.4,
                           sym='',
                           widths=0.15,
                           patch_artist=True)
# pspnet_box = plt.boxplot(pspnet,
#                          positions=np.array(range(len(pspnet)))*2.0+0.4,
#                          sym='',
#                          widths=0.1,
#                          patch_artist=True)
# mask_rcnn_box = plt.boxplot(mask_rcnn,
#                             positions=np.array(range(len(pspnet)))*2.0+0.6,
#                             sym='',
#                             widths=0.1,
#                             patch_artist=True)
set_box_color(fcn_box, 'red', '#f5b5b5')
set_box_color(segnet_box, 'green', '#96eb91')
set_box_color(unet_box, 'blue', '#a3baf7')
set_box_color(deepunet_box, '#9f00de', '#d197e8')
# set_box_color(pspnet_box, '#b3b000', '#e6e35e')
# set_box_color(mask_rcnn_box, '#ad5100', '#fca75b')
# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='#f5b5b5', label='FCN')
plt.plot([], c='#96eb91', label='SegNet')
plt.plot([], c='#a3baf7', label='U-Net')
plt.plot([], c='#d197e8', label='Deep U-Net')
# get rid of border
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
# plt.box(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
plt.grid(axis='y')
plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('iou_boxplot.png',bbox_inches='tight')
plt.show()
