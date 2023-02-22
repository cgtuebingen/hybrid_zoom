import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio as imgio

fresnel_lens = np.zeros([1920,1080])

x = np.arange(1920, dtype=float)
y = np.arange(1080, dtype=float)

x -= 960
y -= 540
pitch = 8*10**-6
x *= pitch
y *= pitch

X, Y = np.meshgrid(x,y)

k = 2*np.pi/(550*10**-9)
focal_length_input = input("Please enter the focal length(mm):")
focal_length = float(focal_length_input) * 10**-3
phi = k/(2*focal_length) * (np.square(X) + np.square(Y))

Z = np.mod(phi,2*np.pi)

plt.imshow(Z)

filename = "results/focallength_" + str(focal_length_input) + "mm.png"
imgio.imwrite(filename,Z)
