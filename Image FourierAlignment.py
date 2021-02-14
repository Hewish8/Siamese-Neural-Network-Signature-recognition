import os
import cv2
import numpy as np
import imreg_dft as ird

# the TEMPLATE
im0 = cv2.imread(r'C:\Users\Hewish\Desktop\SIgnature Samples\p 1.jpg', 0)
# the image to be transformed
im1 = cv2.imread(r'C:\Users\Hewish\Desktop\SIgnature Samples\p 3.jpg', 0)

#Ensure both images are of same sizeC:\Users\Hewish\Desktop\SIgnature Samples
width, height = im0.shape[1], im0.shape[0]
im1 = cv2.resize(im1, (width,height))

#First image is reference. All operations on second image in this libary
result = ird.similarity(im0, im1, numiter=3)

assert "timg" in result
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt
    ird.imshow(im0, im1, result['timg'])
    plt.show()

#print(result)
print('angle rotated:' + str(result['angle']))
print('scale factor:' + str(result['scale']))
print('translation:' + str(result['tvec']))

img = cv2.threshold(result['timg'], 200, 255, cv2.THRESH_BINARY)[1] 
#print(img)
plt.imshow(img, cmap = "gray")
plt.show()