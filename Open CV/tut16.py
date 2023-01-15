# matplotlib with opencv

from matplotlib import pyplot as plt
import cv2 as cv

img = cv.imread('lena.jpg',-1)
cv.imshow('image',img) #openCV reads in BGR format

img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img) #matplotlib reads in RBG format
# plt.xticks([])
# plt.yticks([])

plt.show() # xy axis shows value of pixel

cv.waitKey(0)
cv.destroyAllWindows()

# see tut14 matplotlib