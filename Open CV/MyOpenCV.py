import numpy as np
import cv2

# Handle Mouse Events in OpenCV
# img = np.ones([512,512,3])
# cv2.imshow('image',img)   #window name 'image' should be same everywhere

# # left click prints coordinates of that point
# # right click prints bgr channel of image at that particular point
# def click_event(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,', ',y)
#         font = cv2.FONT_HERSHEY_COMPLEX
#         text = str(x) + ', '+str(y)
#         cv2.putText(img,text,(x,y),font,0.5,(255,255,0),2)
#         cv2.imshow('image',img)
#     if event == cv2.EVENT_RBUTTONDOWN:
#         blue = img[y,x,0]
#         green = img[y,x,1]
#         red = img[y,x,2]
#         font = cv2.FONT_HERSHEY_COMPLEX
#         text = str(blue) + ', '+str(green) + ', '+str(red)
#         cv2.putText(img,text,(x,y),font,0.5,(255,0,255),2)
#         cv2.imshow('image',img)

# # join lines where clicked
# points= []
# def click_event(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN: #left click prints coordinates of that point
#         cv2.circle(img,(x,y),3,(0,0,255),-1)
#         points.append((x,y))
#         if len(points)>=2:
#             cv2.line(img,points[-1],points[-2],(255,0,0),5)
#         cv2.imshow('image',img)

# cv2.setMouseCallback('image',click_event)




# # copy paste one part of image from one place to another
# img = cv2.imread('messi5.jpg')

# print(img.shape) #returns a tuple of number of rows, columns,and channels
# print(img.size) #returns total number of pixel accessed
# print(img.dtype) #returns image datatype

# b,g,r=cv2.split(img) #split into bgr channel
# img = cv2.merge((b,g,r)) #merge bgr channel

# ball = img[280:340,330:390] 
# img[273:333,100:160] = ball

# cv2.imshow('image',img)



# # Add two image
# img = cv2.imread('messi5.jpg')
# img2=cv2.imread('opencv-logo.png')

# ball = img[280:340,330:390] 
# img[273:333,100:160] = ball

# img = cv2.resize(img,(512,512))
# img2 = cv2.resize(img2,(512,512))

# # output_image=cv2.add(img,img2)
# output_image=cv2.addWeighted(img,0.9,img2,0.1,0) #0.9 and 0.1 are weight of each image

# cv2.imshow('image',output_image)



# # Trackbar
# def callbackfun(x):
#     print(x)


# img = np.zeros((300,512,3),np.uint8)
# cv2.namedWindow('image')

# cv2.createTrackbar('B','image',0,255,callbackfun)
# cv2.createTrackbar('G','image',0,255,callbackfun)
# cv2.createTrackbar('R','image',0,255,callbackfun)

# switch = '0 : OFF\n 1 : ON'
# cv2.createTrackbar('switch','image',0,1,callbackfun)

# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(1) & 0xFF
#     if k==27:
#         break

#     b = cv2.getTrackbarPos('B','image')
#     g = cv2.getTrackbarPos('G','image')
#     r = cv2.getTrackbarPos('R','image')
#     s = cv2.getTrackbarPos('switch','image')

#     if s==0:
#         img[:] = 0
#     else:
#         img[:] = [b,g,r]


# # Object Detection and object tracking using HSV color space
# def nothing(x):
#     pass

# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH","Tracking",0,255,nothing)
# cv2.createTrackbar("LS","Tracking",0,255,nothing)
# cv2.createTrackbar("LV","Tracking",0,255,nothing)
# cv2.createTrackbar("UH","Tracking",255,255,nothing)
# cv2.createTrackbar("US","Tracking",255,255,nothing)
# cv2.createTrackbar("UV","Tracking",255,255,nothing)

# while True:
#     frame = cv2.imread('smarties.png')

#     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

#     l_h = cv2.getTrackbarPos("LH","Tracking")
#     l_s = cv2.getTrackbarPos("LS","Tracking")
#     l_v = cv2.getTrackbarPos("LV","Tracking")

#     u_h = cv2.getTrackbarPos("UH","Tracking")
#     u_s = cv2.getTrackbarPos("US","Tracking")
#     u_v = cv2.getTrackbarPos("UV","Tracking")

#     l_b = np.array([l_h,l_s,l_v])
#     u_b = np.array([u_h,u_s,u_v])

#     mask = cv2.inRange(hsv,l_b,u_b)
#     res = cv2.bitwise_and(frame,frame,mask=mask)

#     cv2.imshow("frame",frame)
#     cv2.imshow("mask",mask)
#     cv2.imshow("res",res)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# # Using your camera for detection

# def nothing(x):
#     pass

# cap = cv2.VideoCapture(0)

# cv2.namedWindow("Tracking")
# cv2.createTrackbar("LH","Tracking",0,255,nothing)
# cv2.createTrackbar("LS","Tracking",0,255,nothing)
# cv2.createTrackbar("LV","Tracking",0,255,nothing)
# cv2.createTrackbar("UH","Tracking",255,255,nothing)
# cv2.createTrackbar("US","Tracking",255,255,nothing)
# cv2.createTrackbar("UV","Tracking",255,255,nothing)

# while True:
#     _,frame = cap.read()


#     hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

#     l_h = cv2.getTrackbarPos("LH","Tracking") #82
#     l_s = cv2.getTrackbarPos("LS","Tracking") #51
#     l_v = cv2.getTrackbarPos("LV","Tracking") #51

#     u_h = cv2.getTrackbarPos("UH","Tracking") #133
#     u_s = cv2.getTrackbarPos("US","Tracking") #255
#     u_v = cv2.getTrackbarPos("UV","Tracking") #255

#     l_b = np.array([l_h,l_s,l_v])
#     u_b = np.array([u_h,u_s,u_v])

#     mask = cv2.inRange(hsv,l_b,u_b)
#     res = cv2.bitwise_and(frame,frame,mask=mask)

#     cv2.imshow("frame",frame)
#     cv2.imshow("mask",mask)
#     cv2.imshow("res",res)

#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cap.release()





cv2.waitKey(0)
cv2.destroyAllWindows()