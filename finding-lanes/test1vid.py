import cv2
import numpy as np
import matplotlib.pyplot as plt

def makecoordinates(image,line_parameters):
	slope,intercept=line_parameters

	y1=image.shape[0]
	y2=int(y1*3/5)
	x1=int((y1-intercept)/slope)
	x2=int((y2-intercept)/slope)
	return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
	left_fit=[]
	right_fit=[]
	if lines is None:
		return None
	for line in lines:
		x1,y1,x2,y2=line.reshape(4)
		parameters=np.polyfit((x1,x2),(y1,y2),1)
		slope,intercept=parameters
		if slope<0:
			left_fit.append((slope,intercept))
		else:
			right_fit.append((slope,intercept))
	if len(left_fit)==0 or len(right_fit)==0:
		return np.array([])

	left_fit_average=np.average(left_fit,axis=0)
	right_fit_average=np.average(right_fit,axis=0)
	left_line=makecoordinates(image,left_fit_average)
	right_line=makecoordinates(image,right_fit_average)
	return np.array([left_line,right_line])




def canny(image):
	gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	blur=cv2.GaussianBlur(gray,(5,5),0)
	canny=cv2.Canny(blur,50,150) #gradient image and highlights the edges
	return canny

def region_of_interest(image):
	height=image.shape[0] # image .shape has m,n,l along y x and breadth
	triangle=np.array([[(200,height), (1100,height),(550,250)]])
	mask=np.zeros_like(image) # for an array of pixels with same dimensions as aaray but all pixel values are zero so it will be black
	cv2.fillPoly(mask,triangle,255)
	masked_image=cv2.bitwise_and(image,mask)
	return masked_image

def display_lines(image,lines):
	lined_image=np.zeros_like(image) # this creates a black image with the same dimensions as that of our road image
	if lines is not None: # if some hough lines were detected or grids were detected
		for line in lines: #iterated through the lines
			x1,y1,x2,y2=line.reshape(4) #since the line is a 2d array with 4 columns we reshape it
			cv2.line(lined_image,(x1,y1),(x2,y2),(255,0,0),10) #cv2.line draws a line on lined_image with the coordinated x1y1x2y2 with the color blue and thicknees 10
		return lined_image
	else:
		return image

#image= cv2.imread('test_image.jpg')
#lane_image=np.copy(image) #copy of pixel intensity array
#canny=canny(lane_image)
#cropped_image=region_of_interest(canny) #this is the cropped gradient image
#lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]), minLineLength=40,maxLineGap=5) #the last two arguments represent the dimensions of the grids in the hough space and the grid with the maximum intersections will be voted as the ro theeta valueed line equation that best represents our data,  then we have the last argument as the threshold which is the minimum number of intersections a grid should have to be accepted
#averaged_lines=average_slope_intercept(lane_image,lines)
#lined_image=display_lines(lane_image,averaged_lines)
#combo_image=cv2.addWeighted(lane_image,1,lined_image,1,1)
# imread is used to read our image and return it as a multi dimensional array containinig the relative intensities of each pixel in the image
#cv2.imshow("result",combo_image) #shows the image in a tab called results
#cv2.waitKey(0) #shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard


cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
	_, frame=cap.read() # frame is the image in the video at a particular second and the first one is just a boolean
	if frame is None:
		break
	canny_image=canny(frame)
	cropped_image=region_of_interest(canny_image) #this is the cropped gradient image
	lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]), minLineLength=40,maxLineGap=5) #the last two arguments represent the dimensions of the grids in the hough space and the grid with the maximum intersections will be voted as the ro theeta valueed line equation that best represents our data,  then we have the last argument as the threshold which is the minimum number of intersections a grid should have to be accepted
	averaged_lines=average_slope_intercept(frame,lines)
	lined_image=display_lines(frame,averaged_lines)
	combo_image=cv2.addWeighted(frame,0.8,lined_image,1,1)
 	# imread is used to read our image and return it as a multi dimensional array containinig the relative intensities of each pixel in the image
	cv2.imshow("result",combo_image) #shows the image in a tab called results
	if cv2.waitKey(1) & 0xFF ==ord('q'):#shows the result image for specified amount of milliseconds- if given as 0 it shows till we press a key on keyboard , 1 cuz we wait 1 ms between frames of video
		break

cap.release()
cv2.destroyAllWindows()


 
