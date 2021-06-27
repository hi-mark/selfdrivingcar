import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.layers import Flattenstore
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow.keras.models import save_model,load_model
from imagegrab_fordata import grab_screen
import csv
import math
import pygame


IMAGEWIDTH=320
IMAGEHEIGHT=240
rows=[]


display_width = 640
display_height = 480
x =  0
y = 0
black = (0,0,0)
white = (255,255,255)
green = (0, 255, 0)
blue = (0, 0, 128)

def carla_preprocess(image):
	image=image[100:200,:,:]
	image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
	image=cv2.GaussianBlur(image,(3,3),0)
	image=cv2.resize(image,(200,66))
	image=image/255
	return image



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
		elif slope==0:
			continue
		else:
			right_fit.append((slope,intercept))
	if len(left_fit)==0 or len(right_fit)==0:
		return np.array([])

	left_fit_average=np.average(left_fit,axis=0)
	right_fit_average=np.average(right_fit,axis=0)
	left_line=makecoordinates(image,left_fit_average)
	right_line=makecoordinates(image,right_fit_average)
	return np.array([left_line,right_line])



def display_lines(image,lines):
	slope1=0
	slope2=0
	
	if lines is not None: # if some hough lines were detected or grids were detected
		for line in lines: #iterated through the lines
			x1,y1,x2,y2=line.reshape(4) #since the line is a 2d array with 4 columns we reshape it
			try:
				cv2.line(image,(x1,y1),(x2,y2),(255,0,0),10) #cv2.line draws a line on lined_image with the coordinated x1y1x2y2 with the color blue and thicknees 10
			except:
				continue
		return image
	else:
		return image

def region_of_interest(image):
	height=image.shape[0] # image .shape has m,n,l along y x and breadth
	width=image.shape[1]
	#plt.imshow(image)
	#plt.show()
	triangle=np.array([[(0,height),(0,120),(150,100),(width,120),(width,height)]])
	#triangle=np.array([[(0,380), (width,380),(300,200)]])
	
	mask=np.zeros_like(image) # for an array of pixels with same dimensions as aaray but all pixel values are zero so it will be black
	cv2.fillPoly(mask,triangle,255)
	masked_image=cv2.bitwise_and(image,mask)
	return masked_image

def edge_detection(image):
	gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	blur=cv2.GaussianBlur(gray,(5,5),0)
	canny=cv2.Canny(blur,50,150) #gradient image and highlights the edges
	return canny

def printrows():
	global rows
	with open('tl.csv', 'w', newline='') as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerows(rows)


def store_image_data(imagedata,ourvehicle,model,model2,gameDisplay,clock):
	i=np.array(imagedata.raw_data)
	i2=i.reshape((IMAGEHEIGHT,IMAGEWIDTH,4))
	i3=i2[:, :, : 3]

	
	v=ourvehicle.get_velocity()
	current_speed=float(math.sqrt(v.x**2 + v.y**2 + v.z**2))
	throttlevalue=1.0-(current_speed/6.0)
	copy_i3=np.copy(i3)
	copy_i3_2=np.copy(i3)
	brakevalue=0.0



	finalimage=carla_preprocess(copy_i3)
	newimg=np.array([finalimage])
	#print(newimg.shape)
	prediction=model.predict(newimg)[0]
	steervalue=float(prediction[0])

	ourvehicle.apply_control(carla.VehicleControl(throttle=throttlevalue,steer=steervalue))

	if ourvehicle.is_at_traffic_light():
		print("here")
		traffic_light = ourvehicle.get_traffic_light()
		if traffic_light.get_state() == carla.TrafficLightState.Red or traffic_light.get_state() == carla.TrafficLightState.Red:
			ourvehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=1.0))
			brakevalue=1.0


	font = pygame.font.Font('freesansbold.ttf', 12)
	text = font.render("Steer: "+str(steervalue), True, green, blue)
	text2=font.render("Throttle: "+str(throttlevalue), True, green, blue)
	text3=font.render("Brake: "+str(brakevalue), True, green, blue)
	text4=font.render("Speed: "+str(current_speed)+" m/s", True, green, blue)

	textRect = text.get_rect()
	textRect2 = text2.get_rect()
	textRect3 = text3.get_rect()
	textRect4 = text4.get_rect()
	textRect.center = (100, display_height-display_height//8)
	textRect2.center = (100, 440)
	textRect3.center = (460, 420)
	textRect4.center = (510, 440)


	carImg=pygame.surfarray.make_surface(i3)
	carImg = pygame.transform.smoothscale(carImg, (480,640)) 
	rotated_image = pygame.transform.rotate(carImg, -90)
	gameDisplay.blit(rotated_image, (x,y))
	gameDisplay.blit(text, textRect)
	gameDisplay.blit(text2, textRect2)
	gameDisplay.blit(text3, textRect3)
	gameDisplay.blit(text4, textRect4)
	pygame.display.update()
	clock.tick(60)
'''

	copy_i3_2=copy_i3_2[80:220,0:170,:]
	copy_i3_2=copy_i3_2/255
	lightimage=np.array([copy_i3_2])
	predictionoflight=np.around(model2.predict(lightimage)[0])
	lightstate=''
	if ourvehicle.is_at_traffic_light():
		traffic_light = ourvehicle.get_traffic_light()
		if traffic_light.get_state() == carla.TrafficLightState.Red or traffic_light.get_state() == carla.TrafficLightState.Red:
			ourvehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=1.0))
		ourvehicle.apply_control(carla.VehicleControl(throttle=throttlevalue,brake=0.0))
	
'''


	#canny=edge_detection(copy_i3_2)
	#cropped_image=region_of_interest(canny) #this is the cropped gradient image
	#lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]), minLineLength=40,maxLineGap=5) #the last two arguments represent the dimensions of the grids in the hough space and the grid with the maximum intersections will be voted as the ro theeta valueed line equation that best represents our data,  then we have the last argument as the threshold which is the minimum number of intersections a grid should have to be accepted 
	#averaged_lines=average_slope_intercept(copy_i3_2,lines)
	#lined_image=display_lines(copy_i3_2,averaged_lines)



	#combo_image=cv2.addWeighted(lane_image,1,lined_image,1,1)
	








	







def store_image_data1(imagedata,ourvehicle,model):
	global rows
	traffic_light=ourvehicle.get_traffic_light()
	if traffic_light is not None:
		if traffic_light.get_state() == carla.TrafficLightState.Red:
			i=np.array(imagedata.raw_data)
			i2=i.reshape((IMAGEHEIGHT,IMAGEWIDTH,4))
			i3=i2[:, :, : 3]
			directory=r'D:\WindowsNoEditor\PythonAPI\examples\tt'
			os.chdir(directory)
			cv2.imwrite('out%08d.jpg' % imagedata.frame,i3)
			directory2=r'D:\WindowsNoEditor\PythonAPI\examples'
			os.chdir(directory2)
			imagepath='D:\WindowsNoEditor\PythonAPI\examples\tt\out%08d.jpg' % imagedata.frame
			rows.append([imagepath,np.array([1,0,0,0])])
			traffic_light.set_state(carla.TrafficLightState.Green)
			traffic_light.set_green_time(4.0)

		if traffic_light.get_state() == carla.TrafficLightState.Yellow:
			i=np.array(imagedata.raw_data)
			i2=i.reshape((IMAGEHEIGHT,IMAGEWIDTH,4))
			i3=i2[:, :, : 3]
			directory=r'D:\WindowsNoEditor\PythonAPI\examples\tt'
			os.chdir(directory)
			cv2.imwrite('out%08d.jpg' % imagedata.frame,i3)
			directory2=r'D:\WindowsNoEditor\PythonAPI\examples'
			os.chdir(directory2)
			imagepath='D:\WindowsNoEditor\PythonAPI\examples\tt\out%08d.jpg' % imagedata.frame
			rows.append([imagepath,np.array([0,1,0,0])])
			
	return 0





'''
	i=np.array(imagedata.raw_data)
	i2=i.reshape((IMAGEHEIGHT,IMAGEWIDTH,4))
	i3=i2[:, :, : 3]
	if ourvehicle.is_at_traffic_light():
		traffic_light = ourvehicle.get_traffic_light()
		if traffic_light.get_state() == carla.TrafficLightState.Red:
			traffic_light.set_state(carla.TrafficLightState.Green)
			traffic_light.set_green_time(4.0)
	#copy_i3=np.copy(i3)
	c=ourvehicle.get_control()
	directory=r'D:\WindowsNoEditor\PythonAPI\examples\dataset'
	os.chdir(directory)
	cv2.imwrite('out%08d.jpg' % imagedata.frame,i3)
	directory2=r'D:\WindowsNoEditor\PythonAPI\examples'
	os.chdir(directory2)
	currentsteer=c.steer
	imagepath='D:\WindowsNoEditor\PythonAPI\examples\dataset\out%08d.jpg' % imagedata.frame
	rows.append([imagepath,currentsteer])
	count=count+1
	if count%1000==0:
		print(count)
	return 0
'''



#MAIN CODE---------------START
if __name__=="__main__":

	try:
	    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
	        sys.version_info.major,
	        sys.version_info.minor,
	        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
	except IndexError:
	    pass

	import carla
	actorList=[] #like other vehicles pedestrians
	try:
		#connect client to the server
		#the world is the carla simulation and we as clients can load and reload the simulation/world
		#there are actors such as pedestrians,sensors and vehicles in the world and the layouts for those actors is called blueprint
		client=carla.Client("localhost",2000)
		client.set_timeout(3.0)
		world=client.get_world()
		print("connected")

		blueprint_library=world.get_blueprint_library()
		#vehicle_bp= blueprint_library.filter('vehicle.audi.a2')[0] #we choose a vehicle filtered accoring to id
		vehicle_bp=blueprint_library.filter('vehicle.harley-davidson.low_rider')[0]
		#spawn_point=random.choice(world.get_map().get_spawn_points())
		#print(spawn_point)

		spawn_point=carla.Transform(carla.Location(x=26.940020, y=302.570007, z=0.500000), carla.Rotation(pitch=0.000000, yaw=-179.999634, roll=0.000000))
		
		
		
		
		ourvehicle=world.spawn_actor(vehicle_bp,spawn_point) #make/spawn our vehicle in the simulation
		 #since we want to control our vehicle ourselves
		actorList.append(ourvehicle)
		weather = carla.WeatherParameters( cloudiness=0.0, precipitation=0.0, precipitation_deposits=0.0,wind_intensity=0.0,sun_azimuth_angle=70.0, sun_altitude_angle=70.0,fog_density=0.0, fog_distance=0.0, fog_falloff=0.0, wetness=0.0)
	
		#weather = carla.WeatherParameters.CloudyNoon
		world.set_weather(weather)

		#we are making use of the camera sensor and the collision sensor
		cam_bp=blueprint_library.find("sensor.camera.rgb")
		#obstacle_bp=blueprint_library.find("sensor.other.obstacle")
		cam_bp.set_attribute("image_size_x",f"{IMAGEWIDTH}")
		cam_bp.set_attribute("image_size_y",f"{IMAGEHEIGHT}")
		cam_bp.set_attribute("fov","110")
		cam_bp.set_attribute("enable_postprocess_effects","true")
		#cam_bp.set_attribute("sensor_tick","1.0")
		#we need to spawn the rgb camera on the vehicle cuz its attached to it
		#relative_spawn_point=carla.Transform(carla.Location(x=-5.5,z=2.5)) #this location is relative to our vehicle
		

		relative_spawn_point=carla.Transform(carla.Location(x=-1.5, z=2.5))
		#relative_spawn_point=carla.Transform(carla.Location(x=-0.5,z=2.5)) #this location is relative to our vehicle
		camera_sensor=world.spawn_actor(cam_bp,relative_spawn_point,attach_to=ourvehicle)
		#obstacle_sensor=world.spawn_actor(obstacle_bp,relative_spawn_point,attach_to=ourvehicle)
		actorList.append(camera_sensor)

		#actorList.append(obstacle_sensor)
		model=load_model('D:\WindowsNoEditor\PythonAPI\examples\save_model-rgbdata-nvdia-1output-balanced\saved_model')
		model2=load_model('D:\WindowsNoEditor\PythonAPI\examples\save_model-trafficdata-4output\saved_model')
		

		pygame.init()
		gameDisplay = pygame.display.set_mode((display_width,display_height))

		pygame.display.set_caption('Capstone Project')
		count=1
		clock = pygame.time.Clock()
		crashed = False
		while not crashed:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					crashed = True
			gameDisplay.fill(white)
			camera_sensor.listen(lambda imagedata: store_image_data(imagedata,ourvehicle,model,model2,gameDisplay,clock))
			time.sleep(240)
		pygame.quit()
		quit()
		
		pass
	finally:
		for actor in actorList:
			actor.destroy()
		print("All actors destroyed!")



#MAIN CODE END---------------