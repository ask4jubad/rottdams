# Make sure to have the add-on "Connectivity/ZMQ remote API server"
# running in CoppeliaSim
#
# Following are required and should be installed:
# pip install pyzmq
# pip install cbor
from time import process_time_ns
from multiprocessing import Process

# from turtle import distance
from logging import NullHandler
# from turtle import delay
# from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BinaryTournamentSelection, PolynomialMutation, SBXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization import Plot
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from imageai.Detection import VideoObjectDetection, ObjectDetection
from doctest import OutputChecker
from matplotlib import image
from zmqRemoteApi import RemoteAPIClient
from PIL import Image

import csv
import numpy
import os.path
import os
import cv2
import time
import math, random
import numpy as np

summer = 0 #sum all failures
dist_function_checker= [] #to hold boolean values for when event condition is satified or not.
config_space_holder = [] #list of list containing all rgb values during when event condition is satisfied or not 

print('Program started')

client = RemoteAPIClient()
# client.setsynchronous(True)
client.setStepping(True)
sim = client.getObject('sim')

# print(sim.getSimulationState() != sim.simulation_paused)

defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
sim.setInt32Param(sim.intparam_idle_fps, 0)

# gripperHandle = sim.getObjectHandle("RG2_openCloseLoopDummyA")
simBase=sim.getObjectHandle('LBR_iiwa_7_R800')
simTip=sim.getObjectHandle('IkTip')
simTarget=sim.getObjectHandle('IkTarget')
humanHandle = sim.getObjectHandle("Bill")
visionSensorHandle = sim.getObjectHandle("Vision_sensor") #client to get sensor handle
# cameraStream = sim.getObjectHandle("DefaultCamera")
# human_hand_left = sim.getObjectHandle("IkTip_leftHand") #client to get left hand tip handle
human_hand_right = sim.getObjectHandle("IkTip_rightHand") #client to get left hand tip handle
lightHandleA = sim.getObjectHandle("DefaultLightA")
lightHandleB = sim.getObjectHandle("DefaultLightB")
lightHandleC = sim.getObjectHandle("DefaultLightC")
lightHandleD = sim.getObjectHandle("DefaultLightD")


execution_path = "/home/jubril/Downloads"

with open("result_RS.txt", 'a') as output:
    pass


class Simulation():
    
    def __init__(self, variables_minimums, variables_maximums):

        self.difA1 = random.uniform(variables_minimums[0], variables_maximums[0])
        self.difA2 = random.uniform(variables_minimums[0], variables_maximums[0])
        self.difA3 = random.uniform(variables_minimums[0], variables_maximums[0])

        self.timestep = random.randint(1, 50) #time to start actuating the robot
        self.vel = random.uniform(0.05, 0.5) #velocity of the robot
    
        self.timestepprime = random.randint(1, 50) #time to start actuating the human
        self.velprime = random.uniform(0.05, 0.5) #velocity of the human
        self.objective = 0
        self.variables = [self.difA1, self.difA2, self.difA3, self.timestep, self.vel, self.timestepprime, self.velprime]
    
    def emergency_stop(self):
        print ("Emergency situation detected")
        sim.stopSimulation()

    
    # def evaluate(self, solution: FloatSolution) -> FloatSolution:
    def evaluate(self):

    
        # try:

        simdata = []
        startTime = time.time()

        stateA,x,difA,specA = sim.getLightParameters(lightHandleA)

        sim.setLightParameters(lightHandleA, 1 , [0, 0, 0], [self.difA1, self.difA2, self.difA3], [0,0,0])

        sim.setInt32Signal("waitTimeBill", self.timestepprime)
        sim.setInt32Signal("waitTimeRobot", self.timestep)
        sim.setFloatSignal("velBill", self.velprime)
        sim.setFloatSignal("velRobot", self.vel)
        sim.startSimulation()
        time.sleep(2)


        dist = np.Inf

		#continue to sense while simulation is active
        while sim.getSimulationState() != sim.simulation_stopped:
  
            img, resX, resY = sim.getVisionSensorCharImage(visionSensorHandle)

            #==========================================================================================================================
            '''This part is for processing Vision sensor images. Vision sensors currently not working as expected'''
            #convert np array buffer to an image

            img = np.frombuffer(img, dtype=np.uint8).reshape(resX, resY, 3)
            # img = sim.transformbuffer(img, sim.buffer_uint8rgb, 1, 0, sim.buffer_uint8)

            #convert from BGR to RBG
            img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
                        
            lower = np.array([100, 100, 90])   
            upper = np.array([188, 170, 147])

            # upper = np.array([151, 255, 151])
            mask = cv2.inRange(img, lower, upper)
            
            result, distance_left, objectPair = sim.checkDistance(simTip, human_hand_right, 0)

            #determine if the robot is moving at time of collision (or when evaluation happens)
            #if robot is moving give ctual value, if not, return large value
            #create check velocity functuion in child script and call here
            #make hazards less if experiment returns large hazard size
            jointspeedRobot = sim.getFloatSignal("jointvelRobot")
            if jointspeedRobot is None:
                jointspeedRobot = 0
            # print(jointspeedRobot)
            

            if cv2.countNonZero(mask) > 0:
            # and distance_left[6] < 0.05
                    
                Simulation.emergency_stop(self)
                # self.objective = distance_left[6]
                # return self.objective, self.variables

            else:
        
                output = cv2.bitwise_and(img, img, mask = mask)

                # cv2.imshow('Masked', output)
                # cv2.waitKey(3)
  
            
            client.step()
            _, distance_left, _ = sim.checkDistance(simTip, human_hand_right, 0)

            # print("distance between actors is", distance_left[6])
            if (distance_left[6] < dist):
                dist = distance_left[6]
        
        if (dist < 0.05) and (jointspeedRobot <= 0):
            with open("result_RS.txt", 'a') as output:
                output.write("False_fail: ")
                output.write(str(self.variables).strip("[]"))
                output.write('\n')

        elif (dist < 0.05) and (jointspeedRobot > 0):
            with open("result_RS.txt", 'a') as output:
                output.write("Fail: ")
                output.write(str(self.variables).strip("[]"))
                output.write('\n')
        else:
            with open("result_RS.txt", 'a') as output:
                output.write("Pass: ")
                output.write(str(self.variables).strip("[]"))
                output.write('\n')
        
        cv2.destroyAllWindows() 


        self.objective = dist
        sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

        print('Program ended')


        #to set default light A to on, and then it's diffuse and specular components to their defualt
        sim.setLightParameters(lightHandleA, 1 , [0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5])
        sim.stopSimulation()

        # first column saves the current fitness value
        simdata.append(self.objective)
        # second column stores individual simulation time
        simdata.append((time.time() - startTime))

        # simdataCollection.append(simdata)

        with open('run_details_RS.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(simdata)

        return self.objective, self.variables

def main():

    n = 0
    number_of_runs = 400
    startTimer = time.time()


    while n < number_of_runs:
        # light(3), robot(2), human(2)
        problem = Simulation([0, 1, 0.05, 1, 0.1], [1, 50, 0.5, 50, 0.5])
        solution = problem.evaluate()
        # print(n)
        n += 1
    
    print("Algorithm: {}".format("Random Search"))
    print("Problem: {}".format("Simulation Problem"))
    print("Solution: {}".format(problem.variables))
    print("Fitness: {}".format(problem.objective))
    print("Computing time: {}".format(time.time() - startTimer))
    # cv2.waitKey(0) 

    #closing all open windows 
    cv2.destroyAllWindows()
    exit()
    
if __name__ == "__main__":
    main()
