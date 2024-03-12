# from abc import ABC, abstractmethod

# class Robot(ABC):
#     @abstractmethod
#     def move_forward(self, distance: float):
#         pass

#     @abstractmethod
#     def move_backward(self, distance: float):
#         pass

#     @abstractmethod
#     def turn_left(self, angle: float):
#         pass

#     @abstractmethod
#     def turn_right(self, angle: float):
#         pass

#     @abstractmethod
#     def stop(self):
#         pass




#             print(f"bot mode {mode}")
#         self.mode = mode
#         self.robot = self.getFromDef('agent')
#         self.step(self.timestep)
#         self.keyboard = self.getKeyboard()
#         self.keyboard.enable(self.timestep)
#         self.compass = self.getCompass('compass')
#         # self.leftcamera = self.getCamera('lefteye')
#         # self.rightcamera = self.getCamera('righteye')
#         self.rangeFinderNode = self.getFromDef('range-finder')
#         self.rangeFinder = self.getRangeFinder('range-finder')
#         self.leftBumper = self.getTouchSensor('bumper_left')
#         self.rightBumper = self.getTouchSensor('bumper_right') 
#         self.collided = tf.Variable(np.zeros(2, np.int32))
#         self.display = self.getDisplay('display')
#         self.rotationField = self.robot.getField('rotation')
#         # self.gps = self.getGPS('gps')
#         # self.robotLocation = tf.Variable(tf.zeros((2)))
#         self.leftMotor = self.getMotor('left wheel motor')
#         self.rightMotor = self.getMotor('right wheel motor')
#         self.leftPositionSensor = self.getPositionSensor('left wheel sensor')
#         self.rightPositionSensor = self.getPositionSensor('right wheel sensor')
#         self.leftBumper.enable(self.timestep)
#         self.rightBumper.enable(self.timestep)
#         self.leftPositionSensor.enable(self.timestep)
#         self.rightPositionSensor.enable(self.timestep)
#         # self.leftcamera.enable(self.timestep)
#         # self.leftcamera.recognitionEnable(self.timestep)
#         # self.rightcamera.enable(self.timestep)
#         # self.rightcamera.recognitionEnable(self.timestep)
#         self.rangeFinder.enable(self.timestep)
#         self.compass.enable(self.timestep)