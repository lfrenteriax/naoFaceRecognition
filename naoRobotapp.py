__author__ = 'Leo Fabiano'

from naoRobot import newRobot
from time import sleep
#myRobot=naoRobot()


pythonModule=''
myRobot=newRobot('192.168.1.58',simulate=False)
myRobot.say('Sono qui')

myRobot.LED_face_Blink(number=3)
faceDetected=0




myRobot.newfaceDatabase()

while 1:
   person='nesuno'
   person= myRobot.faceRec()
   if person!= 'none':

       myRobot.say(person)

