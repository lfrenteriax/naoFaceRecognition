# coding: utf-8
__author__ = 'Leo Fabiano'

from facerec.feature import Fisherfaces
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facedet.detector import *
from PIL import Image
import numpy as np
import sys, os
import time
from faceRecognizer import *
#sys.path.append("../..")
import cv2
import multiprocessing
import cv2
from naoqi import ALProxy
from naoqi import ALBroker
from naoqi import ALModule
import vision_definitions
import  array
from random import random
from names import *

model = PredictableModel(Fisherfaces(), NearestNeighbor())

global speech

LED_EAR_FLUSH_UNTIL_FLAG = False
LED_FACE_FLUSH_UNTIL_FLAG = False

# ----------> Face Led List <----------
FaceLedList = ["FaceLed0", "FaceLed1", "FaceLed2", "FaceLed3",
			   "FaceLed4", "FaceLed5", "FaceLed6", "FaceLed7"]
# ----------> Ear Led List <----------
RightEarLedList = ["RightEarLed1", "RightEarLed2", "RightEarLed3", "RightEarLed4",
				   "RightEarLed5", "RightEarLed6", "RightEarLed7", "RightEarLed8",
				   "RightEarLed9", "RightEarLed10"]
LeftEarLedList = ["LeftEarLed1", "LeftEarLed2", "LeftEarLed3", "LeftEarLed4",
				  "LeftEarLed5", "LeftEarLed6", "LeftEarLed7", "LeftEarLed8",
				  "LeftEarLed9", "LeftEarLed10"]
# ----------> Color List <----------
ColorList = ['red', 'white', 'green', 'blue', 'yellow', 'magenta', 'cyan'] # fadeRGB()的预设值


class newRobot:
    def __init__(self,ip='127.0.0.1',port=9559,simulate=True):

        self.vocabulary = ['fine','ok','a',	'b',	'c',	'd',	'e',	'f',
                           'g',	'h',	'i',	'j',	'k',	'l',	'm',	'n',
                           'o',	'p',	'q',	'r',	's',	't',	'u',	'v',	'w',
                           'x',	'y',	'z',	'0',	'1',	'2',	'3',	'4',	'5',	'6',	'7',	'8',	'9','si','no']
        self.ip=ip
        self.port=port
        self.simulate=simulate
        self.camResolution = 2
        self.camColorSpace = 11
        self.camFps = 30
        self.speechThr=50
        self.__faceRecLoaded=False
        self.__runRecog=False
        self.names=nameList()
        self.init()
        self.faceDetInit()
        self.faceRecInit()
        self.speechInit()
        self.gotoPosture("Sit")
        self.motionProxy.setStiffnesses('Body',0.0)

        self.__frontTactilTouched=0
        self.__middleTactilTouched=0
        self.__rearTactilTouched=0

    def init(self):
        if self.simulate==False:

            self.motionProxy = ALProxy("ALMotion", self.ip, self.port)
            self.memProxy = ALProxy("ALMemory", self.ip,self.port)
            self.camProxy = ALProxy("ALVideoDevice", self.ip,  self.port)
            self.tts = ALProxy("ALTextToSpeech", self.ip,  self.port)
            self.postureProxy = ALProxy("ALRobotPosture", self.ip, self.port)
            self.touchProxy = ALProxy("ALTouch", self.ip, self.port)
            self.audioProxy = ALProxy("ALAudioPlayer", self.ip, self.port)
            self.nameId = self.camProxy.subscribe("python_GVM", self.camResolution, self.camColorSpace, self.camFps)
            self.ledsProxy = ALProxy("ALLeds", self.ip, self.port)
            try:

                self.faceProxy = ALProxy("ALFaceDetection", self.ip, self.port)
                self.__faceProxyLooaded=True

            except:
                self.__faceProxyLooaded=False
                print('No ALFaceDetection loaded.. Maybe simulator is run')

            try:
                self.speechProxy = ALProxy("ALSpeechRecognition",self.ip, self.port)
                self.__speechProxyLooaded=True
            except:
                self.__speechProxyLooaded=False
                print('speechProxy no Looaded, Maybe simulator is run')



            # call method




        else:
           pass
           # self.camSimProxy=cv2.VideoCapture(0)
    def LED_face_ON(self):
        self.ledsProxy.on("FaceLeds")

    def LED_face_OFF(self):
        self.ledsProxy.off("FaceLeds")
    def LED_face_SwitchColor(self, color='white', delay=1, duration=0.1):
		
		self.LED_face_Color( color, duration)
		time.sleep(delay)
		self.LED_face_Color( 'white', duration)
    def LED_face_Color( self,color='white', duration=0.1):
        for led in FaceLedList:
            self.ledsProxy.post.fadeRGB(led, color, duration)
    def LED_face_Blink(self, number=1, duration=0.2):
        LedValueList = [0, 0, 1, 0,0, 0, 1, 0]
        for num in range(number):
            for i in range(len(FaceLedList)):
                self.ledsProxy.post.fade(FaceLedList[i], LedValueList[i], duration)
        time.sleep(0.1)
        self.ledsProxy.fade("FaceLeds", 1, duration)

    def LED_face_RandomColor(self, number=5, duration=1):

        for i in range(number):
			rRandTime = random.uniform(0.0, 2.0)
			self.ledsProxy.fadeRGB("FaceLeds",
				256 * random.randint(0,255) + 256*256 * random.randint(0,255) + random.randint(0,255),
				rRandTime)
			time.sleep(random.uniform(0.0, duration))

	def LED_face_Color(color='white', duration=0.1):
		
		for led in FaceLedList:
		    self.ledsProxy.post.fadeRGB(led, color, duration)

    def speechInit(self):
        self.speechStop()
        self.setSpeechLenguage()

    def speechStart(self):
         try:
            self.speechProxy.subscribe("speech")
         except:
            pass
    def speechStop(self):
        for subscriber, period, prec in self.speechProxy.getSubscribersInfo():
            self.speechProxy.unsubscribe(subscriber)

    def setSpeechLenguage(self,lenguage='Italian'):
        if self.simulate==False:
            if self.__speechProxyLooaded:
                try:
                    self.speechProxy.setLanguage(lenguage)
                except:
                    print('set lenaguage error')
                self.setSpeechVoc()
    def setSpeechVoc(self):
        self.speechProxy.setVocabulary(self.vocabulary+self.names.get(), False)
        try:
            self.speechProxy.setVocabulary(self.vocabulary+self.names.get(), False)
        except:
            print('No vocabulary loaded...')

    def getTouch(self,item='FrontTactilTouched'):
        if self.simulate==False:
            status = self.memProxy.getData(item)
        else:
             status=raw_input('Put 1 for button '+str(item)+' pressed, otherwise put 0')
        return (int(status))



    def getName(self):
        '''
        for name in self.names:
            self.say('Si, il tuo nome :')
            time.sleep(0.5)
            self.say(name)
            time.sleep(2)
            self.say('premi il button dal centro sulla mia testa')
            time.sleep(0.50)
            self.say('si no, premi il front Buton sulla mia testa')
            ok=cancel=0
            while not (ok) and not(cancel):
                ok = (self.getTouch('MiddleTactilTouched'))
                cancel=(self.getTouch('FrontTactilTouched'))
                print(ok,cancel)
            if ok:
               break
        '''
        name=''
        self.say('Ciao come ti chiami')
        word=''
        while 1:
            while  word=='':
                word=self.wordDetecded()
            print(word)
            tmp_name=word
            time.sleep(0.5)
            word=''
            self.say('Il tuo nome e? ')
            time.sleep(0.5)
            self.say(tmp_name)

            while  word=='':
                word=self.wordDetecded()
            if word=='si':
                name=tmp_name
                break
            else:
                self.say('Per favore rippete il tuo nome..')
                word=''
                nome=''


        self.say('Fatto..')
        return name



    def takePicture(self):
        picture=''
        if self.simulate==True:
            picture=self.pictureSimucam()
        else:
            picture=self.pictureRobotcam()

        return picture


    def pictureRobotcam(self):

        naoImage=self.camProxy.getImageRemote(self.nameId)
         # Create a PIL Image from our pixel array.

        array = naoImage[6]
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        image = np.fromstring(array, np.uint8).reshape( imageHeight, imageWidth, 3 )
        opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

  # Save the image.

        #zip(*[iter(img_bytes)]*3)
        return opencvImage

    def clik(self):
        self.LED_face_SwitchColor(color='green')
        fileID = self.audioProxy.post.playFile("/var/persistent/home/nao/naoqi/app/audio/camera-shutter-click-01.mp3", 1.0, 1.0)
    def pictureSimucam(self):
        import random
        listing=os.listdir('face_test')

        ret, img = self.camSimProxy.read()
        #img=cv2.imread('face_test/'+random.choice(listing))
        return img

    def say(self,text='anyone'):
        import pyttsx
        engine = pyttsx.init()
        voices = engine.getProperty('voices')

        engine.setProperty('voice',voices[3].id)
        engine.say(text)
        engine.runAndWait()
        if self.simulate==False:
            self.tts.say(text)





    def faceDetInit(self,faceCascadeClassifierPath="./data/haarcascades/haarcascade_frontalface_default.xml",eyeCascadeClassifierPath='./data/haarcascades/haarcascade_eye.xml',
                    noseCascadeClassifierPath= './data/haarcascades/haarcascade_mcs_nose.xml',mouthCascadeClassifierPath='./data/haarcascades/haarcascade_mcs_mouth.xml'):
        self.faceCascadeClassifierPath=faceCascadeClassifierPath
        self.eyeCascadeClassifierPath = eyeCascadeClassifierPath
        self.noseCascadeClassifierPath =noseCascadeClassifierPath
        self.mouthCascadeClassifierPath= mouthCascadeClassifierPath
        self.__detectedFaces=0
        self.detectedEyes=0
        self.detectedMouth=0
        self.detectedSmile=0
        self.facesPath='faces/'
        self.faceDetector=CascadedDetector(self.faceCascadeClassifierPath)
        self.eyeDetector=CascadedDetector(self.eyeCascadeClassifierPath)
        self.faceDetLoad()
        self.faceTracking()
    def getdetectedFaces(self):
        return self.__detectedFaces

    def setLenguage(self,lenguage='Italian'):
        pass
        self.setSpeechLenguage(lenguage)

    def setttsLenguage(self,lenguage='Italian'):
        if self.simulate==False:
            self.tts.setLanguage()
        else:
           self.languageId=3

    def wordDetecded(self,timeOut=5):

        self.memProxy.insertData('WordRecognized','')
        time.sleep(0.5)
        self.speechStart()
        while 1:

            word=self.memProxy.getData('WordRecognized')

            print(word)
            if len(word)>1 and word[0]>=self.speechThr/100:
             self.speechStop()
             return word[0]
             break

    def faceDetLoad(self):
        self.faceCascade = cv2.CascadeClassifier(self.faceCascadeClassifierPath)
        self.eyeCascade = cv2.CascadeClassifier(self.eyeCascadeClassifierPath)
        self.noseCascade = cv2.CascadeClassifier(self.noseCascadeClassifierPath)
        self.mouthCascade = cv2.CascadeClassifier( self.mouthCascadeClassifierPath)


    def faceTracking(self):
        if self.simulate==False:
            if self.__faceProxyLooaded:
                self.faceProxy.enableTracking(True)
                print "Is tracking now enabled on the robot?", self.faceProxy.isTrackingEnabled()

    def faceDetec(self,src):
        faces=self.faceDetector.detect(src)
        imgOut = src.copy()
        self.__detectedFaces=0
        for i,r in enumerate(faces):
           x0,y0,x1,y1 = r
           cv2.rectangle(imgOut, (x0,y0),(x1,y1),(0,255,0),1)
           face =src[y0:y1,x0:x1]
           eyes=self.eyeDetec(face)
           #if eyes.__len__()==2:
           if eyes.__len__()>=0:
                self.__detectedFaces=self.__detectedFaces+1
                self.face=face
           for j,r2 in enumerate(eyes):
              ex0,ey0,ex1,ey1 = r2
              cv2.rectangle(imgOut, (x0+ex0,y0+ey0),(x0+ex1,y0+ey1),(0,255,0),1)
        self.preview(imgOut)
        return faces



    def eyeDetec(self,src):
        return self.eyeDetector.detect(src)

    def preview(self,picture):
        cv2.imshow('preview',picture)
        cv2.waitKey(10)

        pass
    def endPreview(self):
        cv2.destroyAllWindows()

    def newfaceDatabase(self,update=False):
        self.update=update
        if True:
            nome='tmp_name'
            nome=self.getName()
            print('nome',nome)
            print(self.facesPath+nome)

            if not os.path.exists(self.facesPath+nome): os.makedirs(self.facesPath+nome)
            self.say('Sto cercando qualcuno')
            print ( 'sei pronto per farmi scattare qualche foto? \n')
            print ( ' ci vorranno solo 10 secondi\n premi "S" quando sei al centro ')
            counter=0
            while (1):
                picture=self.takePicture()
                faces=self.faceDetec(picture)
                print(self.__detectedFaces)
                if self.__detectedFaces==1:
                   counter=counter+1
                if counter==5:
                   break
                #self.preview(picture)
            self.say('sei pronto per farmi scattare qualche foto')
            time.sleep(0.5)
         #nome = raw_input('Ciao utente '+str(i+1)+' qual e\' il tuo nome?\n nome:')

        else:
             print('From newfaceDatabase: faceRec.load() must be called first..')
        self.say('Per favore, non muoverti')
        self.say('Sto registrando la tua faccia')
        time.sleep(1)

        #comincio a scattare
        start = time.time()
        count = 01
        pathdir='faces/'
        '''
        listing = os.listdir('emotionalFaces')
        for file in listing:
             picture = cv2.imread('emotionalFaces/'+file)
        '''
        #while int(time.time()-start) <= 14:
        taked=0
        while taked <= 10:
             picture=self.takePicture()
             faces=self.faceDetec(picture)
             print(self.__detectedFaces)
             if self.__detectedFaces==1:
                face = cv2.resize(self.face, (273, 273))
                counter=counter+1
                if counter%2 == 0:
                    print  pathdir+nome+str(time.time()-start)+'.jpg'
                    #cv2.imwrite( pathdir+nome+'/'+str(file)+'.jpg', face);
                    cv2.imwrite( pathdir+nome+'/'+nome+str(int(time.time()-start))+'.jpg', face);
                    self.clik()
                    taked=taked+1

        self.faceRecInit()
    def readFaces(self,path, sz=(256,256)):
        """Reads the images in a given folder, resizes images on the fly if size is given.

        Args:
            path: Path to a folder with subfolders representing the subjects (persons).
            sz: A tuple with the size Resizes

        Returns:
            A list [X,y]

                X: The images, which is a Python list of numpy arrays.
                y: The corresponding labels (the unique number of the subject, person) in a Python list.
        """
        c = 0
        X,y = [], []
        folder_names = []

        for dirname, dirnames, filenames in os.walk(path):

            for subdirname in dirnames:

                folder_names.append(subdirname)
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):

                    try:
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)

                        # resize to given size (if given)
                        if (sz is not None):
                            im = cv2.resize(im, sz)
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(c)
                    except IOError, (errno, strerror):
                        print "I/O error({0}): {1}".format(errno, strerror)
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
                        raise
                c = c+1
        return [X,y,folder_names]

    def faceRecInit(self):
        '''
        try:
            [X,y,subject_names] = self.readFaces(self.facesPath)
            list_of_labels = list(xrange(max(y)+1))
            self.subject_dictionary = dict(zip(list_of_labels, subject_names))
            model.compute(X,y)
            print(list_of_labels)
            print('faceRec inited..')
        except:
            e = sys.exc_info()[0]
            print('error',e)
        '''
        self.face=faceRecognizer()
        self.face.Init()


    def faceRec(self):
        '''
        #comincia il riconoscimento.
        person='nesuno'
        picture=self.takePicture()
        faces=self.faceDetec(picture)
        if self.__detectedFaces==1:
            grayFace = cv2.cvtColor(self.face, cv2.COLOR_BGR2GRAY)
            sampleImage = cv2.resize(grayFace, (256,256))
            #capiamo di chi e' sta faccia
            [ predicted_label, generic_classifier_output] = model.predict(sampleImage)
            #scelta la soglia a 700. soglia maggiore di 700, accuratezza minore e v.v.
            if int(generic_classifier_output['distances']) <=  700:
               print(str(self.subject_dictionary[predicted_label]))
               person=str(self.subject_dictionary[predicted_label])
            else:
               print('Nesuno')
        '''
        picture=self.takePicture()
        person=self.face.recFace(picture)
        return person

    def gotoPosture(self,posture):
         if self.simulate==False:
            self.postureProxy.goToPosture(posture, 1.0)







