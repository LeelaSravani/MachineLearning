
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:21:05 2019

@author: LeelaSravani
"""

from keras.models import load_model

from kivy.app import App

from kivy.lang import Builder

from kivy.properties import ObjectProperty, StringProperty

from kivymd.theming import ThemeManager

from kivymd.navigationdrawer import NavigationDrawer

from kivy.uix.label import Label

from kivy.uix.button import Button

from kivy.uix.floatlayout import FloatLayout

from kivy.uix.screenmanager import ScreenManager, Screen ,NoTransition

from kivy.uix.image import Image as Kimage

from kivy.core.image import Image as Cimage

from kivy.clock import Clock

import easygui

import scipy.misc

import cv2

from imutils import face_utils

import numpy as np

import dlib

import imutils

import statistics

from colormath.color_objects import sRGBColor, LabColor

from colormath.color_conversions import convert_color

from colormath.color_diff import delta_e_cie2000

from io import BytesIO

from PIL import Image

import os



Builder.load_string("""


<uploadScreen>:                                                   # image upload screen for facial Features task

    FloatLayout:                                                  # layut which contains instructions and buttons

        orientation:'vertical'                                    # orientation of layout

        Label:                                                     # initialization of label
            
            text:"Facial Features"                                 # text to be displayed on screen
            
            font_size:24                                           # size of letters
            
            pos_hint:{'center_x': 0.5, 'center_y': .9}             # exact center position of label
            
            color:(1,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            
            text:"1. This feature will detect the facial features of the person in image."            # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.435, 'center_y': .75}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"2. Upload image from your device by clicking on upload image button"                 # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.440, 'center_y': .66}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
       
        Label:                                                     # initialization of label
        
            text:"3. You can capture your photo directly by clicking on 'take picture' button."         # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.465, 'center_y': .57}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"4. Make sure that the image you uploaded / captured is clear without any blur."      # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.49, 'center_y': .48}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"5. Capture your image in the environment with proper brightness."                    # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.4175, 'center_y': .39}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed

            
        Button:                                                   # widget for button

            text:'upload image'                                   # text on button

            pos_hint:{'center_x': 0.2, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face()                                # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'back'                                           # text on button

            pos_hint:{'center_x': 0.8, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.manager.current='naviga'              # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'Take picture'                                   # text on button

            pos_hint:{'center_x': 0.5, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face2()                               # what to happen on clicking the button

<uploadScreen3>:                                                  # image upload screen for  rgb values task


    FloatLayout:                                                  # layout which contains instructions and buttons

        orientation:'vertical'                                    # orientation of layout

        Label:                                                     # initialization of label
            
            text:"Skin Tone"                                       # text to be displayed on screen
            
            font_size:24                                           # size of letters
            
            pos_hint:{'center_x': 0.5, 'center_y': .9}             # exact center position of label
            
            color:(1,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"1. This feature will detect the skin portion in the image."            # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.3625, 'center_y': .75}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"2. Upload image from your device by clicking on upload image button"                 # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.440, 'center_y': .66}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
       
        Label:                                                     # initialization of label
        
            text:"3. You can capture your photo directly by clicking on 'take picture' button."         # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.465, 'center_y': .57}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"4. Make sure that the image you uploaded / captured is clear without any blur."      # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.49, 'center_y': .48}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"5. Capture your image in the environment with proper brightness."                    # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.4175, 'center_y': .39}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
            
        

        Button:                                                   # widget for button

            text:'upload image'                                   # text on button

            pos_hint:{'center_x': 0.2, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face()                                # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'back'                                           # text on button

            pos_hint:{'center_x': 0.8, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.manager.current='naviga'              # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'Take picture'                                   # text on button

            pos_hint:{'center_x': 0.5, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face2()                               # what to happen on clicking the button

<uploadScreen2>:                                                  # image upload screen for  skin task

    FloatLayout:                                                  # layut which contains instructions and buttons

        orientation:'vertical'                                    # orientation of layout

        Label:                                                     # initialization of label
            
            text:"RGB Values"                                 # text to be displayed on screen
            
            font_size:24                                           # size of letters
            
            pos_hint:{'center_x': 0.5, 'center_y': .9}             # exact center position of label
            
            color:(1,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"1. This feature will outputs the average pixel value of each feature."            # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.43, 'center_y': .75}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"2. Upload image from your device by clicking on upload image button"                 # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.440, 'center_y': .66}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
       
        Label:                                                     # initialization of label
        
            text:"3. You can capture your photo directly by clicking on 'take picture' button."         # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.465, 'center_y': .57}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"4. Make sure that the image you uploaded / captured is clear without any blur."      # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.49, 'center_y': .48}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"5. Capture your image in the environment with proper brightness."                    # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.4175, 'center_y': .39}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed

        Label:                                                     # initialization of label
            
            text:"6. Upload images with straight faces for accurate prediction."     # text to be displayed on screen
            
            font_size:20                                                             # size of letters
            
            pos_hint:{'center_x': 0.3965, 'center_y': .31}                           # exact center position of label
            
            color:(0,0,0,1)
                                                  
               
           
        Button:                                                   # widget for button

            text:'upload image'                                   # text on button

            pos_hint:{'center_x': 0.2, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face()                                # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'back'                                           # text on button

            pos_hint:{'center_x': 0.8, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.manager.current='naviga'              # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'Take picture'                                   # text on button

            pos_hint:{'center_x': 0.5, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face2()                               # what to happen on clicking the button

<uploadScreen4>:                                                  # image upload screen for  skin task

    FloatLayout:                                                  # layut which contains instructions and buttons

        orientation:'vertical'                                    # orientation of layout

        Label:                                                     # initialization of label
            
            text:"Asymmetry Predictor"                                 # text to be displayed on screen
            
            font_size:24                                           # size of letters
            
            pos_hint:{'center_x': 0.5, 'center_y': .9}             # exact center position of label
            
            color:(1,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"1. This feature will detects the asymmetry in face of a person."            # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.40, 'center_y': .75}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"2. Upload image from your device by clicking on upload image button"                 # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.440, 'center_y': .66}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
       
        Label:                                                     # initialization of label
        
            text:"3. You can capture your photo directly by clicking on 'take picture' button."         # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.465, 'center_y': .57}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"4. Make sure that the image you uploaded / captured is clear without any blur."      # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.49, 'center_y': .48}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"5. Capture your image in the environment with proper brightness."                    # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.4175, 'center_y': .39}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed

        Label:                                                     # initialization of label
            
            text:"6. Upload images with straight faces for accurate prediction."     # text to be displayed on screen
            
            font_size:20                                                             # size of letters
            
            pos_hint:{'center_x': 0.3965, 'center_y': .31}                           # exact center position of label
            
            color:(0,0,0,1)

        Button:                                                   # widget for button

            text:'upload image'                                   # text on button

            pos_hint:{'center_x': 0.2, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face()                                # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'back'                                           # text on button

            pos_hint:{'center_x': 0.8, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.manager.current='naviga'              # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'Take picture'                                   # text on button

            pos_hint:{'center_x': 0.5, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face2()                               # what to happen on clicking the button

<uploadScreen5>:                                                  # image upload screen for  skin task

    FloatLayout:                                                  # layut which contains instructions and buttons

        orientation:'vertical'                                    # orientation of layout

        Label:                                                     # initialization of label
            
            text:"Lip Makeup"                                 # text to be displayed on screen
            
            font_size:24                                           # size of letters
            
            pos_hint:{'center_x': 0.5, 'center_y': .9}             # exact center position of label
            
            color:(1,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"1. This feature will predicts the Lip Makeup of a person.      "            # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.385, 'center_y': .75}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"2. Upload image from your device by clicking on upload image button"                 # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.440, 'center_y': .66}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
       
        Label:                                                     # initialization of label
        
            text:"3. You can capture your photo directly by clicking on 'take picture' button."         # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.465, 'center_y': .57}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"4. Make sure that the image you uploaded / captured is clear without any blur."      # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.49, 'center_y': .48}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"5. Capture your image in the environment with proper brightness."                    # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.4175, 'center_y': .39}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed

        Label:                                                     # initialization of label
            
            text:"6. Upload images with straight faces for accurate prediction."     # text to be displayed on screen
            
            font_size:20                                                             # size of letters
            
            pos_hint:{'center_x': 0.3965, 'center_y': .31}                           # exact center position of label
            
            color:(0,0,0,1)

        Button:                                                   # widget for button

            text:'upload image'                                   # text on button

            pos_hint:{'center_x': 0.2, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face()                                # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'back'                                           # text on button

            pos_hint:{'center_x': 0.8, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.manager.current='naviga'              # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'Take picture'                                   # text on button

            pos_hint:{'center_x': 0.5, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face2()                               # what to happen on clicking the button

<uploadScreen6>:                                                  # image upload screen for  skin task

    FloatLayout:                                                  # layut which contains instructions and buttons

        orientation:'vertical'                                    # orientation of layout

        Label:                                                     # initialization of label
            
            text:"Eye Makeup"                                 # text to be displayed on screen
            
            font_size:24                                           # size of letters
            
            pos_hint:{'center_x': 0.5, 'center_y': .9}             # exact center position of label
            
            color:(1,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"1. This feature will predicts the Eye Makeup of a person.      "            # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.3857, 'center_y': .75}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"2. Upload image from your device by clicking on upload image button"                 # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.440, 'center_y': .66}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
       
        Label:                                                     # initialization of label
        
            text:"3. You can capture your photo directly by clicking on 'take picture' button."         # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.465, 'center_y': .57}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"4. Make sure that the image you uploaded / captured is clear without any blur."      # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.49, 'center_y': .48}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"5. Capture your image in the environment with proper brightness."                    # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.4175, 'center_y': .39}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed

        Label:                                                     # initialization of label
            
            text:"6. Upload images with straight faces for accurate prediction."     # text to be displayed on screen
            
            font_size:20                                                             # size of letters
            
            pos_hint:{'center_x': 0.3965, 'center_y': .31}                           # exact center position of label
            
            color:(0,0,0,1)

        Button:                                                   # widget for button

            text:'upload image'                                   # text on button

            pos_hint:{'center_x': 0.2, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face()                                # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'back'                                           # text on button

            pos_hint:{'center_x': 0.8, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.manager.current='naviga'              # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'Take picture'                                   # text on button

            pos_hint:{'center_x': 0.5, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face2()                               # what to happen on clicking the button

<uploadScreen7>:                                                  # image upload screen for  skin task

    FloatLayout:                                                  # layut which contains instructions and buttons

        orientation:'vertical'                                    # orientation of layout

        Label:                                                     # initialization of label
            
            text:"Full Makeup"                                 # text to be displayed on screen
            
            font_size:24                                           # size of letters
            
            pos_hint:{'center_x': 0.5, 'center_y': .9}             # exact center position of label
            
            color:(1,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"1. This feature will predicts the Full Makeup of a person.      "            # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.3857, 'center_y': .75}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"2. Upload image from your device by clicking on upload image button"                 # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.440, 'center_y': .66}          # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
       
        Label:                                                     # initialization of label
        
            text:"3. You can capture your photo directly by clicking on 'take picture' button."         # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.465, 'center_y': .57}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"4. Make sure that the image you uploaded / captured is clear without any blur."      # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.49, 'center_y': .48}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed
        
        Label:                                                     # initialization of label
        
            text:"5. Capture your image in the environment with proper brightness."                    # text to be displayed on screen
            
            font_size:20                                           # size of letters
            
            pos_hint:{'center_x': 0.4175, 'center_y': .39}           # exact center position of label
            
            color:(0,0,0,1)                                        # color of the text to be displayed

        Label:                                                     # initialization of label
            
            text:"6. Upload images with straight faces for accurate prediction."     # text to be displayed on screen
            
            font_size:20                                                             # size of letters
            
            pos_hint:{'center_x': 0.3965, 'center_y': .31}                           # exact center position of label
            
            color:(0,0,0,1)


        Button:                                                   # widget for button

            text:'upload image'                                   # text on button

            pos_hint:{'center_x': 0.2, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face()                                # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'back'                                           # text on button

            pos_hint:{'center_x': 0.8, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.manager.current='naviga'              # what to happen on clicking the button

        Button:                                                   # widget for button

            text:'Take picture'                                   # text on button

            pos_hint:{'center_x': 0.5, 'center_y': .1}            # exact positions of X-center and Y-center of the button

            size_hint:0.2,0.1                                     # exact size of the button on layout

            on_release:root.face2()                               # what to happen on clicking the button

""")



main_widget_kv = '''                                              # Navigation Drawer user interface kivy code

#:import Toolbar kivymd.toolbar.Toolbar



BoxLayout:                                                        # layout which contains instructions and buttons

    orientation: 'vertical'                                       # orientation of layout

    Toolbar:                                                      # widget for tool bar

        id: toolbar                                               # identity of toolbar

        title: ' '                                          # title (text) of toolbar

        background_color: app.theme_cls.primary_dark              # background color of toolbar

        left_action_items: [['menu', lambda x: app.nav_drawer.toggle()]]        # widgets on drawer

    FloatLayout:                                                  # initialisation of layout
        orientation: 'vertical'                                   # orientation of layout
       
        Image:                                                    # initialisation of image widget
            source: "beauty.png"                                  # image source 
            size_hint: (0.8,0.6)                                  # size of the image
            pos_hint: {'center_x': 0.5, 'center_y': .6}           # exact center position of image
       
       
       
        Label:                                                     # initialization of label
            text:"Welcome to Smart Mirror!"                        # text to be displayed on screen
            font_size:28                                          # size of letters
            pos_hint:{'center_x': 0.5, 'center_y': .2}             # exact center position of label
            color:(0.9,0.23,0.79,1)
      
    

<Navigator>:                                                      # Navigator layout class user interface code

    NavigationDrawerIconButton:                                   # button for facital features Task

        icon: 'face'                                              # face icon for facial features task

        text: 'Facial Features'                                   # text on button

        on_release:root.face_features()                           # what to happen on clicking the button

    NavigationDrawerIconButton:                                   # button for RGB values Task

        icon: 'face'                                              # icon for RGB values task

        text: 'RGB Values'                                        # text on button

        on_release:root.rgb_values()                              # what to happen on clicking the button

    NavigationDrawerIconButton:                                   # button for skin tone  Task

        icon: 'face'                                              # icon for RGB values task

        text: 'Skin Tone'                                         # text on button

        on_release:root.skin_tone()                               # what to happen on clicking the button

    NavigationDrawerIconButton:                                   # button for Asymmetry  Task

        icon: 'face'                                              # icon for Asymmetry task

        text: 'Asymmetry Predictor'                               # text on button

        on_release:root.asym()                                    # what to happen on clicking the button

    NavigationDrawerIconButton:                                   # button for Lip makeup  Task

        icon: 'face'                                              # icon for Lip makeup task

        text: 'Lip Makeup'                                        # text on button

        on_release:root.lip()                                     # what to happen on clicking the button

    NavigationDrawerIconButton:                                   # button for eye makeup  Task

        icon: 'face'                                              # icon for eye makeup task

        text: 'Eye Makeup'                                        # text on button

        on_release:root.eye()                                     # what to happen on clicking the button

    NavigationDrawerIconButton:                                   # button for full makeup  Task

        icon: 'face'                                              # icon for full makeup task

        text: 'Full Makeup'                                       # text on button

        on_release:root.full()                                    # what to happen on clicking the button

    '''

def calcAvgPixel(im):

    # Extracting all the pixels in the image in the form of list
    
    pixels = list(im.getdata())

    pixel_list = [x for sets in pixels for x in sets]

    # Separating red, blue and green pixels

    pixel_red = pixel_list[::3]

    pixel_green = pixel_list[1::3]

    pixel_blue = pixel_list[2::3]
    
    # Calculating the mean of each channel

    red = int(np.mean(np.array(pixel_red)))

    green = int(np.mean(np.array(pixel_green)))

    blue = int(np.mean(np.array(pixel_blue)))
    
    # Returning the rgb values
    
    return (red, green, blue);


s1=Screen(name='display1')                                        # screen widget initialization for task 1

s2=Screen(name='display2')                                        # screen widget initialization for task 2

s3=Screen(name='display3')                                        # screen widget initialization for task 3

s4=Screen(name='display4')                                        # screen widget initialization for task 4

s5=Screen(name='display5')                                        # screen widget initialization for task 5

s6=Screen(name='display6')                                        # screen widget initialization for task 6

s7=Screen(name='display7')                                      # screen widget initialization for task 7

# Float Layout initialization for different tasks [i.e., floater{i} for ith task]

floater2 = FloatLayout()

floater3 = FloatLayout()

floater5 = FloatLayout()

floater6 = FloatLayout()

floater7 = FloatLayout()

# class declaration for upload screen of task 1

class uploadScreen(Screen):

    # Counter to keep track of no. of images captured

    i_dlib = 0

    # function to be executed after pressing the 'back' button

    def back(self,instance):

        # removing the screen widget after going back to navigation drawer

        sm.remove_widget(s1)

        # moving from task screen to upload screen after pressing 'back' button

        sm.current="upload"

        # Removing the images that are captured and used for this task

        if os.path.exists('image{0}.jpg'.format(self.i_dlib)):

            os.remove('image{0}.jpg'.format(self.i_dlib))

        # Increment the counter

        self.i_dlib += 1

        # returning the screen widget

        return sm

    # function to detect facial features

    def face(self):

        # file chooser function

        image2=easygui.fileopenbox()

        # layout widget for the present task

        floater = FloatLayout()

        # initialize dlib's face detector (HOG-based) and then create

        face_detect = dlib.get_frontal_face_detector()

        # the facial landmark predictor

        predictor = dlib.shape_predictor("F:\shape_predictor_68_face_landmarks.dat")

        # load the input image, resize it, and convert it to grayscale

        image = cv2.imread(image2)

        image = cv2.resize(image,(400,400))

        grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image

        rects = face_detect(grayimage,1)

        # loop over the face detections

        for (i,rect) in enumerate(rects):

            # determine the facial landmarks for the face region, then
        	# convert the facial landmark (x, y)-coordinates to a NumPy
        	# array

            shape = predictor(grayimage,rect)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box

            shape = face_utils.shape_to_np(shape)

            (x,y,w,h) = face_utils.rect_to_bb(rect)

            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)

            # show the face number

            cv2.putText(image, "Face {}".format(i+1), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image

            for (x,y) in shape:

                cv2.circle(image,(x,y),1,(0,255,0),2)

        # Convert bgr image into rgb colorspace

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # convert array to image variable

        rgb = Image.fromarray(image)

        # BytesIO object to store image variable

        data=BytesIO()

        # Save the manipulated image in png format

        rgb.save(data,format='png')

        # seeking the saved image

        data.seek(0)

        # Read back the saved image

        img = Cimage(BytesIO(data.read()), ext='png')

        # initialization of kivy image widget

        beeld = Kimage()

        # assigning the read image to image widget

        beeld.texture=img.texture

        # button for going back

        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding function to 'back' button

        enter.bind(on_release=self.back)

        # adding image widget to layout

        floater.add_widget(beeld)

        # adding button widget to layout

        floater.add_widget(enter)

        # adding layout to screen widget

        s1.add_widget(floater)

        sm.add_widget(s1)

        # Traverse from upload screen to task 1 screen

        sm.current='display1'

    # function to capture image using camera and detect facial features

    def face2(self):

        # Counter variable

        i=0

        # Accessing the web cam of the device

        cam = cv2.VideoCapture(0)

        # Loop to capture a image

        while True:

            # Reading the image from the camera

            ret, frame = cam.read()

            # Font Style

            font = cv2.FONT_HERSHEY_SIMPLEX

            # To Display the timer digit at the centre of the screen

            # Taken some random text to make sure you get the centre position

            text = '####'

            # Calculate the text size

            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary

            x = int((frame.shape[1] - textsize[0]) / 2)

            y = int((frame.shape[0] + textsize[1]) / 2)

            pos = (x, y)

            # Display the timer digits based on no.of times the while loop executes

            if(i>=50 and i<=55):

                cv2.putText(frame, '5', pos, font, 5, (255, 0, 0), 3)

            elif(i>=100 and i<=105):

                cv2.putText(frame, '4', pos, font, 5, (255, 0, 0), 3)

            elif(i>=150 and i<=155):

                cv2.putText(frame, '3', pos, font, 5, (255, 0, 0), 3)

            elif(i>=200 and i<=205):

                cv2.putText(frame, '2', pos, font, 5, (255, 0, 0), 3)

            elif(i>=250 and i<=255):

                cv2.putText(frame, '1', pos, font, 5, (255, 0, 0), 3)

            # Increment the counter

            i=i+1

            # Display the window

            cv2.imshow('frame',frame)

            # Capture the image and save it when counter = 270

            if(i==270):

                cv2.imwrite('image{0}.jpg'.format(self.i_dlib),frame)

                image = frame

            # Display the captured image

            if (i>300):

                cv2.imshow('Captured image',image)

            # The while loop breaks when the counter > 380

            if cv2.waitKey(5) & i>380:

                break

        # Release the web camera

        cam.release()

        # Destroy all the windows

        cv2.destroyAllWindows()



        # Initialise layout widget

        floater = FloatLayout()

        # initialize dlib's face detector (HOG-based) and then create

        face_detect = dlib.get_frontal_face_detector()

        # the facial landmark predictor

        predictor = dlib.shape_predictor("F:\shape_predictor_68_face_landmarks.dat")

        # load the input image, resize it, and convert it to grayscale

        image = cv2.imread('image{0}.jpg'.format(self.i_dlib))

        image = cv2.resize(image,(400,400))

        grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image

        rects = face_detect(grayimage,1)

        # loop over the face detections

        for (i,rect) in enumerate(rects):

            # determine the facial landmarks for the face region, then
        	# convert the facial landmark (x, y)-coordinates to a NumPy
        	# array

            shape = predictor(grayimage,rect)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box

            shape = face_utils.shape_to_np(shape)

            (x,y,w,h) = face_utils.rect_to_bb(rect)

            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)

            # show the face number

            cv2.putText(image, "Face {}".format(i+1), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image

            for (x,y) in shape:

                cv2.circle(image,(x,y),1,(0,255,0),2)

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # convert from array to image variable

        rgb = Image.fromarray(image)

         # initialization of BytesIO variable

        data=BytesIO()

        # saving the image using BytesIO variable

        rgb.save(data,format='png')

        # seeking the saved image

        data.seek(0)

        # Read the saved image

        img = Cimage(BytesIO(data.read()), ext='png')

        # initialization of image widget

        beeld = Kimage()

        # assigning the manipulated image to image widget

        beeld.texture=img.texture

        # initialization of 'back' button

        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding function to 'back' button

        enter.bind(on_release=self.back)

        # adding image widget to layout

        floater.add_widget(beeld)

        # adding back button to Layout

        floater.add_widget(enter)

        # adding layout to screen widget

        s1.add_widget(floater)

        # Adding screen widget to the screen

        sm.add_widget(s1)

        # Traversing from upload screen to task screen

        sm.current='display1'


# class declation to calculate avg pixels

class uploadScreen2(Screen):

    # Counter to keep track of captured images

    i_rgb = 100000

    btn=ObjectProperty(None)

    # function to execute after pressing back button

    def back(self,instance):

        # clear all widgets from layout after moving back

        floater2.clear_widgets()

        # clear all layouts from screen after moving back

        s2.clear_widgets()

        # Removing the images captured and used in this task

        if os.path.exists('image{0}.jpg'.format(self.i_rgb)):

            os.remove('image{0}.jpg'.format(self.i_rgb))

        if os.path.exists('face.jpg'):

            os.remove('face.jpg')

            os.remove('eyes.jpg')

        if os.path.exists('nose.jpg'):

            os.remove('nose.jpg')

        if os.path.exists('mouth.jpg'):

            os.remove('mouth.jpg')

        # Increment the counter

        self.i_rgb += 1

        # moving from upload screen to task screeen

        sm.current="upload2"

        # returning screen manager

        return sm

    # function for RGB values task
    def face(self):

        # Initialise the haarcascades frontal face classifier

        face_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        # Initialise the haarcascades eye classifier

        eyes_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_eye.xml")

        # Initialise the nose classifier

        nose_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\nose.xml")

        # Initialise the mouth classifier

        mouth_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\mouth.xml")

        # file choosing box

        image2=easygui.fileopenbox()

        # Read the image using opencv

        image = cv2.imread(image2)

        # Convert image to gray scale

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image

        face = face_detect.detectMultiScale(gray, 1.2, 5)

        d={}

        # If face is detected

        if len(face)!=0:

            # Make a copy of original image

            image1 = image.copy()

            # Loop over the face

            for (x, y, w, h) in face:

                # Draw rectangle around face

                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Crop the image upto face

                roi_gray = gray[y:y + h, x:x + w]

                roi_color = image[y:y+h,x:x+w]

                roi_color_f = image1[y:y + h, x:x + w]

                # Save the face image

                cv2.imwrite("face.jpg", roi_color_f)

                # Read back the face image

                im_f = Image.open(r"face.jpg")

                # Calculate the average pixel inthe face area

                d['face']=calcAvgPixel(im_f)

                # Detect eyes in the image

                eyes = eyes_detect.detectMultiScale(roi_gray, 1.5,5)

                # Counter for no.of eyes detected

                i=1

                # If eyes detected

                if len(eyes)!=0:

                    # Loop over each eye detection

                    for (ex, ey, ew, eh) in eyes:

                        # Draw rectangle around eyes

                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                        # Crop eyes portion in  the face image

                        roi_color_e = roi_color_f[ey:ey+eh,ex:ex+ew]

                        # Save the eyes image

                        cv2.imwrite("eyes.jpg", roi_color_e)

                        # Read back the eyes image

                        im_e = Image.open(r"eyes.jpg")

                        # Check the counter and calculate avg pixle for each eye

                        if(i==1):

                            d['eye 1']=calcAvgPixel(im_e)

                        else:

                            d['eye 2']=calcAvgPixel(im_e)

                        # Increment the counter after each detection

                        i=i+1

                        # Break the loop after both the eyes are detected

                        if i>2:

                            break

                # Detect nose in the image

                nose = nose_detect.detectMultiScale(roi_gray,1.1)

                # Counter

                j=1

                # If nose detected

                if len(nose)!=0:

                    # Loop over the nose

                    for (nx, ny, nw, nh) in nose:

                        # Draw rectangle around nose

                        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 255, 0), 2)

                        # Crop the nose area from the face

                        roi_color_n = roi_color_f[ny:ny + nh, nx:nx + nw]

                        # Save the nose image

                        cv2.imwrite("nose.jpg", roi_color_n)

                        # Read back the nose image

                        im_n = Image.open(r"nose.jpg")

                        # Calculate the avg pixel in the nose area

                        d['nose']=calcAvgPixel(im_n)

                        # Increment the counter

                        j=j+1

                        # Break the loop after nose detected

                        if j>1:

                            break

                # Detect mouth in the image

                mouth = mouth_detect.detectMultiScale(roi_gray, 1.2,45)

                # Counter

                k=1

                # If mouth detected

                if len(mouth)!=0:

                    # Loop over the mouth

                    for (mx, my, mw, mh) in mouth:

                        # Draw rectangle around mouth

                        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)

                        # Crop the mouth portion from the face

                        roi_color_m = roi_color_f[my:my + mh, mx:mx + mw]

                        # Save the mouth image

                        cv2.imwrite("mouth.jpg", roi_color_m)

                        # Read back the mouth image

                        im_m = Image.open(r"mouth.jpg")

                        # Calculate the avg pixel in the mouth area

                        d['mouth']=calcAvgPixel(im_m)

                        # Increment the counter

                        k=k+1

                        # Break the loop after mouth is detected

                        if k>1:

                            break

        # initialization of image widget

        beeld2=Kimage(source=image2,size_hint=(0.35,0.5), pos_hint={'center_x': 0.25, 'center_y': .6})

        # adding image to layout

        floater2.add_widget(beeld2)

        # initialization of label

        lab= Label (text="Average pixels in (R,G,B)" ,font_size=20,pos_hint={'center_x': 0.75, 'center_y': .85},color=(0,1,0,1))

        # initialization of back button

        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .1})

        # binding function to back button

        enter.bind(on_release=self.back)

        l=len(d)

        for i in range(l):

            # initialization of Label

            labe=Label(text="{0}={1}".format(list(d.keys())[i],d[list(d.keys())[i]]),font_size=20,pos_hint={'center_x': 0.75, 'center_y': 0.85-(i+1)*0.1},color=(0,0,0,1))

            # adding label to layout

            floater2.add_widget(labe)

        d={}

        # adding label to layout

        floater2.add_widget(lab)

        # adding back button to layout

        floater2.add_widget(enter)

        # adding  layout to screen

        s2.add_widget(floater2)

        # Traversing from upload screeen to task screen

        sm.current="display2"

    # function to capture image using camera and calc avg pixel

    def face2(self):

        # Counter variable

        i=0

        # Accessing the web cam of the device

        cam = cv2.VideoCapture(0)

        # Loop to capture a image

        while True:

            # Reading the image from the camera

            ret, frame = cam.read()

            # Font Style

            font = cv2.FONT_HERSHEY_SIMPLEX

            # To Display the timer digit at the centre of the screen

            # Taken some random text to make sure you get the centre position

            text = '####'

            # Calculate the text size

            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary

            x = int((frame.shape[1] - textsize[0]) / 2)

            y = int((frame.shape[0] + textsize[1]) / 2)

            pos = (x, y)

            # Display the timer digits based on no.of times the while loop executes

            if(i>=50 and i<=55):

                cv2.putText(frame, '5', pos, font, 5, (255, 0, 0), 3)

            elif(i>=100 and i<=105):

                cv2.putText(frame, '4', pos, font, 5, (255, 0, 0), 3)

            elif(i>=150 and i<=155):

                cv2.putText(frame, '3', pos, font, 5, (255, 0, 0), 3)

            elif(i>=200 and i<=205):

                cv2.putText(frame, '2', pos, font, 5, (255, 0, 0), 3)

            elif(i>=250 and i<=255):

                cv2.putText(frame, '1', pos, font, 5, (255, 0, 0), 3)

            # Increment the counter

            i=i+1

            # Display the window

            cv2.imshow('frame',frame)

            # Capture the image and save it when counter = 270

            if(i==270):

                cv2.imwrite('image{0}.jpg'.format(self.i_rgb),frame)

                image = frame

            # Display the captured image

            if (i>300):

                cv2.imshow('Captured image',image)

            # The while loop breaks when the counter > 380

            if cv2.waitKey(5) & i>380:

                break

        # Release the web camera

        cam.release()

        # Destroy all the windows

        cv2.destroyAllWindows()


        # Initialise the haarcascades frontal face classifier

        face_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        # Initialise the haarcascades eye classifier

        eyes_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_eye.xml")

        # Initialise the nose classifier

        nose_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\nose.xml")

        # Initialise the mouth classifier

        mouth_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\mouth.xml")


        # Read the image using opencv

        image = cv2.imread('image{0}.jpg'.format(self.i_rgb))

        # Convert image to gray scale

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image

        face = face_detect.detectMultiScale(gray, 1.2, 5)

        d={}

        # If face is detected

        if len(face)!=0:

            # Make a copy of original image

            image1 = image.copy()

            # Loop over the face

            for (x, y, w, h) in face:

                # Draw rectangle around face

                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Crop the image upto face

                roi_gray = gray[y:y + h, x:x + w]

                roi_color = image[y:y+h,x:x+w]

                roi_color_f = image1[y:y + h, x:x + w]

                # Save the face image

                cv2.imwrite("face.jpg", roi_color_f)

                # Read back the face image

                im_f = Image.open(r"face.jpg")

                # Calculate the average pixel inthe face area

                d['face']=calcAvgPixel(im_f)

                # Detect eyes in the image

                eyes = eyes_detect.detectMultiScale(roi_gray, 1.5,5)

                # Counter for no.of eyes detected

                i=1

                # If eyes detected

                if len(eyes)!=0:

                    # Loop over each eye detection

                    for (ex, ey, ew, eh) in eyes:

                        # Draw rectangle around eyes

                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                        # Crop eyes portion in  the face image

                        roi_color_e = roi_color_f[ey:ey+eh,ex:ex+ew]

                        # Save the eyes image

                        cv2.imwrite("eyes.jpg", roi_color_e)

                        # Read back the eyes image

                        im_e = Image.open(r"eyes.jpg")

                        # Check the counter and calculate avg pixle for each eye

                        if(i==1):

                            d['eye 1']=calcAvgPixel(im_e)

                        else:

                            d['eye 2']=calcAvgPixel(im_e)

                        # Increment the counter after each detection

                        i=i+1

                        # Break the loop after both the eyes are detected

                        if i>2:

                            break

                # Detect nose in the image

                nose = nose_detect.detectMultiScale(roi_gray,1.1)

                # Counter

                j=1

                # If nose detected

                if len(nose)!=0:

                    # Loop over the nose

                    for (nx, ny, nw, nh) in nose:

                        # Draw rectangle around nose

                        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 255, 0), 2)

                        # Crop the nose area from the face

                        roi_color_n = roi_color_f[ny:ny + nh, nx:nx + nw]

                        # Save the nose image

                        cv2.imwrite("nose.jpg", roi_color_n)

                        # Read back the nose image

                        im_n = Image.open(r"nose.jpg")

                        # Calculate the avg pixel in the nose area

                        d['nose']=calcAvgPixel(im_n)

                        # Increment the counter

                        j=j+1

                        # Break the loop after nose detected

                        if j>1:

                            break

                # Detect mouth in the image

                mouth = mouth_detect.detectMultiScale(roi_gray, 1.2,45)

                # Counter

                k=1

                # If mouth detected

                if len(mouth)!=0:

                    # Loop over the mouth

                    for (mx, my, mw, mh) in mouth:

                        # Draw rectangle around mouth

                        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)

                        # Crop the mouth portion from the face

                        roi_color_m = roi_color_f[my:my + mh, mx:mx + mw]

                        # Save the mouth image

                        cv2.imwrite("mouth.jpg", roi_color_m)

                        # Read back the mouth image

                        im_m = Image.open(r"mouth.jpg")

                        # Calculate the avg pixel in the mouth area

                        d['mouth']=calcAvgPixel(im_m)

                        # Increment the counter

                        k=k+1

                        # Break the loop after mouth is detected

                        if k>1:

                            break

            # Convert bgr image to rgb color space and resize it

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image,(400,500))

            # initialization of image widget

            beeld2=Kimage(source='image{0}.jpg'.format(self.i_rgb),size_hint=(0.35,0.5), pos_hint={'center_x': 0.25, 'center_y': .6})

            # adding image widget to layout

            floater2.add_widget(beeld2)

            # initialization of Label

            lab= Label (text="Average pixels in (R,G,B)" ,font_size=20,pos_hint={'center_x': 0.75, 'center_y': .85},color=(0,1,0,1))

            # initialization of back button

            enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .1})

            # binding the function to back button

            enter.bind(on_release=self.back)

            l=len(d)

            for i in range(l):

                # initialization of label

                labe=Label(text="{0}={1}".format(list(d.keys())[i],d[list(d.keys())[i]]),font_size=20,pos_hint={'center_x': 0.75, 'center_y': 0.85-(i+1)*0.1},color=(0,0,0,1))         # initialization of label

                # adding label to layout

                floater2.add_widget(labe)

            d={}

            # adding label to layout

            floater2.add_widget(lab)

            # adding back button to layout

            floater2.add_widget(enter)

            # adding layout to screen

            s2.add_widget(floater2)

            # Traversing from upload screen to task screen

            sm.current="display2"

        else:

            # Convert bgr image to rgb and resize it

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image,(400,500))

            # initialization of image widget

            beeld2=Kimage(source='image{0}.jpg'.format(self.i_rgb),size_hint=(0.35,0.5), pos_hint={'center_x': 0.25, 'center_y': .6})

            # adding image to layout

            floater2.add_widget(beeld2)

            # initialization of back button

            enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .1})

            # binding function to back button

            enter.bind(on_release=self.back)

            # initialization of label

            lab1= Label (text="Failed to detect the face, try again!" ,font_size=22,pos_hint={'center_x': 0.75, 'center_y': .65},color=(1,0,0,1))

            # adding label to layout

            floater2.add_widget(lab1)

            # adding back button to layout

            floater2.add_widget(enter)

            # adding layout to screen

            s2.add_widget(floater2)

            # Traversing from upload screen to task screen

            sm.current="display2"

# upload screen function for detecting skin tone in an image

class uploadScreen3(Screen):

    # Counter to keep track of captured images

    i_skin = 200000

    btn=ObjectProperty(None)

    # function to execute after pressing bac button

    def back(self,instance):

        # removing the screen widget after pressing the back button

        sm.remove_widget(s3)

        # Removing the captured and stored images used in this task

        if os.path.exists('image{0}.jpg'.format(self.i_skin)):

            os.remove('image{0}.jpg'.format(self.i_skin))

        # Increment the counter

        self.i_skin += 1

        # Traversing from task screen to upload screeen

        sm.current="upload3"

        # returning the screen

        return sm

    # function to be executed after uploading the image

    def face(self):

        # file choosing box

        image2=easygui.fileopenbox()

        # layout initialization

        floater = FloatLayout()

        # Read the image and resize it

        img = cv2.imread(image2)

        img = cv2.resize(img,(400,400))

        # Convert the image from bgr to hsv colorspace

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Set the range of skin color pixel values in rgb

        lower = np.array([0, 48, 80], dtype=np.uint8)

        upper = np.array([20, 255, 255], dtype=np.uint8)

        # Apply the Gaussian blur and get mask of pixels

        img_hsv = cv2.GaussianBlur(img_hsv, (3,3), 0)

        mask = cv2.inRange(img_hsv, lower, upper)

        # convert single channel mask back into 3 channels

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # perform bitwise and on mask to obtain cut-out image that is not skin

        masked_img = cv2.bitwise_and(img, mask_rgb)

        # Convert bgr image to rgb

        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)


        # converting array to image variable

        rgb = Image.fromarray(masked_img)

        # initialization of BytesIO variable

        data=BytesIO()

        # saving image in a png format

        rgb.save(data,format='png')

        # seeking saved image

        data.seek(0)

        # reading the saved image

        imgr = Cimage(BytesIO(data.read()), ext='png')

        # initialization of image widget

        beeld = Kimage(pos_hint={'center_x': 0.5, 'center_y': .6})

         # assigning the manipulated image to image widget

        beeld.texture=imgr.texture

        # initialization of back button

        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding funciton to back button

        enter.bind(on_release=self.back)

        # adding image to layout

        floater.add_widget(beeld)

        # adding back button to layout

        floater.add_widget(enter)

        # adding layout to screen

        s3.add_widget(floater)

        # adding screen to screen manager

        sm.add_widget(s3)

        # Traversing from upload screen to task screen

        sm.current='display3'

    # function to be executed for processing the image through camera

    def face2(self):

        # Counter variable

        i=0

        # Accessing the web cam of the device

        cam = cv2.VideoCapture(0)

        # Loop to capture a image

        while True:

            # Reading the image from the camera

            ret, frame = cam.read()

            # Font Style

            font = cv2.FONT_HERSHEY_SIMPLEX

            # To Display the timer digit at the centre of the screen

            # Taken some random text to make sure you get the centre position

            text = '####'

            # Calculate the text size

            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary

            x = int((frame.shape[1] - textsize[0]) / 2)

            y = int((frame.shape[0] + textsize[1]) / 2)

            pos = (x, y)

            # Display the timer digits based on no.of times the while loop executes

            if(i>=50 and i<=55):

                cv2.putText(frame, '5', pos, font, 5, (255, 0, 0), 3)

            elif(i>=100 and i<=105):

                cv2.putText(frame, '4', pos, font, 5, (255, 0, 0), 3)

            elif(i>=150 and i<=155):

                cv2.putText(frame, '3', pos, font, 5, (255, 0, 0), 3)

            elif(i>=200 and i<=205):

                cv2.putText(frame, '2', pos, font, 5, (255, 0, 0), 3)

            elif(i>=250 and i<=255):

                cv2.putText(frame, '1', pos, font, 5, (255, 0, 0), 3)

            # Increment the counter

            i=i+1

            # Display the window

            cv2.imshow('frame',frame)

            # Capture the image and save it when counter = 270

            if(i==270):

                cv2.imwrite('image{0}.jpg'.format(self.i_skin),frame)

                image = frame

            # Display the captured image

            if (i>300):

                cv2.imshow('Captured image',image)

            # The while loop breaks when the counter > 380

            if cv2.waitKey(5) & i>380:

                break

        # Release the web camera

        cam.release()

        # Destroy all the windows

        cv2.destroyAllWindows()

        # initialization of layout object

        floater = FloatLayout()

        # Read the image and resize it

        img = cv2.imread('image{0}.jpg'.format(self.i_skin))

        img = cv2.resize(img,(400,400))

        # Convert the image from bgr to hsv colorspace

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Set the range of skin color pixel values in rgb

        lower = np.array([0, 48, 80], dtype=np.uint8)

        upper = np.array([20, 255, 255], dtype=np.uint8)

        # Apply the Gaussian blur and get mask of pixels

        img_hsv = cv2.GaussianBlur(img_hsv, (3,3), 0)

        mask = cv2.inRange(img_hsv, lower, upper)

        # convert single channel mask back into 3 channels

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # perform bitwise and on mask to obtain cut-out image that is not skin

        masked_img = cv2.bitwise_and(img, mask_rgb)

        # Convert bgr image to rgb

        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)


        # conversion from array to image variable

        rgb = Image.fromarray(masked_img)

        # initialization of BytesIO object

        data=BytesIO()

        # saving image in png format

        rgb.save(data,format='png')

        # seeking the image saved

        data.seek(0)

        # reading the image saved

        imgr = Cimage(BytesIO(data.read()), ext='png')

        # initialization of image widget

        beeld = Kimage(pos_hint={'center_x': 0.5, 'center_y': .6})

        # assigning the manipulated image to image widget

        beeld.texture=imgr.texture

        # initialization of back button

        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # initialization of back button

        enter.bind(on_release=self.back)

        # adding image to layout

        floater.add_widget(beeld)

        # adding back button to layout

        floater.add_widget(enter)

        # adding layout to screen

        s3.add_widget(floater)

        # Adding screen widget to screen

        sm.add_widget(s3)

        # Traversing from upload screen to task screen

        sm.current='display3'


# class declaration for Asymmetry prediction

class uploadScreen4(Screen):

    # Counter to keep track of no.of captured images

    i_asym = 300000

    btn=ObjectProperty(None)

    # funciton to be executed ater pressing the back button

    def back(self,instance):

        #sm.remove_widget(s4)

        # clearing all widgets from layout after pressing back button

        floater3.clear_widgets()

        # clear the layouts from the screen

        s4.clear_widgets()

        # Removing the captured and stored images that are used in this task

        if os.path.exists('image{0}.jpg'.format(self.i_asym)):

            os.remove('image{0}.jpg'.format(self.i_asym))

        os.remove('left face.jpg')

        os.remove('right face.jpg')

        # Increment the counter

        self.i_asym += 1

        # Traversing from task screen to upload screen

        sm.current="upload4"

        # returning the screen manager

        return sm

    # function to be executed after uploading the image

    def face(self):

        #floater = FloatLayout()

        # file choosing box

        image_path=easygui.fileopenbox()

        # Read the image and resize it

        img = cv2.imread(image_path)

        img = cv2.resize(img,(400,400))

        # Convert the image from bgr to hsv colorspace

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Set the range of skin color pixel values in rgb

        lower = np.array([0, 48, 80], dtype=np.uint8)

        upper = np.array([20, 255, 255], dtype=np.uint8)

        # Apply the Gaussian blur and get mask of pixels

        img_hsv = cv2.GaussianBlur(img_hsv, (3,3), 0)

        mask = cv2.inRange(img_hsv, lower, upper)

        # convert single channel mask back into 3 channels

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # perform bitwise and on mask to obtain cut-out image that is not skin

        masked_img = cv2.bitwise_and(img, mask_rgb)

        # Initialise the haarcascades frontal face classifier

        face_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        # Convert color image to gray scale

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image

        face = face_detect.detectMultiScale(gray, 1.2, 5)

        # Initialise left and right face vars

        left_face=0

        right_half=0

        # Create blank images for left and right half faces

        left_half=np.zeros(img.shape)

        right_half=np.zeros(img.shape)

        # Loop over the face

        for (x,y,w,h) in face:

            #masked_img_face = masked_img[y:y+h, x:x+w]

            # Calculate the coordinates for left half face

            x_left_start = x

            y_left_start = y

            x_left_end = int(x+w/2)

            y_left_end = int(y+h)

            # Calculate the coordinates for right half face

            x_right_start = x_left_end

            y_right_start = y

            x_right_end = x+w

            y_right_end = y+h

            # Crop the left and right half faces from the masked image

            left_half = masked_img[y_left_start:y_left_end, x_left_start:x_left_end]

            right_half = masked_img[y_right_start:y_right_end, x_right_start:x_right_end]


        # Save the masked left and right half faces

        cv2.imwrite("left face.jpg",left_half)

        cv2.imwrite("right face.jpg",right_half)

        # Read back the left and right face images

        left_face = Image.open("left face.jpg")

        right_face = Image.open("right face.jpg")

        # Calculate the avg pixel of each half face

        r1,g1,b1= calcAvgPixel(left_face)

        r2,g2,b2 = calcAvgPixel(right_face)

        color1_rgb = sRGBColor(r1, g1, b1);

        color2_rgb = sRGBColor(r2, g2, b2);


        # Convert from RGB to Lab Color Space

        color1_lab = convert_color(color1_rgb, LabColor);

        color2_lab = convert_color(color2_rgb, LabColor);


        # Find the color difference

        delta_e = delta_e_cie2000(color1_lab, color2_lab);

        # Convert the bgr image to rgb

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # conversion from the array to image variable

        rgb = Image.fromarray(img)

        # initialization of BytesIO object

        data=BytesIO()

        # saving the manipulated image in png format

        rgb.save(data,format='png')

        # seeking the image saved

        data.seek(0)

        # reading the saved image

        imgr = Cimage(BytesIO(data.read()), ext='png')

        # initialization of image widget

        beeld = Kimage(pos_hint={'center_x': 0.5, 'center_y': .6})

        # assigning the maniputed image to image widget

        beeld.texture=imgr.texture

        # Impose a threshold either from user or hard-coded

        #threshold = int(input("Enter a threshold: "))

        threshold = 17

        # Check color difference and the threshold

        if delta_e > threshold:

            # initialization of label widget

            lab1=Label (text="Asymmetry exists" ,font_size=28,pos_hint={'center_x': 0.5, 'center_y': .2},color=(0,0,0,1))

            # adding the label to layout

            floater3.add_widget(lab1)

        else:

            # initialization of label widget

            lab1=Label (text="Asymmetry does't exist" ,font_size=28,pos_hint={'center_x': 0.5, 'center_y': .2},color=(0,0,0,1))

            # adding the label to layout

            floater3.add_widget(lab1)

        # initialization of back button

        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding the fucntion to back button

        enter.bind(on_release=self.back)

        # adding the image widget to layout

        floater3.add_widget(beeld)

        # adding the back button to layout

        floater3.add_widget(enter)

        # adding layout to screen

        s4.add_widget(floater3)

        # adding the layout to screen

        sm.current='display4'

    # function to be executed to for processing the image from camera

    def face2(self):

        # Counter variable

        i=0

        # Accessing the web cam of the device

        cam = cv2.VideoCapture(0)

        # Loop to capture a image

        while True:

            # Reading the image from the camera

            ret, frame = cam.read()

            # Font Style

            font = cv2.FONT_HERSHEY_SIMPLEX

            # To Display the timer digit at the centre of the screen

            # Taken some random text to make sure you get the centre position

            text = '####'

            # Calculate the text size

            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary

            x = int((frame.shape[1] - textsize[0]) / 2)

            y = int((frame.shape[0] + textsize[1]) / 2)

            pos = (x, y)

            # Display the timer digits based on no.of times the while loop executes

            if(i>=50 and i<=55):

                cv2.putText(frame, '5', pos, font, 5, (255, 0, 0), 3)

            elif(i>=100 and i<=105):

                cv2.putText(frame, '4', pos, font, 5, (255, 0, 0), 3)

            elif(i>=150 and i<=155):

                cv2.putText(frame, '3', pos, font, 5, (255, 0, 0), 3)

            elif(i>=200 and i<=205):

                cv2.putText(frame, '2', pos, font, 5, (255, 0, 0), 3)

            elif(i>=250 and i<=255):

                cv2.putText(frame, '1', pos, font, 5, (255, 0, 0), 3)

            # Increment the counter

            i=i+1

            # Display the window

            cv2.imshow('frame',frame)

            # Capture the image and save it when counter = 270

            if(i==270):

                cv2.imwrite('image{0}.jpg'.format(self.i_asym),frame)

                image = frame

            # Display the captured image

            if (i>300):

                cv2.imshow('Captured image',image)

            # The while loop breaks when the counter > 380

            if cv2.waitKey(5) & i>380:

                break

        # Release the web camera

        cam.release()

        # Destroy all the windows

        cv2.destroyAllWindows()


        # Read the image and resize it

        img = cv2.imread('image{0}.jpg'.format(self.i_asym))

        img = cv2.resize(img,(400,400))

        # Convert the image from bgr to hsv colorspace

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Set the range of skin color pixel values in rgb

        lower = np.array([0, 48, 80], dtype=np.uint8)

        upper = np.array([20, 255, 255], dtype=np.uint8)

        # Apply the Gaussian blur and get mask of pixels

        img_hsv = cv2.GaussianBlur(img_hsv, (3,3), 0)

        mask = cv2.inRange(img_hsv, lower, upper)

        # convert single channel mask back into 3 channels

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # perform bitwise and on mask to obtain cut-out image that is not skin

        masked_img = cv2.bitwise_and(img, mask_rgb)

        # Initialise the haarcascades frontal face classifier

        face_detect = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        # Convert color image to gray scale

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image

        face = face_detect.detectMultiScale(gray, 1.2, 5)

        # Initialise left and right face vars

        left_face=0

        right_half=0

        # Create blank images for left and right half faces

        left_half=np.zeros(img.shape)

        right_half=np.zeros(img.shape)

        # Loop over the face

        for (x,y,w,h) in face:

            #masked_img_face = masked_img[y:y+h, x:x+w]

            # Calculate the coordinates for left half face

            x_left_start = x

            y_left_start = y

            x_left_end = int(x+w/2)

            y_left_end = int(y+h)

            # Calculate the coordinates for right half face

            x_right_start = x_left_end

            y_right_start = y

            x_right_end = x+w

            y_right_end = y+h

            # Crop the left and right half faces from the masked image

            left_half = masked_img[y_left_start:y_left_end, x_left_start:x_left_end]

            right_half = masked_img[y_right_start:y_right_end, x_right_start:x_right_end]


        # Save the masked left and right half faces

        cv2.imwrite("left face.jpg",left_half)

        cv2.imwrite("right face.jpg",right_half)

        # Read back the left and right face images

        left_face = Image.open("left face.jpg")

        right_face = Image.open("right face.jpg")

        # Calculate the avg pixel of each half face

        r1,g1,b1= calcAvgPixel(left_face)

        r2,g2,b2 = calcAvgPixel(right_face)

        color1_rgb = sRGBColor(r1, g1, b1);

        color2_rgb = sRGBColor(r2, g2, b2);


        # Convert from RGB to Lab Color Space

        color1_lab = convert_color(color1_rgb, LabColor);

        color2_lab = convert_color(color2_rgb, LabColor);


        # Find the color difference

        delta_e = delta_e_cie2000(color1_lab, color2_lab);

        threshold = 17

        # Check color difference and threshold

        if delta_e > threshold:

            # initialization of label

            lab1=Label (text="Asymmetry exists" ,font_size=28,pos_hint={'center_x': 0.5, 'center_y': .2},color=(0,0,0,1))

            # adding the label to the layout

            floater3.add_widget(lab1)

        else:

            # initialization of label

            lab1=Label (text="Asymmetry does't exist" ,font_size=28,pos_hint={'center_x': 0.5, 'center_y': .2},color=(0,0,0,1))

            # adding the label to layout

            floater3.add_widget(lab1)

        # initialization of image widget

        beeld = Kimage(source='image{0}.jpg'.format(self.i_asym),size_hint=(0.6,0.6), pos_hint={'center_x': 0.5, 'center_y': .6})

        # initialization of back button

        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding the fucntion to back button

        enter.bind(on_release=self.back)

        # adding the image widget to layout

        floater3.add_widget(beeld)

        # adding the back button to layout

        floater3.add_widget(enter)

        # adding the layout to screen

        s4.add_widget(floater3)

        # moving from upload screen to task screen

        sm.current='display4'


# class declaration for the lip makeup task
        
class uploadScreen5(Screen):

    # Counter to keep track of captured images
    
    i_lip = 400000

    # function to be executed after pressing the back button
    
    def back(self,instance):

        # remove all wigets from layout when moving back
        
        floater5.clear_widgets()

        # remove all layouts from screen when moving back
        
        s5.clear_widgets()

        # Removing captured and stored images used in this task
        
        if os.path.exists('image{0}.jpg'.format(self.i_lip)):

            os.remove('image{0}.jpg'.format(self.i_lip))
        
        # Increment the counter
        
        self.i_lip += 1

        # move from task screen to upload screen
        
        sm.current="upload5"

        # return screen manager
        
        return sm

    # function to be executed after uploading image
    
    def face(self):

        # file chooser function
        
        file = easygui.fileopenbox()
        
        # Load the saved deep learning model for lip makeup detection
        
        model = load_model('Lip Makeup.h5')

        # Read the image
        
        test_image = cv2.imread(file)

        # Haarcascade Classifier for frontal face detection
        
        face_cascade = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
        
        # Convert the color image to grayscale
        
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        
        face = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        # Loop over each face
        
        for (x, y, w, h) in face:
            
            # Crop the face in the image
            
            test_image = test_image[y:y + h, x:x + w]
            
        # If face is present 
        
        if (len(test_image)!=0):
            
            # resize the image
            
            test_image = cv2.resize(test_image, (130, 150))
            
            # Convert the image to array format
            
            test_image = np.array(test_image)
            
            # Convert the datatype o float
            
            test_image = test_image.astype('float32')
            
            # Normalize the image
            
            test_image /= 255
            
            # Adding an extra dimension as tensorflow needs it
            
            test_image = np.expand_dims(test_image, axis=0)
            
            # Predicting the class probabilities of the image
            
            probs = model.predict(test_image)
            
            # Predicting the final class of the image
            
            #clas = model.predict_classes(test_image)
            
            # Converting the probalities into percentages
            
            for lis in probs:
                #per = lis[clas[0]]*100
                lips = round(lis[0]*100,3)
                nolip = round(lis[1]*100,3)

            #percent = round(per,3)

            #pred = names[model.predict_classes(test_image)[0]]

            # initialization of image widget
            
            image = Kimage(source = file, size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.6})

            # initialization of label
            
            lab1 = Label(text = 'Prediction: \nLip Makeup {0}% \nNo Lip makeup {1}%'.format(lips,nolip), font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            if lips < 25:

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif lips < 50:

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif lips < 75:

                # initialization of label
                
                lab3 = Label(text = 'Looking good!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            else:

                # initialization of label
                
                lab3 = Label(text = 'Almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            # adding image widget to layout
            
            floater5.add_widget(image)

            # adding label to layout
            
            floater5.add_widget(lab1)

            # adding label to layout
            
            floater5.add_widget(lab3)

        # When there is no face in the image
        
        else:

            # initialization of image widget
            
            image = Kimage(source = file, size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.6})

            # initialization of label
            
            lab2 = Label(text = 'Failed to detect the face, please upload proper image', font_size = 22, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            # adding image widget ot layout
            
            floater5.add_widget(image)

            # adding label to layout
            
            floater5.add_widget(lab2)

        # initialize of back button
        
        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding the function with back button
        
        enter.bind(on_release=self.back)

        # adding back butto to layout
        
        floater5.add_widget(enter)

        # adding layout screen
        
        s5.add_widget(floater5)

        # moving from upload screen to task screen
        
        sm.current = 'display5'

    # function to be executed after capturing image from camera
    
    def face2(self):

        # Counter variable
        
        i=0
        
        # Accessing the web cam of the device
        
        cam = cv2.VideoCapture(0)
        
        # Loop to capture a image
        
        while True:
            
            # Reading the image from the camera
            
            ret, frame = cam.read()
            
            # Font Style
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # To Display the timer digit at the centre of the screen
            
            # Taken some random text to make sure you get the centre position

            text = '####'
            
            # Calculate the text size
            
            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary

            x = int((frame.shape[1] - textsize[0]) / 2)

            y = int((frame.shape[0] + textsize[1]) / 2)

            pos = (x, y)

            # Display the timer digits based on no.of times the while loop executes
            
            if(i>=50 and i<=55):

                cv2.putText(frame, '5', pos, font, 5, (255, 0, 0), 3)

            elif(i>=100 and i<=105):

                cv2.putText(frame, '4', pos, font, 5, (255, 0, 0), 3)

            elif(i>=150 and i<=155):

                cv2.putText(frame, '3', pos, font, 5, (255, 0, 0), 3)

            elif(i>=200 and i<=205):

                cv2.putText(frame, '2', pos, font, 5, (255, 0, 0), 3)

            elif(i>=250 and i<=255):

                cv2.putText(frame, '1', pos, font, 5, (255, 0, 0), 3)

            # Increment the counter
            
            i=i+1
            
            # Display the window
            
            cv2.imshow('frame',frame)
            
            # Capture the image and save it when counter = 270
            
            if(i==270):

                cv2.imwrite('image{0}.jpg'.format(self.i_lip),frame)

                image = frame
            
            # Display the captured image
            
            if (i>300):

                cv2.imshow('Captured image',image)

            # The while loop breaks when the counter > 380
            
            if cv2.waitKey(5) & i>380:

                break
            
        # Release the web camera
        
        cam.release()
        
        # Destroy all the windows
        
        cv2.destroyAllWindows()
        
        
        # Load the saved deep learning model for lip makeup detection
        
        model = load_model('Lip Makeup.h5')
        
        # Read the captured image
        
        test_image = cv2.imread('image{0}.jpg'.format(self.i_lip))

        # Haarcascade Classifier for frontal face detection
        
        face_cascade = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
        
        # Convert the color image to grayscale
        
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        
        face = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        # Loop over each face
        
        for (x, y, w, h) in face:
            
            # Crop the face in the image
            
            test_image = test_image[y:y + h, x:x + w]
            
        # If face is present 
        
        if (len(test_image)!=0):
            
            # resize the image
            
            test_image = cv2.resize(test_image, (130, 150))
            
            # Convert the image to array format
            
            test_image = np.array(test_image)
            
            # Convert the datatype o float
            
            test_image = test_image.astype('float32')
            
            # Normalize the image
            
            test_image /= 255
            
            # Adding an extra dimension as tensorflow needs it
            
            test_image = np.expand_dims(test_image, axis=0)
            
            # Predicting the class probabilities of the image
            
            probs = model.predict(test_image)
            
            # Predicting the final class of the image
            
            #clas = model.predict_classes(test_image)
            
            # Converting the probalities into percentages
            
            for lis in probs:
                #per = lis[clas[0]]*100
                lips = round(lis[0]*100,3)
                nolip = round(lis[1]*100,3)

            #percent = round(per,3)

            #pred = names[model.predict_classes(test_image)[0]]

            # initialization of image widget
            
            image = Kimage(source = 'image{0}.jpg'.format(self.i_lip), size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.6})

            # initialization of label
            
            lab1 = Label(text = 'Prediction: \nLip Makeup {0}% \nNo Lip makeup {1}%'.format(lips,nolip), font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            if lips < 25:

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif lips < 50:

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif lips < 75:

                # initialization of label
                
                lab3 = Label(text = 'Looking good!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            else:

                # initialization of label
                
                lab3 = Label(text = 'Almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            # adding image widget to layout
            
            floater5.add_widget(image)

            # adding label to layout
            
            floater5.add_widget(lab1)

            # adding label to layout
            
            floater5.add_widget(lab3)

        # When there is no face in the image
        
        else:

            # initialization of image widget
            
            image = Kimage(source = 'image{0}.jpg'.format(self.i_lip), size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.6})

            # initialization of label
            
            lab2 = Label(text = 'Failed to detect the face, please capture the image properly!', font_size = 22, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            # adding image widget ot layout
            
            floater5.add_widget(image)

            # adding label to layout
            
            floater5.add_widget(lab2)

        # initialize of back button
        
        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding the function with back button
        
        enter.bind(on_release=self.back)

        # adding back butto to layout
        
        floater5.add_widget(enter)

        # adding layout screen
        
        s5.add_widget(floater5)

        # moving from upload screen to task screen
        
        sm.current = 'display5'


# class declaration for eye makeup detection

class uploadScreen6(Screen):

    # Counter to keep track of captured images
    
    i_eye = 500000

    # function to be executed after pressing the back button
    
    def back(self,instance):

        #clear all the widgets from layout on moving back
        
        floater6.clear_widgets()

        # clear all layouts from screen on moving back
        
        s6.clear_widgets()
        
        # Removing captured and stored images used in this task
        
        if os.path.exists('image{0}.jpg'.format(self.i_eye)):

            os.remove('image{0}.jpg'.format(self.i_eye))

        # Increment the counter
        
        self.i_eye += 1

        # moving from task screen to upload screen
        
        sm.current="upload6"

        # return the screen manager
        
        return sm

    # function to be executed after uploading an image
    
    def face(self):

        # file choosing function
        
        fileeye = easygui.fileopenbox()
        
        # Load the saved deep learning model for Eye makeup detection
        
        model = load_model('Eye Makeup.h5')

        # Read the image
        
        test_image = cv2.imread(fileeye)
        
        # Haarcascade Classifier to detect the frontal face
        
        face_cascade = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        # Convert the color image to grayscale
        
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        
        face = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        # Loop over each face
        
        for (x, y, w, h) in face:
            
            # Crop the face in the image
            
            test_image = test_image[y:y + h, x:x + w]
            
        # If face is present 
        
        if (len(test_image)!=0):
            
            # resize the image
            
            test_image = cv2.resize(test_image, (130, 150))
            
            # Convert the image to array format
            
            test_image = np.array(test_image)
            
            # Convert the datatype o float
            
            test_image = test_image.astype('float32')
            
            # Normalize the image
            
            test_image /= 255
            
            # Adding an extra dimension as tensorflow needs it
            
            test_image = np.expand_dims(test_image, axis=0)
            
            # Predicting the class probabilities of the image
            
            probs = model.predict(test_image)
            
            # Predicting the final class of the image
            
            #clas = model.predict_classes(test_image)
            
            # Converting the probalities into percentages
            
            for lis in probs:
                #per = lis[clas[0]]*100
                eyes = round(lis[0]*100,3)
                noeye = round(lis[1]*100,3)

            #percent = round(per,3)

            #pred = names[model.predict_classes(test_image)[0]]

            # initialization of image widget
            
            image = Kimage(source = fileeye, size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.6})

            # initialization of label to display text
            
            lab1 = Label(text = 'Prediction: \nEye Makeup {0}% \nNo Eye makeup {1}%'.format(eyes,noeye), font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))
            
            # Comment based on percentage of makeup
            
            if eyes < 25:

                # initialization of label to display text
                
                lab3 = Label(text = 'Looks like there\'s no eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif eyes < 50:

                # initialization of label to display text
                
                lab3 = Label(text = 'You can even improve your eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif eyes < 75:

                # initialization of label to display text
                
                lab3 = Label(text = 'Looking good!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            else:

                # initialization of label to display text
                
                lab3 = Label(text = 'Almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            # add image widget to layout
            
            floater6.add_widget(image)

            # add label to layout
            
            floater6.add_widget(lab1)

            # add label to layout
            
            floater6.add_widget(lab3)

        else:

            # initialization of image widget
            
            image = Kimage(source = fileeye, size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.6})

            # initialization of label to display text
            
            lab2 = Label(text = 'Failed to detect the face, please upload proper image', font_size = 22, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            # adding image widget to layout
            
            floater6.add_widget(image)

            # adding label to layout
            
            floater6.add_widget(lab2)

        # initialization of back button
        
        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding the fucntion to back button
        
        enter.bind(on_release=self.back)


        # adding button to layout
        
        floater6.add_widget(enter)

        # adding layout to screen
        
        s6.add_widget(floater6)

        # moving from upload screen to task  screen
        
        sm.current = 'display6'

    # function to be executed after pressing the open camera button
    
    def face2(self):

        # Counter variable
        
        i=0
        
        # Accessing the web cam of the device
        
        cam = cv2.VideoCapture(0)
        
        # Loop to capture a image
        
        while True:
            
            # Reading the image from the camera
            
            ret, frame = cam.read()
            
            # Font Style
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # To Display the timer digit at the centre of the screen
            
            # Taken some random text to make sure you get the centre position

            text = '####'
            
            # Calculate the text size
            
            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary

            x = int((frame.shape[1] - textsize[0]) / 2)

            y = int((frame.shape[0] + textsize[1]) / 2)

            pos = (x, y)

            # Display the timer digits based on no.of times the while loop executes
            
            if(i>=50 and i<=55):

                cv2.putText(frame, '5', pos, font, 5, (255, 0, 0), 3)

            elif(i>=100 and i<=105):

                cv2.putText(frame, '4', pos, font, 5, (255, 0, 0), 3)

            elif(i>=150 and i<=155):

                cv2.putText(frame, '3', pos, font, 5, (255, 0, 0), 3)

            elif(i>=200 and i<=205):

                cv2.putText(frame, '2', pos, font, 5, (255, 0, 0), 3)

            elif(i>=250 and i<=255):

                cv2.putText(frame, '1', pos, font, 5, (255, 0, 0), 3)

            # Increment the counter
            
            i=i+1
            
            # Display the window
            
            cv2.imshow('frame',frame)
            
            # Capture the image and save it when counter = 270
            
            if(i==270):

                cv2.imwrite('image{0}.jpg'.format(self.i_eye),frame)

                image = frame
            
            # Display the captured image
            
            if (i>300):

                cv2.imshow('Captured image',image)

            # The while loop breaks when the counter > 380
            
            if cv2.waitKey(5) & i>380:

                break
            
        # Release the web camera
        
        cam.release()
        
        # Destroy all the windows
        
        cv2.destroyAllWindows()

        # Load the saved deep learning model for Eye makeup detection
        
        model = load_model('Eye Makeup.h5')

        # Read the captured image
        
        test_image = cv2.imread('image{0}.jpg'.format(self.i_eye))

        # Haarcascade Classifier to detect the frontal face
        
        face_cascade = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        # Convert the color image to grayscale
        
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        
        face = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        # Loop over each face
        
        for (x, y, w, h) in face:
            
            # Crop the face in the image
            
            test_image = test_image[y:y + h, x:x + w]
            
        # If face is present 
        
        if (len(test_image)!=0):
            
            # resize the image
            
            test_image = cv2.resize(test_image, (130, 150))
            
            # Convert the image to array format
            
            test_image = np.array(test_image)
            
            # Convert the datatype o float
            
            test_image = test_image.astype('float32')
            
            # Normalize the image
            
            test_image /= 255
            
            # Adding an extra dimension as tensorflow needs it
            
            test_image = np.expand_dims(test_image, axis=0)
            
            # Predicting the class probabilities of the image
            
            probs = model.predict(test_image)
            
            # Predicting the final class of the image
            
            #clas = model.predict_classes(test_image)
            
            # Converting the probalities into percentages
            
            for lis in probs:
                #per = lis[clas[0]]*100
                eyes = round(lis[0]*100,3)
                noeye = round(lis[1]*100,3)

            #percent = round(per,3)

            #pred = names[model.predict_classes(test_image)[0]]

            # initialization of image widget
            
            image = Kimage(source = 'image{0}.jpg'.format(self.i_eye), size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.6})

            # initialization of label to display text
            
            lab1 = Label(text = 'Prediction: \nEye Makeup {0}% \nNo Eye makeup {1}%'.format(eyes,noeye), font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))
            
            # Comment based on percentage of makeup
            
            if eyes < 25:

                # initialization of label to display text
                
                lab3 = Label(text = 'Looks like there\'s no eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif eyes < 50:

                # initialization of label to display text
                
                lab3 = Label(text = 'You can even improve your eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif eyes < 75:

                # initialization of label to display text
                
                lab3 = Label(text = 'Looking good!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            else:

                # initialization of label to display text
                
                lab3 = Label(text = 'Almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            # add image widget to layout
            
            floater6.add_widget(image)

            # add label to layout
            
            floater6.add_widget(lab1)

            # add label to layout
            
            floater6.add_widget(lab3)

        else:

            # initialization of image widget
            
            image = Kimage(source = 'image{0}.jpg'.format(self.i_eye), size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.6})

            # initialization of label to display text
            
            lab2 = Label(text = 'Failed to detect the face, please capture the image properly!', font_size = 22, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            # adding image widget to layout
            
            floater6.add_widget(image)

            # adding label to layout
            
            floater6.add_widget(lab2)

        # initialization of back button
        
        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding the fucntion to back button
        
        enter.bind(on_release=self.back)


        # adding button to layout
        
        floater6.add_widget(enter)

        # adding layout to screen
        
        s6.add_widget(floater6)

        # moving from upload screen to task  screen
        
        sm.current = 'display6'

# declaration of class for the full makeup task
        
class uploadScreen7(Screen):

    i_full = 600000

    # initialization of function to be executed aftet pressing the back button
    
    def back(self,instance):

        # removing all widgets from layout while moving back
        
        floater7.clear_widgets()

        # removing all layouts from screen while moving back
        
        s7.clear_widgets()
        
        # Removing captured and stored images used in this task
        
        if os.path.exists('image{0}.jpg'.format(self.i_full)):

            os.remove('image{0}.jpg'.format(self.i_full))

        # Increment the counter
        
        self.i_full += 1

        # moving from task screen to upload screen
        
        sm.current="upload7"

        # returning the screen manager
        
        return sm

    # function to be executed after uploading the image
    
    def face(self):

        # file chooser function
        
        filefull = easygui.fileopenbox()
        
        # Load the deep learning models for total makeup detection
        
        model_lip = load_model('Lip Makeup.h5')

        model_eye = load_model('Eye Makeup.h5')

        # Read the image
        
        test_image = cv2.imread(filefull)

        # Haarcascade Classifier for frontal face detection
        
        face_cascade = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
        
        # Convert the color image to grayscale
        
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the iamge
        
        face = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        # Loop over each face
        
        for (x, y, w, h) in face:
            
            # Crop the face from the image
            
            test_image = test_image[y:y + h, x:x + w]

        # If face is present
        
        if (len(test_image)!=0):
            
            # Resize the image
            
            test_image = cv2.resize(test_image, (130, 150))
            
            # Convert the image to array format
            
            test_image = np.array(test_image)
            
            # Convert the datatype to float
            
            test_image = test_image.astype('float32')
            
            # Normalize the image
            
            test_image /= 255
            
            # Adding an extra dimension as tensorflow needs it
            
            test_image = np.expand_dims(test_image, axis=0)

            # Calculating the probability of lip makeup and convert to percentage
            
            prob1 = model_lip.predict(test_image)

            prob1 = np.array(prob1[0]*100)

            prob_1 = round(prob1[0],3)

            # Calculating the probability of eye makeup and convert to percentage

            prob2 = model_eye.predict(test_image)

            prob2 = np.array(prob2[0]*100)

            prob_2 = round(prob2[0],3)

            # Calculating the probability of full makeup and convert to percentage
            
            fullm = (prob_1+prob_2)/2

            fullmu = round(fullm,3)


            # initialization of image widget
            
            image = Kimage(source = filefull, size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.7})

            # initialization of label
            
            lab1 = Label(text = 'Prediction: Full makeup: {0}%'.format(fullmu), font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            # Comment based on different combinations of lip and eye makeup
            
            if (prob_1 <= 25 and prob_2 <= 25):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 25 and prob_2 <= 50):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup \n You can even improve your eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 25 and prob_2 <= 75):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup \n Your eye makeup is good enough!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 25 and prob_2 <= 100):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup \n Your eye makeup is almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 50 and prob_2 <= 25):

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip makeup \n Looks like there\'s no eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 50 and prob_2 <= 50):

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip and eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 50 and prob_2 <= 75):

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip makeup \n Your eye makeup is good enough!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 50 and prob_2 <= 100):

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip makeup \n  Your eye makeup is almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 75 and prob_2 <= 25):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is good enough! \nLooks like there\'s no eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 75 and prob_2 <= 50):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup \n You can even improve your eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 75 and prob_2 <= 75):

                # initialization of label
                
                lab3 = Label(text = 'Your lip and eye makeup are good enough!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 75 and prob_2 <= 100):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is good enough! \n Your eye makeup is almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 100 and prob_2 <= 25):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is almost perfect! \nLooks like there\'s no eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 100 and prob_2 <= 50):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is almost perfect! \n You can even improve your eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 100 and prob_2 <= 75):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is almost perfect! \nYour eye makeup is good enough!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 100 and prob_2 <= 100):

                # initialization of label
                
                lab3 = Label(text = 'Your lip and eye makeup are almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            else:

                # initialization of label
                
                lab3 = Label(text = ' ', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))


            # adding image widget to layout
            
            floater7.add_widget(image)

            # adding label to layout
            
            floater7.add_widget(lab1)

            # adding label to layout
            
            floater7.add_widget(lab3)

        else:

            # initialization of image widget
            
            image = Kimage(source = filefull, size_hint=(0.6, 0.6), pos_hint={'center_x': 0.5, 'center_y': 0.7})

            # initialization of label
            
            lab2 = Label(text = 'Failed to detect the face, please upload proper image', font_size = 22, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            # adding image widget to layout
            
            floater7.add_widget(image)

            # adding label to layout
            
            floater7.add_widget(lab2)

        # initialization of back button
        
        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding the function to back button
        
        enter.bind(on_release=self.back)



        # addding the back button to layout
        
        floater7.add_widget(enter)

        # adding the layout screen

        s7.add_widget(floater7)

        # moving from upload screen to task  screen

        sm.current = 'display7'

    # function to be executed after processing the image caputed by camera
    
    def face2(self):

        # Counter variable
        
        i=0
        
        # Accessing the web cam of the device
        
        cam = cv2.VideoCapture(0)
        
        # Loop to capture a image
        
        while True:
            
            # Reading the image from the camera
            
            ret, frame = cam.read()
            
            # Font Style
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # To Display the timer digit at the centre of the screen
            
            # Taken some random text to make sure you get the centre position

            text = '####'
            
            # Calculate the text size
            
            textsize = cv2.getTextSize(text, font, 1, 2)[0]

            # get coords based on boundary

            x = int((frame.shape[1] - textsize[0]) / 2)

            y = int((frame.shape[0] + textsize[1]) / 2)

            pos = (x, y)

            # Display the timer digits based on no.of times the while loop executes
            
            if(i>=50 and i<=55):

                cv2.putText(frame, '5', pos, font, 5, (255, 0, 0), 3)

            elif(i>=100 and i<=105):

                cv2.putText(frame, '4', pos, font, 5, (255, 0, 0), 3)

            elif(i>=150 and i<=155):

                cv2.putText(frame, '3', pos, font, 5, (255, 0, 0), 3)

            elif(i>=200 and i<=205):

                cv2.putText(frame, '2', pos, font, 5, (255, 0, 0), 3)

            elif(i>=250 and i<=255):

                cv2.putText(frame, '1', pos, font, 5, (255, 0, 0), 3)

            # Increment the counter
            
            i=i+1
            
            # Display the window
            
            cv2.imshow('frame',frame)
            
            # Capture the image and save it when counter = 270
            
            if(i==270):

                cv2.imwrite('image{0}.jpg'.format(self.i_full),frame)

                image = frame
            
            # Display the captured image
            
            if (i>300):

                cv2.imshow('Captured image',image)

            # The while loop breaks when the counter > 380
            
            if cv2.waitKey(5) & i>380:

                break
            
        # Release the web camera
        
        cam.release()
        
        # Destroy all the windows
        
        cv2.destroyAllWindows()
        
        # Load the saved deep learning models for total makeup detection

        model_lip = load_model('Lip Makeup.h5')

        model_eye = load_model('Eye Makeup.h5')
        
        # Read the captured image

        test_image = cv2.imread('image{0}.jpg'.format(self.i_full))

        # Haarcascade Classifier for frontal face detection
        
        face_cascade = cv2.CascadeClassifier(r"F:\Anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml")
        
        # Convert the color image to grayscale
        
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the iamge
        
        face = face_cascade.detectMultiScale(gray, 1.2, 5)
        
        # Loop over each face
        
        for (x, y, w, h) in face:
            
            # Crop the face from the image
            
            test_image = test_image[y:y + h, x:x + w]

        # If face is present
        
        if (len(test_image)!=0):
            
            # Resize the image
            
            test_image = cv2.resize(test_image, (130, 150))
            
            # Convert the image to array format
            
            test_image = np.array(test_image)
            
            # Convert the datatype to float
            
            test_image = test_image.astype('float32')
            
            # Normalize the image
            
            test_image /= 255
            
            # Adding an extra dimension as tensorflow needs it
            
            test_image = np.expand_dims(test_image, axis=0)

            # Calculating the probability of lip makeup and convert to percentage
            
            prob1 = model_lip.predict(test_image)

            prob1 = np.array(prob1[0]*100)

            prob_1 = round(prob1[0],3)

            # Calculating the probability of eye makeup and convert to percentage

            prob2 = model_eye.predict(test_image)

            prob2 = np.array(prob2[0]*100)

            prob_2 = round(prob2[0],3)

            # Calculating the probability of full makeup and convert to percentage
            
            fullm = (prob_1+prob_2)/2

            fullmu = round(fullm,3)


            # initialization of image widget
            
            image = Kimage(source = 'image{0}.jpg'.format(self.i_full), size_hint=(0.5, 0.5), pos_hint={'center_x': 0.5, 'center_y': 0.7})

            # initialization of label
            
            lab1 = Label(text = 'Prediction: Full makeup: {0}%'.format(fullmu), font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            # Comment based on different combinations of lip and eye makeup
            
            if (prob_1 <= 25 and prob_2 <= 25):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 25 and prob_2 <= 50):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup \n You can even improve your eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 25 and prob_2 <= 75):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup \n Your eye makeup is good enough!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 25 and prob_2 <= 100):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup \n Your eye makeup is almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 50 and prob_2 <= 25):

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip makeup \n Looks like there\'s no eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 50 and prob_2 <= 50):

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip and eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 50 and prob_2 <= 75):

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip makeup \n Your eye makeup is good enough!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 50 and prob_2 <= 100):

                # initialization of label
                
                lab3 = Label(text = 'You can even improve your lip makeup \n  Your eye makeup is almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 75 and prob_2 <= 25):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is good enough! \nLooks like there\'s no eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 75 and prob_2 <= 50):

                # initialization of label
                
                lab3 = Label(text = 'Looks like there\'s no lip makeup \n You can even improve your eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 75 and prob_2 <= 75):

                # initialization of label
                
                lab3 = Label(text = 'Your lip and eye makeup are good enough!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 75 and prob_2 <= 100):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is good enough! \n Your eye makeup is almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 100 and prob_2 <= 25):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is almost perfect! \nLooks like there\'s no eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 100 and prob_2 <= 50):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is almost perfect! \n You can even improve your eye makeup', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 100 and prob_2 <= 75):

                # initialization of label
                
                lab3 = Label(text = 'Your lip makeup is almost perfect! \nYour eye makeup is good enough!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            elif (prob_1 <= 100 and prob_2 <= 100):

                # initialization of label
                
                lab3 = Label(text = 'Your lip and eye makeup are almost perfect!', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))

            else:

                # initialization of label
                
                lab3 = Label(text = ' ', font_size = 20, pos_hint = {'center_x': 0.5, 'center_y': 0.3},color=(0,0,0,1))


            # adding image widget to layout
            
            floater7.add_widget(image)

            # adding label to layout
            
            floater7.add_widget(lab1)

            # adding label to layout
            
            floater7.add_widget(lab3)

        else:

            # initialization of image widget
            
            image = Kimage(source = 'image{0}.jpg'.format(self.i_full), size_hint=(0.6, 0.6), pos_hint={'center_x': 0.5, 'center_y': 0.7})

            # initialization of label
            
            lab2 = Label(text = 'Failed to detect the face, please capture the image properly!', font_size = 22, pos_hint = {'center_x': 0.5, 'center_y': 0.2},color=(0,0,0,1))

            # adding image widget to layout
            
            floater7.add_widget(image)

            # adding label to layout
            
            floater7.add_widget(lab2)

        # initialization of back button
        
        enter = Button(text='back', size_hint=(0.2,0.1), pos_hint={'center_x': 0.5, 'center_y': .05})

        # binding the function to back button
        
        enter.bind(on_release=self.back)



        # addding the back button to layout
        
        floater7.add_widget(enter)

        # adding the layout screen

        s7.add_widget(floater7)

        # moving from upload screen to task  screen

        sm.current = 'display7'

# initialization the screen manager object with no transition effect

sm = ScreenManager(transition=NoTransition())

# initialization of Splash Screen

splashScr = Screen(name='SplashScreen')

# adding the image to splash screen

splashScr.add_widget(Kimage(source='000.PNG'))

# adding splash screen to screen manager

sm.add_widget(splashScr)

# adding the upload screen of task 1 to screen manager

sm.add_widget(uploadScreen(name="upload"))

# adding the upload screen of task 3 to screen manager

sm.add_widget(uploadScreen3(name="upload3"))

# adding the upload screen of task 2 to screen manager

sm.add_widget(uploadScreen2(name="upload2"))

# adding the upload screen of task 4 to screen manager

sm.add_widget(uploadScreen4(name="upload4"))

# adding the upload screen of task 5 to screen manager

sm.add_widget(uploadScreen5(name="upload5"))

# adding the upload screen of task 6 to screen manager

sm.add_widget(uploadScreen6(name="upload6"))

# adding the upload screen of task 7 to screen manager

sm.add_widget(uploadScreen7(name="upload7"))


#sm.add_widget(s1)

# adding screen of task 2 to screen manager

sm.add_widget(s2)

#sm.add_widget(s3)

# adding screen of task 4 to screen manager

sm.add_widget(s4)

# adding screen of task 5 to screen manager

sm.add_widget(s5)

# adding screen of task 6 to screen manager

sm.add_widget(s6)

# adding screen of task 7 to screen manager

sm.add_widget(s7)

# function which changes screen from splash screen to navigation drawer

def function(instance):

    # moving from splash screen to navigation dreawe
    
    sm.current="naviga"

# class declaration of navigation drawer

class Navigator(NavigationDrawer):

    #image_source = StringProperty('images/me.jpg')

    title = StringProperty('Tasks')

    # function which moves to upload screen after pressing facial features button in navigation drawer
    
    def face_features(self):
        
        # moves from navigation drawer to upload screen
        
        sm.current="upload"
        
        # returning the screen manager
        
        return sm

    # function which moves to upload screen after pressing rgv values  button in navigation drawer
    
    def rgb_values(self):
        
        # moves from navigation drawer to upload screen
        
        sm.current="upload2"
        
        # returning the screen manager
        
        return sm
    
    # function which moves to upload screen after pressing skin tone  button in navigation drawer
    
    def skin_tone(self):
        
        # moves from navigation drawer to upload screen
        
        sm.current="upload3"
        
        # returning the screen manager
        
        return sm

    

    # function which moves to upload screen after pressing Asymmetry prediction button in navigation drawer
    
    def asym(self):
        
        # moves from navigation drawer to upload screen
        
        sm.current="upload4"
        
        # returning the screen manager
        
        return sm

    # function which moves to upload screen after pressing lip makeup  button in navigation drawer
    
    def lip(self):
        
        # moves from navigation drawer to upload screen
        
        sm.current = "upload5"
        
        # returning the screen manager
        
        return sm

    # function which moves to upload screen after pressing eye makeup button in navigation drawer
    
    def eye(self):
        
        # moves from navigation drawer to upload screen
        
        sm.current = "upload6"
        
        # returning the screen manager
        
        return sm

    # function which moves to upload screen after pressing full makeup button in navigation drawer
    
    def full(self):
    
        # moves from navigation drawer to upload screen
        
        sm.current = "upload7"
        
        # returning the screen manager
        
        return sm


# declaration of main class
class SmartMirrorApp(App):

    # declaration of theme manager
    
    theme_cls = ThemeManager()

    # declaration of navigation drawer properties
    
    nav_drawer = ObjectProperty()


    # function which loads main
    
    def build(self):

        # loading the kivy user layouts
        
        main_widget = Builder.load_string(main_widget_kv)

        # initialization of navigation drawer screen
        
        nav=Screen(name="naviga")

        # add the layout to navigation drawer
        
        nav.add_widget(main_widget)

        # add navigation drawer to screen manager
        
        sm.add_widget(nav)

        # display the splash screen image for 5 seconds
        
        Clock.schedule_once(function,10)

        # connect the navigation drawer
        
        self.nav_drawer = Navigator()
        
        # return the screen manager
        
        return sm

# run the app 
SmartMirrorApp().run()
