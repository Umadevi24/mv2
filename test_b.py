import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import PIL
import cv2
import numpy as np
from tensorflow.keras import models
import pygame
from gtts import gTTS
import time
from keytotext import pipeline
from mutagen.mp3 import MP3
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from pickle import load
import numpy as np
import warnings
from collections import Counter
window = tk.Tk()

window.title("gesture")

window.geometry("1300x650")
#window.configure(background ="lightgreen")
img=Image.open("new_bg.jpeg")
img=img.resize((1300,650))
bg=ImageTk.PhotoImage(img)

lbl=tk.Label(window,image=bg)
lbl.place(x=0,y=0)

window.resizable(False, False)

#title = tk.Label(text="Click below to choose video file ....", background = "#587d85", fg="white", font=("times", 24,"bold underline italic"))
#title.place(x=10,y=10)

dirPath = "testing_video"

fileList = os.listdir(dirPath)


predicted_words=[]
#Step 1: Read the Text File
classes = []
with open(os.path.join('Labels.txt'), 'r') as fp:
    for line in fp:
        classes.append(line.split()[1])
#sentence generation 
def gen_sentence():
    
    nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")
    generated_sentence = nlp(predicted_words)
    sentence = tk.Label(text='The generated sentence is \n ', background="#587d85",
                               bg="#587d85",fg="white", font=("", 15))
    sentence.place(x=730,y=490)
    sent.config(text=generated_sentence)
    sent.place(x=730,y=530)
    print("Generated Sentence:", generated_sentence)
    predicted_words.clear()
       

def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype(float) / 255.0
    # Apply additional preprocessing steps if needed
    return frame

def openvideo(aa):
    if aa == 1:
        fileName = askopenfilename(
            initialdir='data\\',
            title='Select video for analysis ',
            filetypes=[('video files', '.mp4')],
        )
    if aa == 0:
        fileName = 0

    # Load the model and classes
    model = models.load_model('sign_language_nlp_model_without_attention.h5')
    button2 = tk.Button(text="Predict", command=lambda:preprocess_and_predict_video(fileName, model, classes),bg="#587d85", fg="white", font=("times", 15, "bold italic"), width=20, height=2)
    button2.place(x=310, y=370)
    # Preprocess and predict the video
    #preprocess_and_predict_video(fileName, model, classes)
def preprocess_and_predict_video(video_path, model, classes):
    cap = cv2.VideoCapture(video_path)
    res=[]
    global final_predicted_words
    final_predicted_words=[]
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        frame = cv2.resize(frame, (640, 480))

        # Preprocess the frame
        input_frame = preprocess_frame(frame)

        # Expand dimensions to create a batch of size 1
        input_frame = np.expand_dims(input_frame, axis=0)

        # Make a prediction
        prediction = model.predict(input_frame)
        predicted_label = np.argmax(prediction)
        print('Analysing.....')
        print('Prediction - {} -- {}'.format(predicted_label, classes[predicted_label]))
        out = str(classes[predicted_label]).replace('"','')
        res.append(out)
        # Display the result
        cv2.putText(frame, out, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Sign Language Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
        print("This video most likely belongs to {}".format(out))
        gesture = tk.Label(text='The identified gesture in a video is \n  ', background="#587d85",
                               bg="#587d85",fg="white", font=("", 15))
        gesture.place(x=730,y=150)
        
        gest.place(x=730,y=200)
        final=tk.Label(text='The recent predicted words are \n  ', background="#587d85",
                               bg="#587d85",fg="white", font=("", 15))
        final.place(x=730,y=300)
        
    cap.release()
    cv2.destroyAllWindows()
    word_counts=Counter(res)
    most_common_word,count=word_counts.most_common(1)[0]
    final_word = res[-1]
    predicted_words.append(most_common_word)
    gest.config(text=most_common_word)
    final_words_label.config(text=" ".join(predicted_words)) 

final_words_label = tk.Label(text=" ",bg="#587d85",fg="white",font=("times", 15, "bold italic underline"))
final_words_label.place(x=800, y=360)
######
'''    
def openvideo(aa):
    global final_predicted_words
    final_predicted_words = []
    if aa==1:
        fileName = askopenfilename(initialdir='data\\', title='Select video for analysis ',
                           filetypes=[('video files', '.mp4')])
    if aa==0:
        fileName=0

    def video(file):

        model = models.load_model('sign_270.h5')
        
        cap = cv2.VideoCapture(file)
        res = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = cv2.resize(frame, (640, 480))

            # Preprocess the frame
            input_frame = preprocess_frame(frame)

            # Make a prediction
            prediction = model.predict(input_frame)
            predicted_label = np.argmax(prediction)
            print('analysing.....')
            print('Prediction - {} -- {}'.format(predicted_label, classes[predicted_label]))
            out=str(classes[predicted_label]);
            res.append(out)

            cv2.putText(frame, out, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Sign Language Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print("This video most likely belongs to {}".format(out))

            gesture = tk.Label(text='The identified gesture in a video is \n  ', background="#587d85",
                               bg="#587d85",fg="white", font=("", 15))
            gesture.place(x=730,y=150)
            gest.config(text=out)
            gest.place(x=730,y=200)
            
            final=tk.Label(text='The recent predicted words are \n  ', background="#587d85",
                               bg="#587d85",fg="white", font=("", 15))
            final.place(x=730,y=300)
            
        cap.release()
        cv2.destroyAllWindows()
        final_word = res[-1]
        
        predicted_words.append(final_word)
        final_words_label.config(text=" ".join(predicted_words))
    
'''  
    
 
gen_button = tk.Button(text="Generate Sentence", command=lambda: gen_sentence(),
                                   bg="#587d85", fg="white", font=("times", 15, "bold italic"), width=20, height=2)
gen_button.place(x=390, y=470)
sent = tk.Label(text="", background="white", bg="#587d85",fg="white", font=("times", 18,"bold italic underline"))
    
gest = tk.Label(text="", bg="#587d85",fg="white", font=("times", 18,"bold italic "))
button1 = tk.Button(text="Capture Live", command = lambda:openvideo(0),bg="#587d85",fg="white",font=("times", 15,"bold italic"),width=20,height=2)
button1.place(x=150,y=180)
button1 = tk.Button(text="Choose file", command = lambda:openvideo(1),bg="#587d85",fg="white",font=("times", 15,"bold italic"),width=20,height=2)
button1.place(x=230,y=270)
but_exit = tk.Button(text="Exit", command=lambda:window.destroy(),bg="#587d85",fg="white",font=("times", 15,"bold italic"),width=10,height=1)
but_exit.place(x=1100,y=550)
window.mainloop()

