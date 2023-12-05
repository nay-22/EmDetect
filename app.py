import tkinter as tk
from tkinter import DISABLED, NORMAL, font as tkfont
from tkinter import messagebox,PhotoImage
from TestEmotionDetector import emotion_test
from PIL import ImageTk, Image
from keras.models import model_from_json
import numpy as np
import pandas as pd
import time
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
names = set()


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Face Recognizer")
        self.resizable(False, False)
        self.geometry("780x520")
        # self.resizable(False, False)
        # self.attributes('-fullscreen', 'false')
        # self.attributes('-zoomed', True) 
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
            frame = self.frames[page_name]
            frame.tkraise()

    def on_closing(self):

        if messagebox.askokcancel("Quit", "Are you sure?"):
            self.destroy()


class StartPage(tk.Frame):

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            self.controller = controller
            #load = Image.open("homepagepic.png")
            #load = load.resize((250, 250), Image.ANTIALIAS)
            render = PhotoImage(file='assets/suspect.png')
            img = tk.Label(self, image=render)
            img.image = render
            img.grid(row=0, column=1, rowspan=4, sticky="nsew")
            label = tk.Label(self, text=" Criminal Investigation ", font=("Times", "24", "bold italic"),fg="black")
            label.grid(row=0, sticky="ew")
            button1 = tk.Button(self, text="   New Case  ", fg="#ffffff", bg="IndianRed1",command=lambda: self.controller.show_frame("PageOne"), font=('Comic Sans MS',19))
            button2 = tk.Button(self, text="   Report  ", fg="#ffffff", bg="IndianRed1",command=lambda: self.controller.show_frame("PageTwo"),font=('Comic Sans MS',19))
            button3 = tk.Button(self, text="Quit", fg="black", bg="misty rose", command=self.on_closing,font=('Comic Sans MS',15))
            button1.grid(row=1, column=0, ipady=5, ipadx=10)
            button2.grid(row=2, column=0, ipady=5, ipadx=10)
            button3.grid(row=3, column=0, ipady=10, ipadx=32)
  
        def on_closing(self):
            if messagebox.askokcancel("Quit", "Are you sure?"):
                self.controller.destroy()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text="Case Name", fg="black", font=('Comic Sans MS',20)).grid(row=0, column=0, ipady=10, ipadx=32)
        self.user_name = tk.Entry(self, borderwidth=3, bg="lightyellow", font='Helvetica 11',justify="center")
        self.user_name.grid(row=0, column=1, pady=15, padx=15)

        tk.Label(self, text="Suspect Name", fg="black", font=('Comic Sans MS',20)).grid(row=2, column=0, ipady=10, ipadx=32)
        self.user_name = tk.Entry(self, borderwidth=3, bg="lightyellow", font='Helvetica 11',justify="center")
        self.user_name.grid(row=2, column=1, pady=15, padx=15)

        self.buttoncanc = tk.Button(self, text="Cancel", bg="misty rose", fg="black", command=lambda: controller.show_frame("StartPage"),font=('Comic Sans MS',15))
        self.buttonext = tk.Button(self, text="Detect", fg="#ffffff", bg="IndianRed1", command=lambda: controller.show_frame("PageTwo"),font=('Comic Sans MS',15))
        self.buttoncanc.grid(row=3, column=0, pady=10, ipadx=5, ipady=4)
        self.buttonext.grid(row=3, column=1, pady=10, ipadx=5, ipady=4)

    def start_training(self):
        global names
        if self.user_name.get() == "None":
            messagebox.showerror("Error", "Name cannot be 'None'")
            return
        elif self.user_name.get() in names:
            messagebox.showerror("Error", "User already exists!")
            return
        elif len(self.user_name.get()) == 0:
            messagebox.showerror("Error", "Name cannot be empty!")
            return
        name = self.user_name.get()
        names.add(name)
        self.controller.active_name = name
        self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        # ****************************** PARAMETERS *************************************
        self.fieldnames = ["x_Time", "y_EmoSum"]
        self.x_Time = []
        self.y_Emotion = []
        self.y_EmoSum = 0
        self.json_file = open('model/emotion_model.json', 'r')
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.emotion_model = model_from_json(self.loaded_model_json)
        # load weights into new model
        self.emotion_model.load_weights("model/emotion_model.h5")
        print("Loaded model from disk")
        self.ratioList = []
        self.blinkList = []
        self.idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
        self.blinkCounter = 0
        self.counter = 0
        self.final_count = 0
        self.timed_blink = 0
        self.avg_blinks = 0
        self.color = (255, 0, 255)
        self.lie_Counter = 0
        self.truth_Counter = 0
        self.neutral_Counter = 0

        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        self.emo_freq_dict = {
            "Anger": 0,
            "Disgust": 0,
            "Fear": 0,
            "Happy": 0,
            "Neutral": 0,
            "Sad": 0,
            "Surprise": 0
        } 
        self.time_Count = 0
        self.detector = FaceMeshDetector(maxFaces=2)
        self.cap = cv2.VideoCapture(0)
        # ****************************** PARAMETERS *************************************

        self.videoFrame = tk.Frame(self, width=360, height=430, bg='red')
        self.videoFrame.grid(row=0, column=0, padx=10, pady=10)

        self.video_label = tk.Label(self.videoFrame)
        self.video_label.pack()

        self.graphFrame = tk.Frame(self, width=375, height=430, bg='blue')
        self.graphFrame.grid(row=0, column=1, padx=10, pady=10)

        self.bottom_frame = tk.Frame(self, width=750, height=100)
        self.bottom_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.start_button = tk.Button(self.bottom_frame, text="   Start   ", command=lambda: controller.show_frame("StartPage"),fg="#ffffff", bg="#263942")
        self.stop_button = tk.Button(self.bottom_frame, text="   Stop   ", command=lambda: controller.show_frame("StartPage"),fg="#ffffff", bg="#263942")
        self.end_button = tk.Button(self.bottom_frame, text="   End  ", command=lambda: controller.show_frame("StartPage"),fg="#ffffff", bg="#263942")

        self.start_button.grid(row=0, column=0, ipadx=20, ipady=10)
        self.stop_button.grid(row=0, column=1, ipadx=20, ipady=10)
        self.end_button.grid(row=0, column=2, ipadx=20, ipady=10)

    def start_stream(self):
        global stop_id
        self.start_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        # stop_id = root.after(10, video_stream)

    def stop_stream(self):
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        # root.after_cancel(stop_id)
        



app = MainUI()
app.iconphoto(False, tk.PhotoImage(file='assets/icon.ico'))
app.mainloop()

