from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import numpy as np

plt.style.use('dark_background')

def graphPlot():
    def animate(i):
        df = pd.read_csv('EmotionsDetected.csv')
        x = df['Time']
        y = df['Emotions']
        ax.clear()
        ax.plot(x, y)
    ani = FuncAnimation(fig, animate, interval=500)
    canvas.draw()


root = tk.Tk()

frame = tk.Frame(root)
frame.pack()

label = tk.Label(frame)
label.config(text='tkinter matplotlib')
label.pack()


fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, frame)
canvas.get_tk_widget().pack()




# plot()

root.mainloop()