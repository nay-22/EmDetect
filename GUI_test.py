from tkinter import messagebox
import ttkbootstrap as ttk
import tkinter as tk

def convert():
    miles = miles_var.get()
    kilometers = miles*1.61
    kilometers_var.set(kilometers)

# Window
# ['cosmo', 'flatly', 'litera', 'minty', 'lumen', 'sandstone', 'yeti', 'pulse', 'united', 'morph', 
# 'journal', 'darkly', 'superhero', 'solar', 'cyborg', 'vapor', 'simplex', 'cerculean']
app = ttk.Window(themename=('darkly'))
app.title('miles - km')
app.minsize(650, 375)

# Window Size
width = 650
height = 375
display_width = app.winfo_screenwidth()
display_height = app.winfo_screenheight()
left = int(display_width/2 - width/2)
top = int(display_height/2 - height/2)
app.geometry(f'{width}x{height}+{left}+{top-40}')

# Label
label = ttk.Label(master=app, text='Miles To Kilometers', font='Arial 24 bold')
label.pack(pady=20)

# Form Container
form = ttk.Frame(master=app)
form.pack()

# Entry Field
miles_var = tk.DoubleVar(value=10)
miles_input = ttk.Entry(master=form, textvariable=miles_var)
miles_input.pack(side='left' ,padx=10, pady=10)

# Button
convert_button = ttk.Button(master=form, text='Convert', command=convert)
convert_button.pack(side='left',padx=10, pady=10)

# Output Label
kilometers_var = tk.DoubleVar()
output_label = ttk.Label(master=app, text='output', font='Arial 22', textvariable=kilometers_var)
output_label.pack(pady=40)

# Execute GUI
app.mainloop()
