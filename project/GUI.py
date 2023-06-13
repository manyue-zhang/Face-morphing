from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import *
import PIL
from face_morhper import main

root = Tk()
root.title('Face-Morphing')
root.geometry('1200x500')

decoration = PIL.Image.open('D:/Desktop/DBV/code/utils/background.png').resize((1200, 500))
render = ImageTk.PhotoImage(decoration)
img = Label(image=render)
img.image = render
img.place(x=0, y=0)

global path1_, path2_, rate, seg_img_path


# Original Bild 1
def show_original1_pic():
    global path1_
    path1_ = askopenfilename(title='choose the image')
    print(path1_)
    Img = PIL.Image.open(r'{}'.format(path1_))
    Img = Img.resize((270,270),PIL.Image.ANTIALIAS)   # 256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original1.config(image=img_png_original)
    label_Img_original1.image = img_png_original  # keep a reference
    cv_orinial1.create_image(5, 5,anchor='nw', image=img_png_original)


# Originale Bild 2
def show_original2_pic():
    global path2_
    path2_ = askopenfilename(title='choose the image')
    print(path2_)
    Img = PIL.Image.open(r'{}'.format(path2_))
    Img = Img.resize((270,270),PIL.Image.ANTIALIAS)   # 256x256
    img_png_original = ImageTk.PhotoImage(Img)
    label_Img_original2.config(image=img_png_original)
    label_Img_original2.image = img_png_original  # keep a reference
    cv_orinial2.create_image(5, 5,anchor='nw', image=img_png_original)


# face morphing 
def show_morpher_pic():
    global path1_,seg_img_path,path2_
    print(entry.get())
    mor_img_path = main(path1_,path2_,entry.get())
    Img = PIL.Image.open(r'{}'.format(mor_img_path))
    Img = Img.resize((270, 270), PIL.Image.ANTIALIAS)  # 256x256
    img_png_seg = ImageTk.PhotoImage(Img)
    label_Img_seg.config(image=img_png_seg)
    label_Img_seg.image = img_png_seg  # keep a reference


def quit():
    root.destroy()


# Bild 1
Button(root, text = "open the image 1", command = show_original1_pic).place(x=50,y=120)
# Bild 2
Button(root, text = "open the image 2", command = show_original2_pic).place(x=50,y=200)
# Face Morphing
Button(root, text = "Face Morphing", command = show_morpher_pic).place(x=50,y=280)

Button(root, text = "Exit", command = quit).place(x=900,y=40)

Label(root,text = "alpha",font=10).place(x=50,y=10)
entry = Entry(root)
entry.place(x=130,y=10)


Label(root,text = "Image1",font=10).place(x=280,y=120)
cv_orinial1 = Canvas(root,bg = 'white',width=270,height=270)
cv_orinial1.create_rectangle(8,8,260,260,width=1,outline='red')
cv_orinial1.place(x=180,y=150)
label_Img_original1 = Label(root)
label_Img_original1.place(x=180,y=150)


Label(root,text="Image2",font=10).place(x=600,y=120)
cv_orinial2 = Canvas(root,bg = 'white',width=270,height=270)
cv_orinial2.create_rectangle(8,8,260,260,width=1,outline='red')
cv_orinial2.place(x=500,y=150)
label_Img_original2 = Label(root)
label_Img_original2.place(x=500,y=150)

Label(root, text="Result", font=10).place(x=920,y=120)
cv_seg = Canvas(root, bg='white', width=270,height=270)
cv_seg.create_rectangle(8,8,260,260,width=1,outline='red')
cv_seg.place(x=820,y=150)
label_Img_seg = Label(root)
label_Img_seg.place(x=820,y=150)


root.mainloop()