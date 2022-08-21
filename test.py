from tkinter import *
from PIL import Image,ImageTk
from tkinter.filedialog import askopenfilename
from keras.models import model_from_json
import cv2
root=Tk()
filename=askopenfilename()
root.geometry("500x500")
image1=ImageTk.PhotoImage(Image.open(filename))
if filename=='':
    exit()
json_file=open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
image=cv2.imread(filename)
image=cv2.resize(image,(180,180))
image=image.reshape(-1,180,180,3)
output=loaded_model.predict(image)
output=list(output[0])
m=max(output)
ind=output.index(m)
d={0:'healthy',1:'parkinson'}
print('result:',d[ind])
Label(root,text=d[ind],image=image1,compound='top').pack()
b2=Button(root,text='exit',command=root.destroy).pack()
root.mainloop()
