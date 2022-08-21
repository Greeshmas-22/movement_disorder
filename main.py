import os
import cv2
import numpy
import tensorflow
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Model
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
path=os.getcwd()
data_path=os.path.join(path,"archive","drawings","spiral","training")
data_list=os.listdir(data_path)
x=[]
y=[]
count=0
t=0
for i in data_list:
    subfolder=os.path.join(data_path,i)
    subfolder_list=os.listdir(subfolder)
    for j in subfolder_list:
        image_path=os.path.join(subfolder,j)
        print(image_path)
        image=cv2.imread(image_path)
        cv2.imshow('image',image)
        cv2.waitKey(100)
        image=cv2.resize(image,(180,180))
        x.append(image)
        y.append(count)
        t=t+1
    count=count+1
print(y)
cv2.destroyAllWindows()
x=numpy.array(x)
x=x.reshape(t,180,180,3)
y=numpy.array(y)

resnet_model=tensorflow.keras.applications.ResNet50(include_top=False,
input_shape=(180,180,3),pooling='avg',classes=2,weights='imagenet')
resnet_model.summary()
for layer in resnet_model.layers:
    layer.trainable=False

m=resnet_model.output
m=Dense(4096,activation='relu')(m)
m=Dense(4096,activation='relu')(m)
m=Dense(2,activation='softmax')(m)

new_model=Model(resnet_model.input,m)
new_model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.0001),metrics=['accuracy'])
new_model.summary()
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)
ytrain1=[]
ytest1=[]
for i in ytrain:
    zero_list=[0,0]
    zero_list[i]=1
    ytrain1.append(zero_list)
for i in ytest:
    zero_list=[0,0]
    zero_list[i]=1
    ytest1.append(zero_list)
ytrain1=numpy.array(ytrain1)
ytest1=numpy.array(ytest1)
history=new_model.fit(xtrain,ytrain1,epochs=15,verbose=1,validation_data=(xtest,ytest1),batch_size=8)
loss,accuracy=new_model.evaluate(xtest,ytest1,verbose=1,batch_size=24)
print(loss)
print(accuracy)

plt.title('trainingprogress-loss')
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.title('trainingprogress-accuracy')
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

prediction=new_model.predict(xtest)
y_list=[]

for i in prediction:
    i_list=list(i)
    output=max(i_list)
    ind=i_list.index(output)
    y_list.append(ind)
tp=0
fp=0
fn=0
tn=0

for i in range(0,len(ytest)):
    if ytest[i]==0 and y_list[i]==0:
        tp=tp+1
    elif ytest[i]==0 and y_list[i]==1:
        fp=fp+1
    elif ytest[i]==1 and y_list[i]==0:
        fn=fn+1
    else:
        tn=tn+1

cm=[[tp,fp],[fn,tn]]
cm=numpy.array(cm)
display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['healthy','parkinson'])
display.plot()
plt.title('resnet')
plt.show()

p1=tp/(tp+fp)
p2=tn/(tn+fn)
r1=tp/(tp+fn)
r2=tn/(tn+fp)
print('precision for class healthy:',p1)
print('precision for class parkinson:',p2)
print('recall for class healthy:',r1)
print('recall for class parkinson:',r2)
f1=2*((p1*r1)/(p1+r1))
f2=2*((p2*r2)/(p2+r2))
print('f1 score for class healthy:',f1)
print('f1 score for class parkinson:',f2)
paverage=(p1+p2)/2
print('average for precision',paverage)
raverage=(r1+r2)/2
print('average for recall',raverage)
f1average=(f1+f2)/2
print('average for f1 score',f1average)
model_json=new_model.to_json()
with open('model.json','w')as json_file:
    json_file.write(model_json)
new_model.save_weights('model.h5')




                



        

        

    

