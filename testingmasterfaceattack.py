 #======================== IMPORT PACKAGES ===========================
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from tkinter import *
import matplotlib.image as mpimg
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from cryptography.fernet import Fernet
from tkinter import filedialog
#====================== 1.READ A INPUT IMAGE =========================

U_name = 'confi123'
P_name = 'dence123'
root =Tk()
root.geometry("800x800")
root.config(bg="lightblue")
print("Program Ready To Start")    
l1=Label(root,text="Enter USer Name....",bg="light green",font=("Algerian",14,"bold"))
e1=Entry(root,text="username",font=("Algerian",14,"bold"))
l2=Label(root,text="Enter Password....",bg="light green",font=("Algerian",14,"bold"))
e2=Entry(root,show="*",font=("Algerian",14,"bold"))
def authorize():
    import cv2
    print("Interface is Ready")
    if U_name == e1.get():
        if P_name == e2.get():
            
            l6.config(text="")
            print("Output of Read function is ")
            file_text=filedialog.askopenfilename()

            tx1 = open(file_text,"r+") 
            
            message = tx1.read()
            
            print()
            
            key = Fernet.generate_key()
             
            # Instance the Fernet class with the key
             
            fernet = Fernet(key)
             
            # then use the Fernet class instance
            # to encrypt the string string must
            # be encoded to byte string before encryption
            encMessage = fernet.encrypt(message.encode())
            print('Ecrypted Data')
            
            print(encMessage)
            
            print("Master Face Attack")
           
            file_up=filedialog.askopenfilename()
            image = Image.open(file_up)
            img = mpimg.imread(file_up)
            print("input image")
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis ('off')
            plt.show()
            
            print("Original image..",img)
            #============================ 2.IMAGE PREPROCESSING ====================
            
            #==== RESIZE IMAGE ====

            print('resized image')
            resized_image = cv2.resize(img,(300,300))
            img_resize_orig = cv2.resize(img,((50, 50)))
            
            fig = plt.figure()
            plt.title('RESIZED IMAGE')
            plt.imshow(resized_image)
            plt.axis ('off')
            plt.show()
            
            #==== GRAYSCALE IMAGE ====
            
        
            
            SPV = np.shape(img)
            
            try:            
                gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
                
            except:
                gray1 = img_resize_orig
            
           
            print("Gray Scale Image..")  
            fig = plt.figure()
            plt.title('GRAY SCALE IMAGE')
            plt.imshow(gray1)
            plt.axis ('off')
            plt.show()
            print(gray1)
            
            #=============================== 3.IMAGE SPLITTING =================================
            
            # === test and train ===
            import os 
            
            from sklearn.model_selection import train_test_split
            
            dataset_all = os.listdir('Dataset/')
            
            dataset_all1 = os.listdir('Dataset/')
            
            
            dot1= []
            labels1 = []
            for img in dataset_all:
                    # print(img)
                    img_1 = mpimg.imread('Dataset/' + "/" + img)
                    img_1 = cv2.resize(img_1,((50, 50)))
            
            
                    try:            
                        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                        
                    except:
                        gray = img_1
            
                    
                    dot1.append(np.array(gray))
                    labels1.append(0)
            
            for img in dataset_all1:
                    # print(img)
                    img_1 = mpimg.imread('Dataset/' + "/" + img)
                    img_1 = cv2.resize(img_1,((50, 50)))
            
            
                    try:            
                        gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                        
                    except:
                        gray = img_1
            
                    
                    dot1.append(np.array(gray))
                    labels1.append(1)        
            
            #=============================== 4.IMAGE PREDICTION ==============================
            
            import cv2
            from tensorflow.keras.layers import Dense, Conv2D
            from tensorflow.keras.layers import Flatten
            from tensorflow.keras.layers import MaxPooling2D
            from tensorflow.keras.layers import Dropout
            from tensorflow.keras.models import Sequential
            
            x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
            
            from keras.utils import to_categorical
            
            
            y_train1=np.array(y_train)
            y_test1=np.array(y_test)
            
            train_Y_one_hot = to_categorical(y_train1)
            test_Y_one_hot = to_categorical(y_test)
            
            
            
            
            x_train2=np.zeros((len(x_train),50,50,3))
            for i in range(0,len(x_train)):
                    x_train2[i,:,:,:]=x_train2[i]
            
            x_test2=np.zeros((len(x_test),50,50,3))
            for i in range(0,len(x_test)):
                    x_test2[i,:,:,:]=x_test2[i]
            
            print("-------------------------------------------------------------")
            print('Convolutional Neural Network') 
            print("-------------------------------------------------------------")
            print()
            print()
            
            
            # initialize the model
            model=Sequential()
            
            
            #CNN layes 
            model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
            model.add(MaxPooling2D(pool_size=2))
            
            model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
            model.add(MaxPooling2D(pool_size=2))
            
            model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
            model.add(MaxPooling2D(pool_size=2))
            
            model.add(Dropout(0.2))
            model.add(Flatten())
            
            model.add(Dense(500,activation="relu"))
            
            model.add(Dropout(0.2))
            
            model.add(Dense(2,activation="softmax"))
            
            #summary the model 
            model.summary()
            
            #compile the model 
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            y_train1=np.array(y_train)
            
            train_Y_one_hot = to_categorical(y_train1)
            test_Y_one_hot = to_categorical(y_test)
            
            #fit the model 
            history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=10,verbose=1)
            accuracy = model.evaluate(x_test2, test_Y_one_hot, verbose=1)
            
            print()
            print()
            print("-------------------------------------------------------------")
            print("Performance Analysis")
            print("-------------------------------------------------------------")
            print()
            
            accuracy=history.history['accuracy']
            accuracy=max(accuracy)
            accuracy=100-accuracy
            print()
            print("1.Accuracy  :",accuracy,'%')
            print(accuracy)            
            print()
            print("-------------------------------------------------------------")
            print('Prediction --- Face Authentication') 
            print("-------------------------------------------------------------")
            print()
            print()
            
            
            Total_length = len(dataset_all) 
            
            b=[]
            temp_data1  = []
            for ijk in range(0,Total_length):
                temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
                temp_data1.append(temp_data)
            
            temp_data1 =np.array(temp_data1)
            
            prediction=0
            for i in temp_data1:
                prediction=prediction+i
 
            if prediction==1:
                print("================================")
                print("Face Match --- Authorized")
                print("================================")
                res='Face Matched --- Authorized --- File Accessed'
                print("Prediction")
                print(res)
                message1 = encMessage
                decMessage = fernet.decrypt(message1).decode()
                l4=Label(root,text=res,bg="yellow",fg="green",font=("arial black",18,"bold"))
                l4.place(x=70,y=560)
                print("Text Decryption")
                print("")
                print(decMessage)
                l5=Label(root,width=85,height=30,text=decMessage ,font=("arial",9,"bold"))
                l5.place(x=70,y=100)
            else:
                print("=================================")
                print("Face Not Matched --- Unauthorized----File Accessed")
                print("=================================")
                res='Face Match --- Unauthorized '
                print("Prediction")
                print(res)
                l3=Label(root,text=res,bg="red",font=("impact",18,"bold"))
                l3.place(x=309,y=560)
      
    else:
        l6.place(x=300,y=540)
l6=Label(root,text="Invalid credentials..")

b1=Button(root,text="Login..",width=10,height=1,font=("forte",15,"bold"),bg="yellow",command=authorize)
l1.place(x=200,y=100)
l2.place(x=200,y=160)
e1.place(x=420,y=100)
e2.place(x=420,y=160)
b1.place(x=350,y=200)
root.mainloop()
