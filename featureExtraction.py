"""
featureExtraction.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: John See, 2017
@modified by: Lai Kuan, Wong, 2018

"""
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from computeFeatures import computeFeatures

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
dbpath = 'D:\\MMU\\course\\VIP\\2018Assignment2\\plantdb\\train'

# these labels are the classes assigned to the actual plant names
labels = ('C1','C2','C3','C4','C5','C6','C7','C8','C9','C10')   # BUG FIX 1: changed class label from integer to string
    
featvect = []  # empty list for holding features
FEtime = np.zeros(500)

for idx in range(500):
    img = cv2.imread( os.path.join(dbpath, str(idx+1) + ".jpg") )  # BUG FIX 2: str(idx) --> str(idx+1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # display image
    #plt.imshow(img), plt.xticks([]), plt.yticks([])
    #plt.show()
    
    # compute features and append to list
    e1 = cv2.getTickCount() # start timer
    feat = computeFeatures(img)
    e2 = cv2.getTickCount()  # stop timer
    
    featvect.append( feat ); 
    FEtime[idx] = (e2 - e1) / cv2.getTickFrequency() 
    
    print('Extracting features for image #%d'%idx )

print('Feature extraction runtime: %.4f seconds'%np.sum(FEtime))

temparr = np.array(featvect)
fv = np.reshape(temparr, (temparr.shape[0], temparr.shape[1]) )
del temparr

# pickle your features
pickle.dump( fv, open( "feat.pkl", "wb" ) )
print('Features pickled!')
