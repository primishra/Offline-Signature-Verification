#!/usr/bin/env python
# coding: utf-8

# In[1]:


## reading the images files of train folder

import os
import cv2


path = "/home/rrupadhyay/signature/only_genuine/train/genuine"
unique_id = [] ## name or id of each image (same as the id in train_data.csv)
images = [] ## image array

for folder, a, b in os.walk(path):
    for file in b:
        img = cv2.imread(os.path.join(folder, file)) ## reading image as matrix
        
        if img is not None:
            images.append(img)
            
            ## creating  unique id for each image
            if "forg" in str(folder):
                unique_id.append(str(folder[-8:]) + "/" + str(file.lower()))
            else:
                unique_id.append(str(folder[-3:]) + "/" + str(file.lower()))
    





## grayscaling and resizing the images

resized_images = []
dimension = (60, 40) ## new size (width, height)

for img in images:
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## grayscaling
    
    resized_image = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA) ## resizing
    
    resized_images.append(resized_image)
    





# In[26]:


## finding co-ordinates

import numpy as np


height = [] ## Y coordinates in our case 
width = [] ## X coordinates in our case

for image in resized_images:   
    indices = np.where(image!= [0]) ## finds where signature curves lie 
    
    height.append(indices[0])
    width.append(indices[1])


# In[27]:


## creating dataframe from above info

import pandas as pd

features_1_df = pd.DataFrame(list(zip(unique_id, width, height)), columns = ["ID", "X", "Y"])


# In[28]:


del height


# In[29]:


del width


# In[30]:


features_1_df.head()


# In[31]:


## exploding X & Y columns of the dataframe

features = features_1_df.set_index('ID').apply(pd.Series.explode).reset_index()


# In[32]:


del features_1_df


# In[33]:


features.shape


# In[34]:


features.head()


# In[35]:


features.isnull().sum(axis = 0)


# In[36]:


## finding the position of pen-tip

import math

Xs = features["X"].tolist()
Ys = features["Y"].tolist()

Rs = []

for i in range(len(Xs)):
    Rs.append(math.sqrt(Xs[i]**2 + Ys[i]**2))

features["R"] = Rs ## "R" is the position column


# In[37]:


features.head()


# In[38]:


## angle of "R"

theta = []

for i in range(len(Xs)):
    try:
        theta.append(np.arctan(Ys[i]/Xs[i]))
    except ZeroDivisionError:
        theta.append(0)

features["Angle"] = theta


# In[39]:


del Xs, Ys, Rs, theta


# In[40]:


features.head()


# In[41]:


features.Angle.isna().sum(axis = 0)


# In[ ]:


## features.Angle = features.Angle.fillna(0)


# In[42]:


## finding change in X direction
features['del_x'] = np.where(features.ID == features.ID.shift(1), features.X.diff(), 0)
features['del_x'] = features.groupby("ID")["del_x"].shift(-1)


## finding change in Y direction
features['del_y'] = np.where(features.ID == features.ID.shift(1), features.Y.diff(), 0)
features['del_y'] = features.groupby("ID")["del_y"].shift(-1)


# In[43]:


features.head()


# In[44]:


## Magnitude and angle of displacement vector

del_x_l = features.del_x.tolist()
del_y_l = features.del_y.tolist()

disp = [] ## magnitude
disp_theta = [] ## angle

for i in range(len(del_x_l)):
    
    ## magnitude
    disp.append(math.sqrt(del_x_l[i]**2 + del_y_l[i]**2))
    
    ## angle
    try:
        disp_theta.append(np.arctan(del_y_l[i]/del_x_l[i]))
    except ZeroDivisionError:
        disp_theta.append(0)
    
features["displacement"] = disp
features["disp_angle"] = disp_theta


# In[45]:


del disp_theta


# In[46]:


features[56896:57000]


# In[ ]:


#features.to_csv(r"G:\Signature Verification Project\data processing\features.csv")


# In[ ]:


#import pandas as pd

#features = pd.read_csv(r"G:\Signature Verification Project\data processing\features.csv")


# In[ ]:


#features = features.drop(["Unnamed: 0"], axis = 1)


# In[ ]:


#del_x_l = features.del_x.tolist()
#del_y_l = features.del_y.tolist()
#disp = features.displacement.tolist()


# In[47]:


cos_angle = []
sine_angle = []

for i in range(len(del_x_l)):
    
    ## cosine angle (angle between x-axis and signature curve)
    cos_angle.append((del_x_l[i])/(disp[i]))
    
    ## sine angle (angle between y-axis and signature curve)
    sine_angle.append((del_y_l[i])/(disp[i]))
    
features["cos_angle"] = cos_angle
features["sine_angle"] = sine_angle


# In[48]:


del cos_angle, sine_angle, del_x_l, del_y_l, disp


# In[49]:


features.columns


# In[50]:


features.head()


# In[51]:


## finding time
features["Time"] = features.groupby(["ID"]).cumcount()+1


# In[52]:


## finding change in Time
features['del_t'] = np.where(features.ID == features.ID.shift(1), features.Time.diff(), 0)
features['del_t'] = features.groupby("ID")["del_t"].shift(-1)


# In[53]:


features.head()


# In[54]:


del_x_l = features.del_x.tolist()
del_y_l = features.del_y.tolist()
del_t_l = features.del_t.tolist()

vel_x = [] ## velocity in x-direction
vel_y = [] ## velocity in y-direction

for i in range(len(del_x_l)):
    vel_x.append(del_x_l[i]/del_t_l[i])
    vel_y.append(del_y_l[i]/del_t_l[i])
    
features["vel_x"] = vel_x
features["vel_y"] = vel_y


# In[55]:


del del_x_l, del_y_l, del_t_l


# In[56]:


features.head()


# In[57]:


res_vel = []  ## resultant velocity 
dir_res_vel = []  ## direction of the resultant velocity

for i in range(len(vel_x)):
    res_vel.append(math.sqrt(vel_x[i]**2 + vel_y[i]**2))
    
    try:
        dir_res_vel.append(np.arctan(vel_y[i]/vel_x[i]))
    except ZeroDivisionError:
        dir_res_vel.append(0)
        
features["res_vel"] = res_vel
features["dir_res_vel"] = dir_res_vel


# In[58]:


del vel_x, vel_y, res_vel, dir_res_vel


# In[59]:


del_t_l = features.del_t.tolist()

## finding change in displacement angle
features['del_disp_angle'] = np.where(features.ID == features.ID.shift(1), features.disp_angle.diff(), 0)
features['del_disp_angle'] = features.groupby("ID")["del_disp_angle"].shift(-1)
## converting it to list
del_disp_angle = features.del_disp_angle.tolist()

angular_vel = [] ## angular velocity

for i in range(len(del_disp_angle)):
    angular_vel.append(del_disp_angle[i]/del_t_l[i])

features["angular_vel"] = angular_vel


# In[60]:


del del_disp_angle, del_t_l, angular_vel


# In[61]:


features.head()


# In[62]:


del_vel_x = features.vel_x.tolist()
del_vel_y = features.vel_y.tolist()
del_t_l = features.del_t.tolist()

acceleration_x = [] ## acceleration in x-dir
acceleration_y = [] ## acceleration in y-dir
res_acceleration = [] ## resultant acceleration

for i in range(len(del_t_l)):
    acceleration_x.append(del_vel_x[i]/del_t_l[i])
    acceleration_y.append(del_vel_y[i]/del_t_l[i])
    
    res_acceleration.append(math.sqrt(acceleration_x[i]**2 + acceleration_y[i]**2))
    
features["acceleration_x"] = acceleration_x
features["acceleration_y"] = acceleration_y
features["res_acceleration"] = res_acceleration

del del_vel_x, del_vel_y, del_t_l, acceleration_x, acceleration_y, res_acceleration


# In[63]:


features.head()


# In[64]:


angular_vel = features.angular_vel.tolist()
displacement = features.displacement.tolist()
res_acceleration = features.res_acceleration.tolist()

centripetal_accln = [] ## centripetal acceleration
tangential_accln = [] ## tangential acceleration

for i in range(len(angular_vel)):
    centripetal_accln.append((angular_vel[i]**2)*displacement[i])
    
    tangential_accln.append(math.sqrt(res_acceleration[i]**2 - centripetal_accln[i]))
    
features["centripetal_accln"] = centripetal_accln
features["tangential_accln"] = tangential_accln

del angular_vel, res_acceleration, displacement, centripetal_accln, tangential_accln


# In[65]:


features.head()


# In[66]:


features.columns


# In[67]:


features = features.drop(["Time", "del_t", "del_disp_angle"], axis = 1)


# In[68]:


features.head()


# In[69]:


features["Angle"] = features["Angle"].fillna(0)
features["disp_angle"] = features["disp_angle"].fillna(0)
features["dir_res_vel"] = features["dir_res_vel"].fillna(0)


# In[70]:


features.head()


# In[71]:


features.to_csv("/home/rrupadhyay/signature/prep_data/genuine_train.csv", index = False)


# ## 1st phase Feature extraction ends here.
