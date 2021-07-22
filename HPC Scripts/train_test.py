#!/usr/bin/env python
# coding: utf-8

import pandas as pd

############################  TEST #######################################################

genuine = pd.read_csv("/home/rrupadhyay/signature/prep_data/genuine_test.csv")


gen_forg = pd.read_csv("/home/rrupadhyay/signature/prep_data/gen_forg_test.csv")



genuine = genuine.groupby('ID').agg(lambda x: list(x)).reset_index()

gen_forg = gen_forg.groupby('ID').agg(lambda x: list(x)).reset_index()


genuine['g_key'] = genuine['ID'].str[:3]
gen_forg['fg_key'] = gen_forg['ID'].str[:3]


dicti = {"ID":"g_ID",
         "X":"g_X",
         "Y":"g_Y",
         'R':"g_R", 'Angle':"g_Angle", 'del_x':"g_del_x", 'del_y':"g_del_y",
       'displacement':"g_displacement", 'disp_angle':"g_disp_angle", 'cos_angle':"g_cos_angle", 'sine_angle':"g_sine_angle", 
         'vel_x':"g_vel_x", 'vel_y':"g_vel_y", 'res_vel':"g_res_vel", 'dir_res_vel':"g_dir_res_vel", 'angular_vel':"g_angular_vel", 
         'acceleration_x':"g_acceleration_x",
       'acceleration_y':"g_acceleration_y", 'res_acceleration':"g_res_acceleration", 'centripetal_accln':"g_centripetal_accln",
       'tangential_accln':"g_tangential_accln"}

genuine.rename(columns = dicti, inplace = True)



dicti_fg = {"ID":"fg_ID",
         "X":"fg_X",
         "Y":"fg_Y",
         'R':"fg_R", 'Angle':"fg_Angle", 'del_x':"fg_del_x", 'del_y':"fg_del_y",
       'displacement':"fg_displacement", 'disp_angle':"fg_disp_angle", 'cos_angle':"fg_cos_angle", 'sine_angle':"fg_sine_angle", 
         'vel_x':"fg_vel_x", 'vel_y':"fg_vel_y", 'res_vel':"fg_res_vel", 'dir_res_vel':"fg_dir_res_vel", 'angular_vel':"fg_angular_vel", 
         'acceleration_x':"fg_acceleration_x",
       'acceleration_y':"fg_acceleration_y", 'res_acceleration':"fg_res_acceleration", 'centripetal_accln':"fg_centripetal_accln",
       'tangential_accln':"fg_tangential_accln"}

gen_forg.rename(columns = dicti_fg, inplace = True)

genuine.rename(columns = {"g_key": "key"}, inplace = True)
gen_forg.rename(columns = {"fg_key": "key"}, inplace = True)

result = pd.merge(genuine, gen_forg, on ='key').drop("key", 1)

test_data = pd.read_csv("/home/rrupadhyay/signature/gen_forge/test_data.csv")

result["g_ID"][:2], result["fg_ID"][:2]

result["new_key"] = result[["g_ID", "fg_ID"]].apply(lambda x: "_".join(x), axis = 1)
test_data["new_key"] = test_data[["ID", "opposite_image"]].apply(lambda x: "_".join(x), axis = 1)

result["new_key"].head(), test_data["new_key"].head()

final_test = result.merge(test_data, how = "left", left_on = "new_key", right_on = "new_key")

final_test = final_test.drop(["new_key", "ID", "opposite_image"], axis = 1)

print("complete test")

print(final_test.head())
print(final_test.shape)

del genuine, gen_forg, dicti, dicti_fg, result, test_data

############################  Train  #######################################################


genuine = pd.read_csv("/home/rrupadhyay/signature/prep_data/genuine_train.csv")


gen_forg = pd.read_csv("/home/rrupadhyay/signature/prep_data/gen_forg_train.csv")



genuine = genuine.groupby('ID').agg(lambda x: list(x)).reset_index()

gen_forg = gen_forg.groupby('ID').agg(lambda x: list(x)).reset_index()


genuine['g_key'] = genuine['ID'].str[:3]
gen_forg['fg_key'] = gen_forg['ID'].str[:3]


dicti = {"ID":"g_ID",
         "X":"g_X",
         "Y":"g_Y",
         'R':"g_R", 'Angle':"g_Angle", 'del_x':"g_del_x", 'del_y':"g_del_y",
       'displacement':"g_displacement", 'disp_angle':"g_disp_angle", 'cos_angle':"g_cos_angle", 'sine_angle':"g_sine_angle", 
         'vel_x':"g_vel_x", 'vel_y':"g_vel_y", 'res_vel':"g_res_vel", 'dir_res_vel':"g_dir_res_vel", 'angular_vel':"g_angular_vel", 
         'acceleration_x':"g_acceleration_x",
       'acceleration_y':"g_acceleration_y", 'res_acceleration':"g_res_acceleration", 'centripetal_accln':"g_centripetal_accln",
       'tangential_accln':"g_tangential_accln"}

genuine.rename(columns = dicti, inplace = True)



dicti_fg = {"ID":"fg_ID",
         "X":"fg_X",
         "Y":"fg_Y",
         'R':"fg_R", 'Angle':"fg_Angle", 'del_x':"fg_del_x", 'del_y':"fg_del_y",
       'displacement':"fg_displacement", 'disp_angle':"fg_disp_angle", 'cos_angle':"fg_cos_angle", 'sine_angle':"fg_sine_angle", 
         'vel_x':"fg_vel_x", 'vel_y':"fg_vel_y", 'res_vel':"fg_res_vel", 'dir_res_vel':"fg_dir_res_vel", 'angular_vel':"fg_angular_vel", 
         'acceleration_x':"fg_acceleration_x",
       'acceleration_y':"fg_acceleration_y", 'res_acceleration':"fg_res_acceleration", 'centripetal_accln':"fg_centripetal_accln",
       'tangential_accln':"fg_tangential_accln"}

gen_forg.rename(columns = dicti_fg, inplace = True)

genuine.rename(columns = {"g_key": "key"}, inplace = True)
gen_forg.rename(columns = {"fg_key": "key"}, inplace = True)

result = pd.merge(genuine, gen_forg, on ='key').drop("key", 1)

train_data = pd.read_csv("/home/rrupadhyay/signature/gen_forge/train_data.csv")

result["g_ID"][:2], result["fg_ID"][:2]

result["new_key"] = result[["g_ID", "fg_ID"]].apply(lambda x: "_".join(x), axis = 1)
train_data["new_key"] = train_data[["ID", "opposite_image"]].apply(lambda x: "_".join(x), axis = 1)

result["new_key"].head(), train_data["new_key"].head()

final_train = result.merge(train_data, how = "left", left_on = "new_key", right_on = "new_key")

final_train = final_train.drop(["new_key", "ID", "opposite_image"], axis = 1)

print("complete train")

print(final_train.head())
print(final_train.shape)

del genuine, gen_forg, dicti, dicti_fg, result, train_data
