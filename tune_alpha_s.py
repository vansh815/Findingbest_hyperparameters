#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 19:42:02 2019

@author: vanshsmacpro
"""

import numpy as np

# optimization technique 


def optimization_technique(d,r,phi,w , alpha):
    phi_t = phi.T
    #alpha = 10
    I = np.identity(len(phi[0]), dtype = float)
    H_without_i = phi_t.dot(r)
    H_without_i = -H_without_i.dot(phi) - I.dot(alpha)
    H_inverse = np.linalg.inv(H_without_i)
    g = phi_t.dot(d) - w.dot(alpha)
    
    new_w = w - H_inverse.dot(g)
    return new_w
    

# method to calculate w for ordinal 


def calculate_train_Ordinal(matrix_data, matrix_label, number_features,alpha, S):
    w = number_features*[0]
    w = np.array(w)
    S = 10
    S_square = np.square(S)
    phi_all = [-np.inf , -2 , -1 , 0 , 1, np.inf]
    for i in range(100):
        y_ij = []
        r = []
        d = []
        phi = matrix_data
        phi = np.array(phi)
            
        a = phi.dot(w)
            
        for z in range(len(phi_all)):
            y = 1/(1+np.exp(-S*(phi_all[z] - a)))
            y_ij.append(y)
        y_ij = np.array(y_ij)
        y_ij_t = y_ij.T
            
            
        t_all = matrix_label
        for z in range(len(y_ij_t)):
            t_now = int(t_all[z])
            y_now = y_ij_t[z]
            d_all = y_now[t_now] + y_now[t_now-1] - 1
            d.append(d_all)
            r_all = S_square*(y_now[t_now]*(1-y_now[t_now]) + y_now[t_now - 1]*(1 - y_now[t_now]))
            r.append(r_all)
                
        r = np.diag(r)
        new_w = optimization_technique(d,r,phi,w , alpha) 
        temp = w
        w = new_w
        stopping_condition = np.sqrt(np.sum(np.square(w - temp)))/np.sqrt(np.sum(np.square(temp)))
            
        if stopping_condition <= 0.001 or i == 99 : 
            break
            
                
           
           
        
    return w

# method to calculate error for ordinal  

def calculate_test_Ordinal(test_data, test_label,learning_parameter_w , S):
   
    
    phi_all = [-np.inf , -2 , -1 , 0 , 1, np.inf]
    error = 0
    y_ij = []
    p_all = []
    a = test_data.dot(learning_parameter_w)
    predicted_all = []
    predicted_final = []
    for z in range(len(phi_all)):
        y = 1/(1+np.exp(-S*(phi_all[z] - a)))
        y_ij.append(y)
    y_ij = np.array(y_ij)
        
    count = len(y_ij) - 1
    while count > 0:
        p = y_ij[count] - y_ij[count-1]
        p_all.append(p)
        count = count -1
    p_all = np.array(p_all)
    p_t = p_all.T
    len(p_t)
    for z in range(len(p_t)):
        data_now = p_t[z]
        predicted = np.argmax(data_now)
        predicted_all.append(predicted)
        
    len(predicted_all)
        
    for z in range(len(predicted_all)):
        predicted_f = abs(5 - predicted_all[z])
        predicted_final.append(predicted_f)
        
    len(predicted_final) 
            
        
    for z in range(len(predicted_final)):
        error  = error + abs(test_label[z] - predicted_final[z])
    
                
    return error/len(test_label)
# function to call  ordinal train and test methods

def tuning_param(alpha_1 , matrix_data , matrix_label, number_features , test_data , test_label , method , S):
    
    if method == "Ordinal" : 
        learning_parameter_w  = calculate_train_Ordinal(matrix_data, matrix_label, number_features , alpha_1, S )
            
    
        
        error = calculate_test_Ordinal(test_data, test_label,learning_parameter_w, S )
        return error 

## main function 

if __name__ == "__main__":
   
    
    data = np.genfromtxt("/Users/vanshsmacpro/Desktop/sem1/ml assignments/pp3data/AO.csv", delimiter = ",")
    label = np.genfromtxt("/Users/vanshsmacpro/Desktop/sem1/ml assignments/pp3data/labels-AO.csv")
    method = "Ordinal" ### take this as user input 
    data = np.array(data)
    label = np.array(label)
    number_features = len(data[0])
    #print(data , label)
    mean_sd = []
    time_all = []
    position_all = []
    
    #index = np.random.permutation(len(label))
    #data , label =  data[index] , label[index]
    
    #print(data, label)
    
    x = int(len(data)*2/3)
    train_label = label[:x]
    test_label = label[x:]
    train_data = data[:x]
    test_data = data[x:]
    len(test_data)
    # now splitting data in the range of 0.1, 0.2, 0.3 .... 1
        
    matrix_data , matrix_label = train_data ,train_label 
    
    error_all = []
    alpha_all = [] 
    # considering it as an error  of +- 5
    alpha = 19
    alpha_1 = alpha - 5
    alpha_2 = alpha + 5
    for j in range(10): 
        S1 = 1
        S2 = 20
        
        error_1 = tuning_param(alpha , matrix_data , matrix_label, number_features , test_data , test_label , method , S1) 
    
        error_2  = tuning_param(alpha , matrix_data , matrix_label, number_features , test_data , test_label , method , S2)
    
        for i in range(30):
            S3 = random.randint(S1, S2)
            error_3  = tuning_param(alpha , matrix_data , matrix_label, number_features , test_data , test_label , method ,S3)
   
            if error_3 < error_2 : 
                S2 = S3
                error_2 = tuning_param(alpha , matrix_data , matrix_label, number_features , test_data , test_label , method , S2) 
            elif error_3 > error_1 and error_3 > error_1 :
                S2 = S3
                error_2 = tuning_param(alpha , matrix_data , matrix_label, number_features , test_data , test_label , method, S2) 
                    
           
        error_all.append(error_2)
        alpha_all.append(S2)
    

        
    index = np.argmin(error_all)
    S = alpha_all[index]
    S2 = S + 3
    alpha_all = []
    S1 =  S - 2
    error_now = []
    index_all_S = []
    for i in range(alpha_1, alpha_2):
        for j in range(S1 , S2):
            error_3  = tuning_param(i , matrix_data , matrix_label, number_features , test_data , test_label , method ,j)
            error_now.append(error_3)
            
        error_all.append(np.min(error_now))
        index = np.argmin(error_now)
        index_all_S.append(index)
        error_now = []
    
    index = np.argmin(error_all)
    S = index_all_S[index]
    for i in range(alpha_1,alpha_2):
        alpha_all.append(i)
    alpha = alpha_all[index]
    error = np.min(error_all)
    
    print(S , alpha)
    
    
    
    
    

    
    
    
    