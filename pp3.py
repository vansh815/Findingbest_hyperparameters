#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 19:42:02 2019

@author: vanshsmacpro
"""

import numpy as np
import timeit
import sys
import matplotlib.pyplot as plt

# method to split data in 0.1 .... 1 for train data 

def split_data(train_data , train_label):
    matrix_data = []
    matrix_label = []
    n = int(0.1*len(train_label))
    x = 0.1*len(train_label)
    count = 0 
    flag = 1
    while n <= len(train_label) : 
        traini = []
        testi = []
        for i in range(n):
            
            traini.append(train_data[count])
            testi.append(train_label[count])
            count = count + 1
            
            if count == (n-1):
                flag = flag + 1
                n = int( flag*x  )
                count = 0  
                
        matrix_data.append(traini)
        matrix_label.append(testi)
    return matrix_data , matrix_label

# optimization function which returns the w whenever called 

def optimization_technique(d,r,phi,w):
    phi_t = phi.T
    alpha = 10
    I = np.identity(len(phi[0]), dtype = float)
    H_without_i = phi_t.dot(r)
    H_without_i = -H_without_i.dot(phi) - I.dot(alpha)
    H_inverse = np.linalg.inv(H_without_i)
    g = phi_t.dot(d) - w.dot(alpha)
    
    new_w = w - H_inverse.dot(g)
    return new_w
    
# logistic method to calculate W
def calculate_train_logistic(matrix_data, matrix_label , number_features):
    w = number_features*[0]
    w = np.array(w)
    w_all = []
    position = []
    time_for_all = []
    start = 0 
    for j in range(len(matrix_data)):
        for i in range(100):
        
            phi = matrix_data[j]
            phi = np.array(phi)
        
        
            y_without_logi = phi.dot(w)
            y = 1/(1 + np.exp(-y_without_logi))
            t = matrix_label[j]
            d = t - y
            r = y.dot(1-y)
            new_w = optimization_technique(d,r,phi,w) 
            temp = w
            w = new_w
            stopping_condition = np.sqrt(np.sum(np.square(w - temp)))/np.sqrt(np.sum(np.square(temp)))
            
            if stopping_condition <= 0.001 or i == 99 : 
                position.append(i)
                end = timeit.default_timer()
                time = end - start
                time_for_all.append(time)
                break
            
                
           
           
        w_all.append(w)
    return w_all , position , time_for_all

# poisson method to calculate W

def calculate_train_Poisson(matrix_data, matrix_label, number_features):
    w = number_features*[0]
    w = np.array(w)
    w_all = []
    position = []
    time_for_all = []
    start = 0 
    for j in range(len(matrix_data)):
        start = timeit.default_timer()
        for i in range(100):
        
            phi = matrix_data[j]
            phi = np.array(phi)
            
            a = phi.dot(w)
            y = np.exp(a)
            t = matrix_label[j]
            d = t - y
            r = np.diag(y)
        
            new_w = optimization_technique(d,r,phi,w) 
            temp = w
            w = new_w
            stopping_condition = np.sqrt(np.sum(np.square(w - temp)))/np.sqrt(np.sum(np.square(temp)))
            
            if stopping_condition <= 0.001 or i == 99 : 
                position.append(i)
                end = timeit.default_timer()
                time = end - start
                time_for_all.append(time)
                break
            
                
           
           
        w_all.append(w)
    return w_all , position , time_for_all

# ordinal method to calculate W

def calculate_train_Ordinal(matrix_data, matrix_label, number_features):
    w = number_features*[0]
    w = np.array(w)
    S = 1
    w_all = []
    position = []
    time_for_all = []
    S_square = np.square(S)
    phi_all = [-np.inf , -2 , -1 , 0 , 1, np.inf]
    start = 0 
    for j in range(len(matrix_data)):
        start = timeit.default_timer()
        for i in range(100):
            y_ij = []
            r = []
            d = []
            phi = matrix_data[j]
            phi = np.array(phi)
            
            a = phi.dot(w)
            
            for z in range(len(phi_all)):
                y = 1/(1+np.exp(-S*(phi_all[z] - a)))
                y_ij.append(y)
            y_ij = np.array(y_ij)
            y_ij_t = y_ij.T
            
            
            t_all = matrix_label[j]
            for z in range(len(y_ij_t)):
                t_now = int(t_all[z])
                y_now = y_ij_t[z]
                d_all = y_now[t_now] + y_now[t_now-1] - 1
                d.append(d_all)
                r_all = S_square*(y_now[t_now]*(1-y_now[t_now]) + y_now[t_now - 1]*(1 - y_now[t_now]))
                r.append(r_all)
                
            r = np.diag(r)
            new_w = optimization_technique(d,r,phi,w) 
            temp = w
            w = new_w
            stopping_condition = np.sqrt(np.sum(np.square(w - temp)))/np.sqrt(np.sum(np.square(temp)))
            
            if stopping_condition <= 0.001 or i == 99 : 
                position.append(i)
                end = timeit.default_timer()
                time = end - start
                time_for_all.append(time)
                break
            
                
           
           
        w_all.append(w)
    return w_all , position , time_for_all

# method to test poisson W and calculate error       

def calculate_test_Poisson(test_data, test_label,learning_parameter_w ):
    error_all = []
    for j in range(len(learning_parameter_w)):
        error = 0
    
        a = test_data.dot(learning_parameter_w[j])
        lambdaa = np.exp(a)
        error = 0 
        t_predicted = np.floor(lambdaa)
        for i in range(len(t_predicted)):
            error  = error + abs(test_label[i] - t_predicted[i])
    
        error_all.append(error/len(test_label))
    return error_all
    
 # method to test logistic W and calculate error 
    
def calculate_test_logistic(test_data, test_label,learning_parameter_w):   
    error_all = []
    for j in range(len(learning_parameter_w)):
        error = 0
    
        test_without_sigmoid = test_data.dot(learning_parameter_w[j])
        predicted = 1/(1+np.exp(-test_without_sigmoid))
        predicted_final = []
        for i in range(len(predicted)):
            if predicted[i] >= 0.5:
                predicted_final.append(1)
            else:
                predicted_final.append(0)
    
        for i in range(len(predicted)):
            if predicted_final[i] != test_label[i]:
                error = error + 1
        error_all.append(error/len(test_label))
    return error_all
# method to test ordinal W and calculate error 

def calculate_test_Ordinal(test_data, test_label,learning_parameter_w ):
    error_all = []
    S = 1
    phi_all = [-np.inf , -2 , -1 , 0 , 1, np.inf]
    for j in range(len(learning_parameter_w)):
        error = 0
        y_ij = []
        p_all = []
        a = test_data.dot(learning_parameter_w[j])
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
    
        error_all.append(error/len(test_label))
    return error_all
    
# fucntion to calculate the mean and standard deviation for all size of data 0.1 ..  1

def calculate_mean_sd(mean_sd):
    count = 0
    len(mean_sd)
    mean_same_train = []
    mean_sd_same = []
    while(count < len(mean_sd[0])):
        for i in range(len(mean_sd)) : 
            x = mean_sd[i]
            for j in range(len(mean_sd[0])):
                if count == j : 
                    mean_same_train.append(x[j])
                    
        count = count + 1
        mean_sd_same.append(mean_same_train)
        mean_same_train = []
    #print(mean_sd_same)                
    
    
    mean_all = []
    sd_all = []
    
    for i in mean_sd_same:
        mean = np.mean(i)
        mean_all.append(mean)
        sd = np.std(i)
        sd_all.append(sd)
    
    return mean_all , sd_all

# fucntion to plot graph 

def plot_graph(mean_all , sd_all , method):
    x = [0.1, 0.2, 0.3 , 0.4, 0.5 , 0.6, 0.7, 0.8, 0.9 , 1.0]
    z = "learning curve for " + method
    plt.errorbar(x, mean_all,sd_all, label = z)
    plt.xlabel("increasing data size")
    plt.ylabel("error")
    plt.legend()
    plt.show()

# main function 

if __name__ == "__main__":
    start = timeit.default_timer()
    
    data = np.genfromtxt(sys.argv[1], delimiter = ",")
    label = np.genfromtxt(sys.argv[2])
    method = str(sys.argv[3]) ### take this as user input 
    data = np.array(data)
    label = np.array(label)
    number_features = len(data[0])
    #print(data , label)
    mean_sd = []
    time_all = []
    position_all = []
    
    for i in range(30):
        index = np.random.permutation(len(label))
        data , label =  data[index] , label[index]
    
    #print(data, label)
    
        x = int(len(data)*2/3)
        train_label = label[:x]
        test_label = label[x:]
        train_data = data[:x]
        test_data = data[x:]
        len(test_data)
    # now splitting data in the range of 0.1, 0.2, 0.3 .... 1
        
        matrix_data , matrix_label = split_data(train_data ,train_label )
        if method == "Logistic":
            
            learning_parameter_w , position , time_for_all = calculate_train_logistic(matrix_data, matrix_label, number_features)
            
            time_all.append(time_for_all)
            position_all.append(position) 
        
            error_all = calculate_test_logistic(test_data, test_label,learning_parameter_w )
            mean_sd.append(error_all)
            
        elif method == "Poisson":
            
            learning_parameter_w , position , time_for_all = calculate_train_Poisson(matrix_data, matrix_label, number_features)
            
            time_all.append(time_for_all)
            position_all.append(position) 
        
            error_all = calculate_test_Poisson(test_data, test_label,learning_parameter_w )
            mean_sd.append(error_all)
            
        elif method == "Ordinal":
            ## only this is left !!!!!!!!
            
            learning_parameter_w , position , time_for_all = calculate_train_Ordinal(matrix_data, matrix_label, number_features)
            
            time_all.append(time_for_all)
            position_all.append(position) 
            
            error_all = calculate_test_Ordinal(test_data, test_label,learning_parameter_w )
            mean_sd.append(error_all)
    # calculate mean and sd 
    mean_all , sd_all = calculate_mean_sd(mean_sd)
    plot_graph(mean_all , sd_all , method)
    average_position , sd = calculate_mean_sd(position_all)
    average_time , sd = calculate_mean_sd(time_all)
    print("\n average position for convergence for different training size :")
    print(average_position)
    print("\n average time for convergence for different training size:") 
    print(average_time)    
    
            
    end = timeit.default_timer()
    time = end - start
    print("\n running time is"  , time)
    
    
    
    
    
    
    
    

    
    
    
    