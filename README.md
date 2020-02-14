Here I have implemented Poisson, Ordinal and Logistic Classification. 
The project is to check that for each method and on each dataset , what is best set of hyper parameters and to learn and better the learning curve. The main task is that all the three classifications are implemented using generalized model using the Newton's method. Hence with a single piece of code one can run all the above three methods by just changing the user input method name. 

The Report,learning curves,best set of hyper-parameters are presented in the report file. 


How to run :
python3 (pp3.py) datafile labelfile methodname

method name should be [Poisson , Ordinal , Logistic]
please take care for uppercase letters.

for finding the values of alpha :
python3 tune_alpha.py datafile labelfile methodname

for finding the best value of alpha and S at the same time 
python3 tune_alpha_S.py
