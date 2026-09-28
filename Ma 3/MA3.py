""" MA3.py

Student: Sara Basim
Mail: Sara.basim.4566@student.uu.se
Reviewed by:
Date reviewed:

"""
# Making a change for git 
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1
def approximate_pi(n):
    x_i = [] 
    y_i = []
    x_u= [] 
    y_u= []
    
    for i in range (n): 
        x= random.uniform (-1, 1)
        y = random.uniform (-1, 1)
        if x**2 + y**2 <= 1:
            x_i.append (x)
            y_i.append (y)
        else: 
            x_u.append (x) 
            y_u.append (y) 
                
    nc = len (x_i)
    pi_approx = 4* nc / n
    
    plt.scatter(x_i, y_i, color = "blue")
    plt.scatter (x_u, y_u, color = "red")
    plt.axis("equal")
    plt.title (f"n={n} med pi {pi_approx}")
    plt.show ()
    return pi_approx

# Exc2, approximation

def sphere_volume(n, d): #volymen av en sfär är radien*2 ^d 
    punkter = [[random. uniform (-1, 1) for _ in range (d)] for _ in range (n)]        
    sum_av_sqrt = lambda p: sum (koordinat**2 for koordinat in p)
    lista= list (map (sum_av_sqrt, punkter))
    inside = 0 
    for i in lista: 
        if i <= 1: 
            inside += 1 
    return (inside)/n * 2** d 


#Exc2, real value
def hypersphere_exact(n, d):
    return (m.pi ** (d/2)/ m.gamma(d/2 + 1))
    


#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    punkter = [[random. uniform (-1, 1) for _ in range (d)] for _ in range (n)]        
    sum_av_sqrt = lambda p: sum ([koordinat**2 for koordinat in p])
    lista= list (map (sum_av_sqrt, punkter))
    inside = 0 
    for i in lista: 
        if i <= 1: 
            inside += 1 
    return (inside)/n * 2** d
    

def _count_inside(n_sub, d):
        inside = 0
        for _ in range(n_sub):
            if sum(random.uniform(-1, 1)**2 for _ in range(d)) <= 1:
                inside += 1
        return inside
    
#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    n_sub = n // np
    tasks = [n_sub] * np
    
    with future.ProcessPoolExecutor(max_workers=np) as executor:
        results = executor.map(_count_inside, tasks, [d] * np)
    
    total_inside = sum (results)
    return (total_inside /n)* (2**d)
    

    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    for i in range (3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()
        print(f"Exc3 Naive python implementation nr: {i}: Sequential time of d={d} and n={n}: {stop-start}")
    
    print("What is numba time?")
    
    for i in range (3):
        start = pc()
        sphere_volume_numba(n, d)
        stop = pc()
        print(f"Exc3 numba run nr: {i}: Sequential time of d= {d} and n= {n}: {stop-start}")
    print('Subsequent numba calls are much faster')

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of d= {d} and n= {n}: {stop-start}")
    print("What is parallel time?")
    
    start = pc()
    sphere_volume_parallel(n, d)
    stop = pc()
    print(f"Exc4: parallel time of d= {d} and n= {n}: {stop-start}")
    #Det går mycket snabbare på linux
    
    

if __name__ == '__main__':
	main()
