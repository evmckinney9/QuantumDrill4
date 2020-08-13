import ProgrammingDrillCh4 as qs
import numpy as np
import matplotlib.pyplot as plt

""" Programming Exercise 2.1 """

def debugObservationProbability():
    ket1 = np.array([[-3-1j, -2j, 1j, 2]]).T

    a = qs.simpleQuantumSystem(ket1)
    print(a)

def debugTransitionCalc():
    ket1 = [1,-1j]
    ket2 = [1j,1] 
    v = qs.simpleQuantumSystem(ket1, ket2)

    ket1 = [(2**.5)/2,1j*(2**.5)/2]
    ket2 = [1j*(2**.5)/2,-1*(2**.5)/2] 
    v = qs.simpleQuantumSystem(ket1, ket2)
    print()

""" Programming Exercise 2.2 """

def debugVerifyHermitian():
    l1 = [[5,4+5j,6-16j], [4-5j,13,7],[6+16j,7,-2.1]]
    qs.verifyHermitian(l1)

def debugObservableMeanValue():
    l1 = np.array([[1, -1j], [1j, 2]])
    ket1 = np.array([[2**.5/2, (2**.5/2)*1j]]).T
    a = qs.observableMeanValue(l1,ket1)
    print(a) #a=2.5

    l1 = np.array([[-1, -1j], [1j, 1]])
    ket1 = np.array([[-1, -1-1j]]).T
    a = qs.observableMeanValue(l1,ket1)
    print(a) #a==3

    l1 = np.array([[-1, -1j], [1j, 1]])
    ket1 = np.array([[.5, .5]]).T
    a = qs.observableMeanValue(l1,ket1)
    print(a) #a==0

def debugoObservableVariance():
    l1 = np.array([[1, -1j], [1j, 2]])
    ket1 = np.array([[2**.5/2, (2**.5/2)*1j]]).T
    a = qs.observableVariance(l1,ket1)
    print(a)

""" Programming Exercise 3.2 """

def debugObservableEigenValues():
    o = [[-1, -1j], [1j, 1]]
    e = qs.observableEigenVectors(o)
    print(e)

def debugEigenstatesProability():
    l1 = np.array([[1, -1j], [1j, 2]])
    ket1 = np.array([[2**.5/2, (2**.5/2)*1j]]).T
    a = qs.eigenvalueMeanValue(l1,ket1)
    b = qs.observableMeanValue(l1,ket1)
    print(a==b) #a=2.5, calculates a=2.5
    
    l1 = np.array([[-1, -1j], [1j, 1]])
    ket1 = np.array([[-1, -1-1j]]).T
    a = qs.eigenvalueMeanValue(l1,ket1)
    b = qs.observableMeanValue(l1,ket1)
    print(a==b) #a==3, but calculates a=1

    l1 = np.array([[-1, -1j], [1j, 1]])
    ket1 = np.array([[.5, .5]]).T
    a = qs.eigenvalueMeanValue(l1,ket1)
    b = qs.observableMeanValue(l1,ket1)
    print(a==b) #a==0, calculates a=0

"""Programming Excercise 4.2"""

def debugDynamics():
    m = [[0,.7071067812,.7071067812,0],[.7071067812,0,0,-.7071067812],[.7071067812,0,0,.7071067812],[0,-.7071067812,.7071067812,0]]
    ket = np.array([[1,0,0,0]]).T
    ms = [m,m,m]
    k = qs.dynamicSimulation(ket, ms, 2)
    print(k)

"""Programming Excercise 5.2"""

if __name__ == "__main__":
    debugDynamics()
    