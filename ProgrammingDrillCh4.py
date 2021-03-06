import numpy as np
import matplotlib.pyplot as plt

""" Programming Exercise 2.1 """
max_points = 10
max_particles = 5

def simpleQuantumSystem(ket1, ket2=np.array(None)):

    numpoints = len(ket1) #input validation
    if (numpoints > max_points):
        raise Exception("Specified points exceeds maximum")

    if (ket2.shape == ()): 
    # calculate the likelihood of finding the particle at a given point

        while (1): #input validation
            i = int(input(f"Input between 0 and {numpoints-1}: "))
            if i in range(0, numpoints):

                #equation (4) tells us the probability we will detect
                #particle at point x_i, norm square of c_i
                #divided by norm squared of |psi>
                return abs(ket1[i])**2 / (np.linalg.norm(ket1))**2 
               
   
    else:
        # calculate the probability of transitioning from the first ket 
        # to the second, after an observation

        if (ket2.size != ket1.size): #input validation
            raise Exception("Ket2 does not match size of Ket1")
        
        #equation (7) tells us how to calculate the transition amplitude
        #inner product (using complex conjugate) divided by product of norms
        amplitude = np.vdot(ket2.T,ket1)/(np.linalg.norm(ket1) * np.linalg.norm(ket2))

        #probability is norm squared because eigenvectors
        #constitute an orthognal basis of the state space
        return abs(amplitude)**2


""" Programming Exercise 2.2 """
def square(sq):
    #checks if a matrix is square
    rows = len(sq)
    for row in sq:
        if len(row) != rows:
            return False
    return True

def verifyHermitian(matrix):
    if not square(matrix): #input validation
        raise Exception("Not a square matrix")

    X = np.matrix(matrix)
    #iff conjugate transpose equals itself
    return (X.getH() == X).all()

def observableMeanValue(observable, ket):
    #see example 4.2.5 from book

    #bra is row vector complex conjugate
    bra = np.matmul(observable, ket).T

    #vdot will take the complex conjugate of bra
    return np.vdot(bra, ket)


def observableVariance(observable, ket):
    mean = observableMeanValue(observable,ket)
    #hermitian operator from equation (13)
    h_o = observable - mean * np.identity(len(observable))
    h_o_squared = np.matmul(h_o, h_o)

    #calculate variance using equation (15)
    #see example 4.2.6 from book
    return np.matmul(np.matmul(np.conj(ket.T), h_o_squared), ket)

"""Programming Excercise 4.2"""
def dynamicSimulation(ket, matrix_sequence, timesteps=-1,):
    if timesteps == -1: #input validation
        timesteps = len(matrix_sequence)
    if timesteps > len(matrix_sequence) or timesteps < 0:
        raise Exception("invalid timesteps")

    if timesteps == 0:
        return ket
    
    unitary = matrix_sequence[1]
    for i in range(1,timesteps):
        unitary = np.matmul(unitary, matrix_sequence[i])
    return np.matmul(unitary,ket)

"""Programming Excercise 5.2"""
def multiQuantumSystem(*multi_stateSpace):
    #returns a tensor product of the vector spaces
    num_particles = len(multi_stateSpace)
    if num_particles <= 0 or num_particles > max_particles:
        raise Exception("invalid number of particles")
    
    tensor_product = multi_stateSpace[0]

    for i in range(1, num_particles):
        tensor_product = np.kron(tensor_product, multi_stateSpace[i])

    return tensor_product
    