import numpy as np
import matplotlib.pyplot as plt

""" Programming Exercise 2.1 """
max_points = 10

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

def debugObservationProbability():
    ket1 = [-3-1j, -2j, 1j, 2]
    a = simpleQuantumSystem(ket1)
    print(a)

def debugTransitionCalc():
    ket1 = [1,-1j]
    ket2 = [1j,1] 
    v = simpleQuantumSystem(ket1, ket2)

    ket1 = [(2**.5)/2,1j*(2**.5)/2]
    ket2 = [1j*(2**.5)/2,-1*(2**.5)/2] 
    v = simpleQuantumSystem(ket1, ket2)
    print()

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

def debugVerifyHermitian():
    l1 = [[5,4+5j,6-16j], [4-5j,13,7],[6+16j,7,-2.1]]
    verifyHermitian(l1)

def debugObservableMeanValue():
    l1 = np.array([[1, -1j], [1j, 2]])
    ket1 = np.array([[2**.5/2, (2**.5/2)*1j]]).T
    a = observableMeanValue(l1,ket1)
    print(a) #a=2.5

    l1 = np.array([[-1, -1j], [1j, 1]])
    ket1 = np.array([[-1, -1-1j]]).T
    a = observableMeanValue(l1,ket1)
    print(a) #a==3

    l1 = np.array([[-1, -1j], [1j, 1]])
    ket1 = np.array([[.5, .5]]).T
    a = observableMeanValue(l1,ket1)
    print(a) #a==0

def debugoObservableVariance():
    l1 = np.array([[1, -1j], [1j, 2]])
    ket1 = np.array([[2**.5/2, (2**.5/2)*1j]]).T
    a = observableVariance(l1,ket1)
    print(a)

""" Programming Exercise 2.3 """
def observableEigenVectors(observable):
    _, eigenvectors = np.linalg.eigh(observable)
    return eigenvectors

def observableEigenValues(observable):
    eigenvalues, _ = np.linalg.eigh(observable)
    return eigenvalues

def eigenstatesProability(observable, ket):
    #return the probability that the state will transition to each of the eigenstates
    eigenstates = observableEigenVectors(observable)
    eigenvalues = observableEigenValues(observable)
    pairs = []
    for eigenvalue, eigenstate in zip(eigenvalues, eigenstates.T):
        #eigenstate should be a column vector
        eigenstate.shape = (eigenstate.size,1)
        probability = simpleQuantumSystem(ket, eigenstate)
        pairs.append((eigenvalue, probability))
    return pairs

def eigenvalueMeanValue(observable, ket):
    #calculates expected value of observable on a state
    #using eigenvalues and probabilities
    pairs = eigenstatesProability(observable, ket)
    mean = 0
    for i in pairs:
        mean += i[0] * i[1]
    return mean

def plotProbabilityDistribution(li):
    xk = [x[0] for x in li]
    pk = [y[1] for y in li]
    plt.plot(xk,pk)
    plt.savefig('books_read.png')

def debugObservableEigenValues():
    o = [[-1, -1j], [1j, 1]]
    e = observableEigenVectors(o)
    print(e)

def debugEigenstatesProability():
    l1 = np.array([[1, -1j], [1j, 2]])
    ket1 = np.array([[2**.5/2, (2**.5/2)*1j]]).T
    a = eigenvalueMeanValue(l1,ket1)
    b = observableMeanValue(l1,ket1)
    print(a==b) #a=2.5, calculates a=2.5
    
    l1 = np.array([[-1, -1j], [1j, 1]])
    ket1 = np.array([[-1, -1-1j]]).T
    a = eigenvalueMeanValue(l1,ket1)
    b = observableMeanValue(l1,ket1)
    print(a==b) #a==3, but calculates a=1

    l1 = np.array([[-1, -1j], [1j, 1]])
    ket1 = np.array([[.5, .5]]).T
    a = eigenvalueMeanValue(l1,ket1)
    b = observableMeanValue(l1,ket1)
    print(a==b) #a==0, calculates a=0
    
if __name__ == "__main__":
    debugEigenstatesProability()
    