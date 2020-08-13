import numpy as np

""" Programming Exercise 2.1 """
max_points = 10

def simpleQuantumSystem(ket1, ket2=None):
    # kets are 1D lists of complex ampltiudes

    X = np.array(ket1)
    Y = np.array(ket2)

    numpoints = len(ket1) #input validation
    if (numpoints > max_points):
        raise Exception("Specified points exceeds maximum")

    if (Y.shape == ()): 
    # calculate the likelihood of finding the particle at a given point

        while (1): #input validation
            i = int(input(f"Input between 0 and {numpoints-1}: "))
            if i in range(0, numpoints):

                #equation (4) tells us the probability we will detect
                #particle at point x_i, norm square of c_i
                #divided by norm squared of |psi>
                return abs(X[i])**2 / (np.linalg.norm(X))**2 
               
   
    else:
        # calculate the probability of transitioning from the first ket 
        # to the second, after an observation

        if (len(ket2) != numpoints): #input validation
            raise Exception("Ket2 does not match size of Ket1")
        
        #equation (7) tells us how to calculate the transition amplitude
        #inner product (using complex conjugate) divided by product of norms
        amplitude = np.vdot(Y,X.T)/(np.linalg.norm(X) * np.linalg.norm(Y))

        #probability is norm squared because eigenvectors
        #constitute an orthognal basis of the state space
        return abs(amplitude)**2

def debugObservationProbability():
    ket1 = [-3-1j, -2j, 1j, 2]
    simpleQuantumSystem(ket1)

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
    bra = np.matmul(observable, np.transpose(ket))
    #I think maybe you dont take complex conjugate, and only take inner product
    #vdot takes the complex conjugate of the first param
    #return np.vdot(bra, np.transpose(ket))
    return np.dot(bra, np.transpose(ket))

def observableVariance(observable, ket):
    mean = observableMeanValue(observable,ket)
    #hermitian operator from equation (13)
    h_o = observable - mean * np.identity(len(observable))
    h_o_squared = np.matmul(h_o, h_o)

    #calculate variance using equation (15)
    #see example 4.2.6 from book
    return np.matmul(np.matmul(np.conj(ket), h_o_squared), np.transpose(ket))

def debugVerifyHermitian():
    l1 = [[5,4+5j,6-16j], [4-5j,13,7],[6+16j,7,-2.1]]
    verifyHermitian(l1)

def debugObservableMeanValue():
    l1 = [[1, -1j], [1j, 2]]
    ket1 = [2**.5/2, (2**.5/2)*1j]
    a = observableMeanValue(l1,ket1)
    print(a) #a=2.5

    l1 = [[-1, -1j], [1j, 1]]
    ket1 = [-1, -1-1j]
    a = observableMeanValue(l1,ket1)
    print(a) #a==3

    l1 = [[-1, -1j], [1j, 1]]
    ket1 = [.5, .5]
    a = observableMeanValue(l1,ket1)
    print(a) #a==0

def debugoObservableVariance():
    l1 = [[1, -1j], [1j, 2]]
    ket1 = [2**.5/2, (2**.5/2)*1j]
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
    mean = 0
    for eigenvalue, eigenstate in zip(eigenvalues, eigenstates):
        a = simpleQuantumSystem(ket, eigenstate)
        mean += a * eigenvalue
    return mean

def debugObservableEigenValues():
    o = [[-1, -1j], [1j, 1]]
    e = observableEigenVectors(o)
    print(e)

def debugEigenstatesProability():
    o = [[-1, -1j], [1j, 1]]
    k = [.5, .5]
    a = observableMeanValue(o,k)
    e = eigenstatesProability(o,k)

if __name__ == "__main__":
    debugObservableMeanValue()
    