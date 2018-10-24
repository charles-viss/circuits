import numpy
import sympy 
import itertools


#naive approach for enumerating C(A,B)
#Input: A_eq and B_ineq are (m_a x n) and (m x n) numpy arrays
#Output: list of circuits in C(A,B) given by n-dimensional numpy arrays
def enumerate_circuits(B_ineq, A_eq=None):
    B = sympy.Matrix(B_ineq)
    m,n = B.shape
    r = 0
    if A_eq is not None:
        A = sympy.Matrix(A_eq)
        A, pivot_columns = A.rref()    #use reduced echelon form of A
        r = len(pivot_columns)         #r is the rank of A
           
    circuits = []
    for I in itertools.combinations(range(m),n-r-1):
        B_I = B[I,:]
        
        if A_eq is not None:
            D = A.col_join(B_I)
        else:
            D = B_I  
            
        ker_D = D.nullspace()
        if len(ker_D) == 1:   #circuit direction is found iff null space of D is one-dimensional
            g = numpy.array(ker_D[0]).flatten()
            g = g*sympy.lcm([g[i].q for i in range(n) if g[i] != 0]) #normalize to coprime integers
            
            g_is_duplicate = False
            for y in circuits:
                if numpy.array_equal(y, g):
                    g_is_duplicate = True
            if not g_is_duplicate:
                circuits.append(g)
                circuits.append(-1*g)
                
    return circuits
        