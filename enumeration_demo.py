import numpy
import timeit
from scipy import special

from polyhedron import Polyhedron

n = 6    #number of variables
m_B = 8  #number of inequality constraints
m_A = 2  #number of equality constraints

#randomly generate matrices for a polyhedron
B_values = numpy.random.randint(low=-10, high=10, size=n*m_B)
B = numpy.array(B_values).reshape(m_B, n)
A = None
if(m_A > 0):
    A_values = numpy.random.randint(low=-10, high=10, size=n*m_A)
    A = numpy.array(A_values).reshape(m_A, n)

#build polyhedron (right-hand side vectors d and b do not affect set of circuits)
P = Polyhedron(B=B, d=None, A=A, b=None)
print('enumerating set of circuits C(A,B) for randomly generated A and B: ')
print('A = ' + str(A))
print('B = ' + str(B))
print('\ntesting naive method:')
start = timeit.default_timer()
circuits = P.naive_circuit_enumeration()
stop = timeit.default_timer()
print('number of circuits: ' + str(len(circuits)))
print('runtime: ' + str(round(stop - start, 3)) + ' seconds')
print('circuits:')
for g in circuits:
    print g


print('\ntesting standard form method:')
start = timeit.default_timer()
circuits, standard_circuits = P.get_standard_form_circuits()
stop = timeit.default_timer()
print('number of circuits: ' + str(len(circuits)))
print('runtime: ' + str(round(stop - start, 3)) + ' seconds')
#print('circuits:')
#for g in circuits:
#    print g
    
print('\ntesting polyhedral model method:')
start = timeit.default_timer()
circuits = P.polyhedral_model_circuit_enumeration()
stop = timeit.default_timer()
print('number of circuits: ' + str(len(circuits)))
print('runtime: ' + str(round(stop - start, 3)) + ' seconds')
#print('circuits:')
#for g in circuits:
#    print g

r = 0
if A is not None:
    r = numpy.linalg.matrix_rank(numpy.matrix(A))
print('\nmax number of circuits of P: ' + str(int(2*special.binom(m_B,n-r-1))))
print('actual number of circuits of P: ' + str(len(circuits)))

sum = 0
for d in range(r+1,n+1):
    sum += 2*special.binom(n,d)*special.binom(m_B,d-r-1)
print('\nmax number of standard form circuits of P: ' + str(int(sum)))
print('actual number of standard form circuits of P: ' + str(len(standard_circuits)))



#set parameters for dual transportation problem.
#assume underlying graph is complete bipartite with partite set sizes m and n
m = 3
n = 3
print('\n\ndual transportation polytope demo with m = ' + str(m)
         + ' and n = ' + str(n) + ':')

A = numpy.zeros((1,m+n), dtype=int)
A[0,0] = 1

B = numpy.zeros((1,m+n), dtype=int)
for i in range(m):
    for j in range(n):
        row = numpy.zeros(m+n, dtype=int)
        row[i] = -1
        row[m+j] = 1
        B = numpy.concatenate((B, row.reshape((1, m + n))))
B = numpy.delete(B, 0, 0)
        
#build polyhedron (right-hand side vectors d and b do not affect set of circuits)
P = Polyhedron(B=B, d=None, A=A, b=None)

print('\ntesting naive method:')
start = timeit.default_timer()
circuits = P.naive_circuit_enumeration()
stop = timeit.default_timer()
print('number of circuits: ' + str(len(circuits)))
print('runtime: ' + str(round(stop - start, 3)) + ' seconds')
print('circuits:')
for g in circuits:
    print g

print('\ntesting standard form method:')
start = timeit.default_timer()
circuits, standard_circuits = P.get_standard_form_circuits()
stop = timeit.default_timer()
print('number of circuits: ' + str(len(circuits)))
print('number of standard form circuits: ' + str(len(standard_circuits)))
print('runtime: ' + str(round(stop - start, 3)) + ' seconds')

print('\ntesting polyhedral model method:')
start = timeit.default_timer()
circuits = P.polyhedral_model_circuit_enumeration()
stop = timeit.default_timer()
print('number of circuits: ' + str(len(circuits)))
print('runtime: ' + str(round(stop - start, 3)) + ' seconds')