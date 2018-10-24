import numpy
import timeit

from polyhedron import Polyhedron


n = 7    #number of variables
m_B = 14 #number of inequality constraints

#randomly generate matrix for a full-dimensional polyhedron
B_values = numpy.random.randint(low=-10, high=10 ,size=n*m_B)
B = numpy.array(B_values).reshape(m_B, n)

#build polyhedron
P = Polyhedron(B=B, d=None, A=None, b=None)

#randomly generate a vector to determine sign-compatibility direction
u = numpy.random.randint(low=-10, high=10, size=n)
u = numpy.array(u).reshape(n)
B_u = B.dot(u)

print('\nB = ' + str(B))
print('sign: ' + str(B_u))

print('\ntesting naive method:')
start = timeit.default_timer()
circuits = P.naive_circuit_enumeration()
stop = timeit.default_timer()

#post-processing for finding sign-compatible circuits
sign_compatible_circuits = []
for g in circuits:
    B_g = B.dot(g)
    is_sign_compatible = False
    if all([B_u[i]*B_g[i] >= 0 for i in range(m_B)]):
        is_sign_compatible = True
        for i in range(m_B):
            if B_u[i]==0 and B_g[i] != 0:
                is_sign_compatible = False
    if(is_sign_compatible):
        sign_compatible_circuits.append(g)
    
print('number of sign-compatible circuits: ' + str(len(sign_compatible_circuits)))
print('runtime: ' + str(round(stop - start, 3)) + ' seconds')
print('sign-compatible circuits:')
for g in sign_compatible_circuits:
    print g

    
print('\ntesting polyhedral model method:')
start = timeit.default_timer()
sign_compatible_circuits = P.get_sign_compatible_circuits(u)
stop = timeit.default_timer()
print('number of sign-compatible circuits: ' + str(len(sign_compatible_circuits)))
print('runtime: ' + str(round(stop - start, 3)) + ' seconds')
print('sign-compatible circuits:')
for g in sign_compatible_circuits:
    print g