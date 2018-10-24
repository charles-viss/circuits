import numpy
from scipy import optimize

import steepest_descent as sd
from polyhedron import Polyhedron

#demo of steepest-descent augmentation scheme on small examples in R^3
#A = numpy.array([[1,1,0]])
#b = numpy.array([0])
A = None
b = None

B = numpy.array([
        [-1, 0, 0],
        [0, 0, -1],
        [0, -1, -1],
        [0, -1, 0],
        [2, 1, 1],
        [3, 3, -1]])

d = numpy.array([
        0,
        0,
        2,
        4,
        6,
        8])

m_B, n = B.shape
m_A = 0
#m_A = A.shape[0]

c = numpy.array([-8, -1, -5])

P = Polyhedron(B=B, d=d, A=A, b=b)

x_initial = numpy.array([0, 0, 0])

print('\nstrictly feasible circuits:')
circuits = P.get_strictly_feasible_circuits(x=x_initial)
for g in circuits:
    print g

g, steepness = P.get_steepest_descent_circuit([0, 0, 0], c=c)

print('\nobjective f: ')
print(c)
print('steepest-descent circuit:')
print(g)
print('steepness:')
print(steepness)
    
result = sd.steepest_descent_augmentation_algorithm(P, c, x_initial)
x_optimal = result.x

print('\nsolution using steepest-descent augmentation:')
print(result)
for i in range(len(result.circuits)):
    print('g_' + str(i) + ': ' + str(result.circuits[i].T) + ' with alpha_'
              + str(i) + ' = ' + str(result.steps[i]))

print('\noptimal solution using direct linear programming:')
result = optimize.linprog(c=c, A_eq=A, b_eq=b, A_ub=B, b_ub=d, bounds=(None,None), method='simplex')
print(result.x)

circuits, lambdas = P.get_sign_compatible_sum(x_1=x_initial, x_2=x_optimal, c=c)
print('\nf-optimal sign-compatible walk from x_initial (' + str(x_initial) 
            + ') to x_optimal:')
for i in range(len(circuits)):
    print('g_' + str(i) + ': ' + str(circuits[i].T) + ' with lambda_' + str(i) + ' = ' + str(lambdas[i]))
