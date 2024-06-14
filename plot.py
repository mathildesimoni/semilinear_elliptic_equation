import numpy as np
import matplotlib.pyplot as plt

fixed_point = [12, 353]
anderson = [12, 24, 25]
newton = [5, 7, 8]
alpha = [0.1, 2.0, 5]


plt.semilogy(alpha[:2], fixed_point, color='lightsteelblue', label='Fixed-point iteration')
plt.semilogy(alpha, anderson, color='cornflowerblue', label='Anderson acceleration')
plt.semilogy(alpha, newton, color='royalblue', label="Newton's method")

texts =[str(fixed_point[0]) +' iter', 
        str(fixed_point[1]) +' iter', 
        str(anderson[1]) +' iter', 
        str(anderson[2]) + ' iter', 
        str(newton[0]) + ' iter',
        str(newton[1]) + ' iter', 
        str(newton[2]) + ' iter'
        ]

locs = [(alpha[0], fixed_point[0]), 
        (alpha[1], fixed_point[1]),
        (alpha[1], anderson[1]), 
        (alpha[2], anderson[2]),
        (alpha[0], newton[0]),
        (alpha[1], newton[1]),
        (alpha[2], newton[2])
        ]

for text, loc in zip(texts, locs):
    plt.annotate(text, loc)

plt.grid(True)
plt.ylabel('Number of iterations')
plt.xlabel(r'Value for $\alpha$')
plt.legend()

plt.savefig('iterations.svg',format='svg', bbox_inches='tight')

plt.show()