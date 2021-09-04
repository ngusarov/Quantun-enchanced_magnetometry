import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
font = {'fontname': 'Times New Roman'}
ax.set_title(r'')

ax.minorticks_on()
ax.grid(which='major', axis='both')
ax.grid(which='minor', axis='both', linestyle=':')

# Подписи:
ax.set_xlabel(r'$\phi$, $radians$', **font)
ax.set_ylabel(r'$P_{|state>}(\phi)$', **font)

x = np.linspace(0, 1.7*3.1415)
ax.plot(
    x, (np.sin(x/2))**2, label=r'$P_{|0>}(\phi)$'
)
ax.plot(
    x, (np.cos(x/2))**2, label=r'$P_{|1>}(\phi)$'
)

plt.legend(loc="best")

plt.show()
plt.close()

