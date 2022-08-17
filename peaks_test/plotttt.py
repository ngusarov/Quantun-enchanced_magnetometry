import math

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
font = {'fontname': 'Times New Roman'}
ax.set_title(r'')

ax.minorticks_on()

#x = np.linspace(0, 1.7*3.1415)

j00 = 1000
sigma = 300 #math.sqrt(j00)

def gaussian(x): #sigma  = sqrt(b), x_0 = b
    return 1 / (sigma * (2 * math.pi) ** 0.5) * math.exp(-(j00 - x) ** 2 / (2 * sigma ** 2))

plt.plot([each for each in range(1, 2*j00)], [gaussian(each) for each in range(1, 2*j00)])

plt.show()
fig.savefig("gaussian.pdf", dpi=500)
plt.close()

