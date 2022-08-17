from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

size = [2, 2, 2, 2, 2]
max_error = [0.00457, 0.005, 0.06, 0.39, 0.66]
time = [21*60+13, 19*60+34, 10*60+2, 5*60+31, 3*60+52]
epsilon = [0.005, 0.01, 0.1, 0.5, 1]

types_of_dots = [
        '.',
        'o',
        'x',
        'v',
        '>'
    ]
types_of_colors = [
        'r',
        'b',
        'g',
        'v',
        'black'
]

def preparing_figure(fig_title, x_label, y_label):
    fig, ax = plt.subplots()
    font = {'fontname': 'Times New Roman'}
    ax.set_title(fig_title, fontsize=20)

    # Подписи:
    ax.set_xlabel(x_label, **font, fontsize=25)
    ax.set_ylabel(y_label, **font, fontsize=25)

    # Сетка:
    ax.minorticks_on()
    ax.grid(which='major', axis='both')
    ax.grid(which='minor', axis='both', linestyle=':')

    fig.set_figheight(7)
    fig.set_figwidth(10)

    # Оси:
    #plt.yscale('log')
    #plt.xscale('log')

    # Легенда:
    # matplotlib.rcParams["legend.framealpha"] = 1
    return fig, ax

x = epsilon#[math.log(each) for each in N]
y = time


title = ''
x_label = 'Final precision $\epsilon$'
y_label = r'Time of performance t, [sec]'
fig, ax = preparing_figure(title, x_label, y_label)

font_2 = font_manager.FontProperties(family='Times New Roman',
                                         size=25)
ax.plot(x, y, types_of_dots[1], c=types_of_colors[0], ls='-', label='experiment')
bb = 90
ax.plot(x, [bb*(np.log(1/eps))**1.5+230 for eps in x], types_of_dots[1], c=types_of_colors[1], ls='-', label=r'theory $\sim  log^{1,5}(1/\epsilon)$')

ax.legend(prop=font_2)
plt.xlim(1.05, -0.05)
plt.show()
fig.savefig('Cheb_eps_res.pdf', dpi=500)
