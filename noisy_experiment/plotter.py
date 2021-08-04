import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def preparing_figure(fig_title, x_label, y_label):
    fig, ax = plt.subplots()
    font = {'fontname': 'Times New Roman'}
    ax.set_title(fig_title)

    # Подписи:
    ax.set_xlabel(x_label, **font)
    ax.set_ylabel(y_label, **font)

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


def fitting(x, y, deg, zero=False):
    if zero:
        def fit_poly_through_origin(x, y, n=1):
            a = x[:, np.newaxis] ** np.arange(1, n + 1)
            coeff = np.linalg.lstsq(a, y)[0]
            return np.concatenate(([0], coeff))

        z = fit_poly_through_origin(x, y, deg)

        p = np.polynomial.Polynomial(z)
    else:
        z = np.polyfit(x, y, deg)
        p = np.poly1d(z)
        t = 0
    return p


def plotting(sigma):
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

    title = 'Half-width of the distr. decrease'
    x_label = r'$\log(t_{sum})$'
    y_label = r'$\log(\sigma)$'
    fig, ax = preparing_figure(title, x_label, y_label)
    p = 0
    approx = True
    x = [math.log(each) for each in list(sigma.keys())]
    y = [math.log(each) for each in list(sigma.values())]
    x_p = np.linspace(min(x), max(x))
    if approx:
        p = fitting(x, y, 1)

        ax.plot(x_p, p(x_p), c=types_of_colors[0], ls='-', label=r'$\sigma(t_{sum})$')
        ax.plot(x, y, types_of_dots[2], c=types_of_colors[0])
    else:
        ax.plot(x, y, types_of_dots[2], c=types_of_colors[0], label=r'$\sigma(t_{sum})$')

    p_min = fitting([math.log(each) for each in list(sigma.keys())], [math.log(1/each) for each in list(sigma.keys())], 1)
    plt.plot(x_p, p_min(x_p) + p(x_p[0]) - p_min(x_p[0]), label=r'$\frac{1}{ t_{sum} }$')

    p_max = fitting([math.log(each) for each in list(sigma.keys())  ], [math.log((1/each) ** 0.5) for each in list(sigma.keys())], 1)
    plt.plot(x_p, p_max(x_p) + p(x_p[0]) - p_max(x_p[0]), label=r'$\frac{1}{ \sqrt{ t_{sum} } }$')

    plt.legend(loc='best')

    plt.show()

    #fig.savefig('sigma_' + '.png', dpi=500)
    #fig.savefig('files_to_send\\' + str(user_id) + '.pdf', dpi=500)

    plt.close()

    return (p_min, p, p_max)
