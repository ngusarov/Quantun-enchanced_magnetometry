import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import font_manager


def preparing_figure(fig_title, x_label, y_label):
    fig, ax = plt.subplots()
    font = {'fontname': 'Times New Roman'}
    #ax.set_title(fig_title, **font, fontsize=30)

    # Подписи:
    ax.set_xlabel(x_label, **font, fontsize=52)
    ax.set_ylabel(y_label, **font, fontsize=55)
    plt.tick_params(axis='both', which='major', labelsize=44)

    # Сетка:
    ax.minorticks_on()
    ax.grid(which='major', axis='both')
    ax.grid(which='minor', axis='both', linestyle=':')

    fig.set_figheight(13)
    fig.set_figwidth(19)

    # Оси:
    #plt.yscale('log')
    #plt.xscale('log')

    # Легенда:
    # matplotlib.rcParams["legend.framealpha"] = 1
    return fig, ax


dependence = 'time of k-th step $t_{k}$, $\mu s$'

title = 'a)'
x_label = r'Total delay time $t_{sum}$, $ms$'
y_label = r'Half-width of PDD $\sigma$, $nT$'
fig, ax = preparing_figure(title, x_label, y_label)
p = 0
approx = False


noiseless_sigma = {1.8396787012671877e-06: 11, 3.6793574025343754e-06: 10, 7.358714805068751e-06: 7, 1.4717429610137502e-05: 4.0, 2.9434859220275004e-05: 2.0, 5.886971844055001e-05: 1.0, 0.00011773943688110001: 0.5, 0.00023547887376220003: 0.25, 0.00047095774752440006: 0.125, 0.0009419154950488001: 0.0625, 0.0018838309900976002: 0.03125, 0.0037676619801952005: 0.015625, 0.007535323960390401: 0.0078125, 0.015070647920780802: 0.00390625}
single_qubit_sigma = {1.8396787012671877e-06: 11.0, 3.6793574025343754e-06: 10.0, 7.358714805068751e-06: 7.0, 1.4717429610137502e-05: 4.0, 2.9434859220275004e-05: 2.0, 5.886971844055001e-05: 1.125, 0.00011773943688110001: 0.5, 0.00023547887376220003: 0.25, 0.00047095774752440006: 0.125, 0.0009419154950488001: 0.0625, 0.0018838309900976002: 0.03125, 0.0037676619801952005: 0.01953125, 0.007535323960390401: 0.009765625, 0.015070647920780802: 0.005859375}
q2_sigma = {9.422744567466083e-07: 11.0, 1.8845489134932167e-06: 10.0, 3.7690978269864334e-06: 7.0, 7.538195653972867e-06: 4.0, 1.5076391307945734e-05: 2.0, 3.0152782615891467e-05: 1.225, 6.0305565231782934e-05: 0.5, 0.00012061113046356587: 0.25, 0.00024122226092713174: 0.125, 0.0004824445218542635: 0.0625, 0.000964889043708527: 0.03125, 0.001929778087417054: 0.01953125, 0.003859556174834108: 0.009765625, 0.007719112349668216: 0.0068359375}
q3_sigma = {6.281829711644055e-07: 11.0, 1.256365942328811e-06: 10.0, 2.512731884657622e-06: 7.0, 5.025463769315244e-06: 4.0, 1.0050927538630488e-05: 2.0, 2.0101855077260977e-05: 1.125, 4.0203710154521954e-05: 0.5, 8.040742030904391e-05: 0.25, 0.00016081484061808782: 0.125, 0.00032162968123617563: 0.0625, 0.0006432593624723513: 0.03125, 0.0012865187249447025: 0.01953125, 0.002573037449889405: 0.009765625, 0.00514607489977881: 0.005859375}
q4_sigma = {4.935723344863186e-07: 11.0, 9.871446689726372e-07: 10.0, 1.9742893379452744e-06: 7.0, 3.948578675890549e-06: 4.0, 7.897157351781098e-06: 2.0, 1.5794314703562195e-05: 1.0, 3.158862940712439e-05: 0.5, 6.317725881424878e-05: 0.25, 0.00012635451762849756: 0.125, 0.00025270903525699513: 0.0625, 0.0005054180705139903: 0.0390625, 0.0010108361410279805: 0.01953125, 0.002021672282055961: 0.01171875, 0.004043344564111922: 0.0087890625}
q5_sigma = {4.038319100342607e-07: 11.0, 8.076638200685214e-07: 10.0, 1.6153276401370429e-06: 7.0, 3.2306552802740857e-06: 4.0, 6.461310560548171e-06: 2.0, 1.2922621121096343e-05: 1.125, 2.5845242242192686e-05: 0.5, 5.169048448438537e-05: 0.25, 0.00010338096896877074: 0.125, 0.00020676193793754149: 0.0625, 0.00041352387587508297: 0.03125, 0.0008270477517501659: 0.015625, 0.0016540955035003319: 0.0078125, 0.0033081910070006638: 0.00390625}

lw = 5
markersize = 15

sigma = noiseless_sigma
x = [each*10**(3) for each in list(sigma.keys())]
y = [each for each in list(sigma.values())]
ax.plot(x, y, '.', c='#EE0000',ls='-', label=r'1 qub. simulator', lw=lw, markersize=markersize)

sigma = single_qubit_sigma
x = [each*10**(3) for each in list(sigma.keys())]
y = [each for each in list(sigma.values())]
ax.plot(x, y, 'd', c='#EE00BB',ls='-', label=r'1 qubit', lw=lw, markersize=markersize)

sigma = q2_sigma

x = [each*10**(3) for each in list(sigma.keys())]
y = [each for each in list(sigma.values())]
ax.plot(x, y, 'o', c='#2800EE',ls='-', label=r'2 qubits', lw=lw, markersize=markersize)

sigma = q3_sigma

x = [each*10**(3) for each in list(sigma.keys())]
y = [each for each in list(sigma.values())]
ax.plot(x, y, '^', c='#00E3EE',ls='-', label=r'3 qubits', lw=lw, markersize=markersize)

sigma = q4_sigma

x = [each*10**(3) for each in list(sigma.keys())]
y = [each for each in list(sigma.values())]
ax.plot(x, y, 's', c='#00EE33',ls='-', label=r'4 qubits', lw=lw, markersize=markersize)

sigma = q5_sigma

x = [each*10**(3) for each in list(sigma.keys())]
y = [each for each in list(sigma.values())]
ax.plot(x, y, 'p', c='#EEB500',ls='-', label=r'5 qubits', lw=lw, markersize=markersize)

plt.plot([each*10**(3) for each in list(sigma.keys())], [1/each*10**(-5) for each in list(sigma.keys())], label=r'HL', lw=lw)
plt.plot([each *10**(3) for each in list(sigma.keys())], [(1/each) ** 0.5*10**(-1.7) for each in list(sigma.keys())], label=r'SQL', lw=lw)


font_2 = font_manager.FontProperties(family='Times New Roman', size=32)

plt.legend(loc='best', prop=font_2)

plt.xscale('log')
plt.yscale('log')

plt.show()
fig.savefig('materials_for_article\\sigma.pdf', dpi=500)
#fig.savefig('files_to_send\\' + str(user_id) + '.pdf', dpi=500)

#plt.close()