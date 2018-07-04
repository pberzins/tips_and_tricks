import math
import scipy.stats as scs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

coke = np.loadtxt('data/coke_weights.txt')

def compute_power(n, sigma, alpha, mu0, mua):
    standard_error = sigma/n**.5
    h0 = scs.norm(mu0, standard_error)
    ha = scs.norm(mua, standard_error)
    critical_value = h0.ppf(1-alpha)
    power = 1 - ha.cdf(critical_value)
    return power

def plot_pdf(data_array,alpha, effect_size = 0, error_range = 4):
    """ Takes in an array
    """

    data_mean = data_array.mean()
    data_stdev = data_array.std()
    data_std_error = data_stdev/(np.sqrt(len(coke)))

    errors_to_plot = np.linspace((-error_range*data_std_error)+data_mean,
                                 (error_range*data_std_error)+data_mean+.1, num= 300)

    null_distribution = scs.norm(data_mean,data_std_error)
    alternate_distribution = scs.norm(data_mean+0.1,data_std_error)

    critical_value = null_distribution.ppf(1-alpha)
    power = 1- alternate_distribution.cdf(critical_value)

    alt_pdf_list = []
    null_pdf_list =[]
    error_list = []
    for e in errors_to_plot:
        null_pdf_list.append(null_distribution.pdf(e))
        alt_pdf_list.append(alternate_distribution.pdf(e))
        error_list.append(e)
    return null_pdf_list, alt_pdf_list, error_list, critical_value, power


#shaded = error_list[np.array(error_list) >= critical_value]
fig,axs = plt.subplots(2,1, figsize=(10,8))
alpha_to_plot= [0.05,0.1]
for alpha, ax in zip(alpha_to_plot, axs.flatten()):
    null_pdf_list, alt_pdf_list, error_list, critical_value, power = plot_pdf(coke, alpha)
    ax.plot(error_list, null_pdf_list, label='Null', linewidth=3)
    ax.plot(error_list, alt_pdf_list, label='Alternate', linewidth=3)
    ax.set_xlabel('Weight in Ounces')
    ax.set_ylabel('Relative Probability')
    ax.set_title(f'Alpha:{alpha}, Power: {power*100:.1f}%')
    ax.axvline(x=critical_value, color='black', label=f'Critical Value: {critical_value:.2f}')
    ax.fill_between(error_list, alt_pdf_list,
                    where=(error_list >= critical_value),
                    color= 'orange',alpha=.3,label=f'Power: {power*100:.1f}%')
    ax.fill_between(error_list, null_pdf_list,
                   where=(error_list>= critical_value),
                    color= 'blue', alpha=.3, label='False Positive (Alpha)')
    ax.fill_between(error_list, alt_pdf_list,
                   where=(error_list<= critical_value),
                    color= 'green', alpha=.3, label='False Negative (Beta)')
    ax.set_ylim(0,5)
    fig.tight_layout()

    ax.legend()
