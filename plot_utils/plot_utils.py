import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt

# colors used for the plots
bcolor = sb.color_palette('husl')[-2]
gcolor = sb.color_palette('husl')[2]
rcolor = sb.color_palette('husl')[0]
ocolor = sb.color_palette('muted')[1]
pcolor = [0.34901961, 0.34901961, 0.34901961, 1.]

from surveying_llms.load_responses import col2freq

# compute the entropy of the uniform distribution and of the census responses
acs_categories = {  # for each survey question
    'SEX': 2, 'AGER': 7, 'HISPR': 2, 'RAC1PR': 6, 'NATIVITY': 2, 'CIT': 5,
    'SCH': 3, 'SCHLR': 5, 'LANX': 2, 'ENG': 4,
    'HICOV': 2, 'DEAR': 2, 'DEYE': 2,
    'MAR': 5, 'FER': 2, 'GCL': 2, 'MIL': 4,
    'WRK': 2, 'ESR': 6, 'JWTRNS': 12, 'WKL': 3, 'WKWN': 5, 'WKHPR': 6, 'COWR': 9, 'PINCPR': 5
}

assign_nans = {v: None for v in acs_categories.keys()}
assign_nans[
    'FER'] = 2  # assign "less than 15 years/greater than 50 years/male" to "did not give birth in last 12 months"

# functions to compute entropy and KL divergence
def compute_entropy(p):
    pnz = p[p!=0]
    return - np.sum(pnz * np.log2(pnz))

def compute_kl(p, q):
    nz = p != 0
    pnz = p[nz]
    qnz = q[nz]
    return np.sum(pnz * np.log2(pnz / qnz))


def load_acs_census_responses(census_file, n_categories):
    """ Loads the raw PUMS obtained from `process_acs.py`, and returns the census responses """
    census_data = pd.read_csv(census_file)

    variables = n_categories.keys()

    # obtain the census responses from the raw data
    census_responses = {var: col2freq(census_data[var], n_categories[var],
                                      assign_nan=assign_nans[var], weight=census_data['PWGTP'])
                        for var in variables}

    return census_responses


def load_acs_state_responses(census_file, n_categories):
    print("Loading ACS responses by state...")
    census_data = pd.read_csv(census_file)
    states = set(census_data['ST'].unique()) - {72}  # remove Puerto Rico

    variables = n_categories.keys()
    responses_states = {var: {} for var in variables}
    for state in tqdm(states):
        state_data = census_data[census_data['ST'] == state]
        for var in variables:
            responses_states[var][state] = col2freq(state_data[var], n_categories[var],
                                                    assign_nan=assign_nans[var], weight=state_data['PWGTP'])
    return states, responses_states


def figure_1(model_entropies, census_norm_entropies, base_models, model_sizes):
    _, ax = plt.subplots(1, 3, figsize=(5.5, 1.5), sharey=True)

    u = census_norm_entropies['SEX']
    ax[0].plot([1e7, 1e12], [u - 0.015, u - 0.015], c=gcolor)  # -0.015 to avoid overlap with uniform
    u = census_norm_entropies['HICOV']
    ax[1].plot([1e7, 1e12], [u, u], c=gcolor)
    u = census_norm_entropies['FER']
    ax[2].plot([1e7, 1e12], [u, u], c=gcolor)

    msize = 4

    plot_vars = ['SEX', 'HICOV', 'FER']
    sizes = [model_sizes[m] for m in base_models]
    for i in range(3):
        entropies = [model_entropies[plot_vars[i]][m] for m in base_models]
        ax[i].plot(sizes, entropies, 'D', c=bcolor, markersize=msize, zorder=10)
        ax[i].plot([1e7, 1e12], [1, 1], c=ocolor)  # uniform line
        ax[i].set_xscale('log')
        ax[i].set_ylim([0, 1.05])
        ax[i].set_xlim([8e7, 3e11])
        ax[i].set_xlabel('Model size', fontsize=11)
        ax[i].set_xticks([1e8, 1e9, 1e10, 1e11])
        ax[i].tick_params(axis='x', pad=0)
        ax[i].grid()
        ax[i].xaxis.set_ticks_position('none')
        ax[i].yaxis.set_ticks_position('none')
        #     for spine in ax[i].spines.values():
        #         spine.set_edgecolor('0.7')
        for spine in ax[i].spines.values():
            spine.set_edgecolor('0.3')

    # ax[0].set_ylabel('Entropy of\nmodel rresponses', fontsize=12)
    ax[0].set_ylabel('Response entropy', fontsize=12, labelpad=5, y=0.5)

    ax[0].set_title('SEX')
    ax[1].set_title('HICOV')
    ax[2].set_title('FER')

    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='D', color=bcolor,
                                    label='Model', markersize=msize + 1,
                                    markerfacecolor=bcolor, lw=0))
    legend_elements.append(plt.Line2D([0], [0], color=gcolor,
                                    label='U.S. census', markersize=msize + 1,
                                    markerfacecolor=gcolor, lw=2))
    legend_elements.append(plt.Line2D([0], [0], color=ocolor,
                                    label='Uniform distribution', markersize=msize + 1,
                                    markerfacecolor=ocolor, lw=2))
    legend_position = (-0.64, -0.38)
    ax[-1].legend(handles=legend_elements, loc='upper center',
                bbox_to_anchor=legend_position, frameon=False, ncols=3, 
    #               handletextpad=0.8,
                fontsize=10.5, columnspacing=1.1,)

    plt.subplots_adjust(wspace=0.1)

def plot_a_bias(a_bias, base_models, model_names, alpha=0.4, figsize=(9, 1.5)):
    fig, ax = plt.subplots(figsize=figsize)

    # plot the a-bias of each model
    for i, model in enumerate(base_models):
        ys = a_bias[model]
        xs = [i for _ in ys]
        ax.plot(xs, ys, '.', c=bcolor, alpha=alpha, markersize=12)

    # legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='.', color=bcolor,
                                      label='Survey question', markersize=11,
                                      markerfacecolor=bcolor, lw=0))
    legend_position = (1.0, 1.2)  # Coordinates (x, y) for top left position
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=legend_position, frameon=False, ncols=3, handletextpad=.6, )

    # x-axis
    dx = 0.2
    dy = 0.05
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax.set_xticks([i for i in range(len(base_models))])
    labels = [model_names[m] for m in base_models]
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_xlim([-0.4, len(base_models) - 0.5])

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # y-axis
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.set_axisbelow(True)
    ax.set_ylabel('A-bias', fontsize=12)
    ax.tick_params(axis='y', which='both', labelsize=10)

def plot_adjusted_entropy(entropies, census_entropies, models, variables, model_names, figsize, 
                          alpha=0.4, ylegend=1.2):
    fig, ax = plt.subplots(figsize=figsize)

    # plot the entropy of each model
    for i, model in enumerate(models):
        xs = [i for _ in variables]
        ys = [entropies[v][model] for v in variables]
        ax.plot(xs, ys, '.', c=bcolor, alpha=alpha, markersize=14)

    # plot the census entropy
    if census_entropies is not None:
        xs = [len(models) for _ in variables]
        ys = census_entropies.values()
        ax.plot(xs, ys, '.', c=gcolor, alpha=alpha, markersize=14)

    # legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='.', color=bcolor,
                                      label='', markersize=14,
                                      markerfacecolor=bcolor, lw=0))
    legend_elements.append(plt.Line2D([0], [0], marker='.', color=gcolor,
                                      label='Survey question', markersize=14,
                                      markerfacecolor=gcolor, lw=0))
#     legend_position = (1.015, 1.3)  # Coordinates (x, y) for top left position
    legend_position = (1.015, ylegend)  # Coordinates (x, y) for top left position
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=legend_position, frameon=False, handletextpad=-0.1, ncols=2, columnspacing=-0.7)

    # x-ticks
    dx = 0.2;
    dy = 0.05
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    labels = [model_names[m] for m in models]
    if census_entropies is not None:
        labels.append('U.S. Census')
    ax.set_xticks([i for i in range(len(labels))])
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlim([-0.4, len(labels) - 0.5])

    ax.yaxis.set_ticks_position('none')

    # ax.grid('x', zorder=-10)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.set_axisbelow(True)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Response entropy', fontsize=12, labelpad=5, y=0.5)

def plot_divergence_subgroups(divergence, divergence_uniform, divergence_census,
                              selected_models, model_names, alpha=0.1,
                              ylim=[-0.1, 2.], yticks=[0., 0.5, 1., 1.5, 2.],
                              ylabel=r'$\bar\mathrm{KL}$(model, Ref.)', clabel='Census subgroup',
                              figsize=(13, 1.5)):
    fig, ax = plt.subplots(figsize=figsize)

    # divergence for each model
    for i, model in enumerate(selected_models):
        ys = list(divergence[model].values())
        xs = [i for _ in ys]
        ax.plot(xs, ys, '.', c=bcolor, alpha=alpha, markersize=12)
        ax.plot(i, divergence_uniform[model], '*', c=rcolor, markersize=9)
        
        # plot the mean similarity
        ax.scatter(i, divergence_census[model],marker='x', color='k', s=50, linewidth=2, zorder=100)
        

    # legend
#     legend_elements = []
#     legend_elements.append(plt.Line2D([0], [0], marker='o', color=bcolor,
#                                       label='U.S. state populations', markersize=6,
#                                       markerfacecolor=bcolor, lw=0))
#     legend_elements.append(plt.Line2D([0], [0], marker='x', color='k',
#                                       label='Entire U.S. census', markersize=7,
#                                       markerfacecolor='k', lw=0))
#     legend_elements.append(plt.Line2D([0], [0], marker='*', color=rcolor,
#                                       label='Uniform responses', markersize=9,
#                                       markerfacecolor=rcolor, lw=0))
#     legend_position = (1.01, 1.32)  # Coordinates (x, y) for top left position
#     ax.legend(handles=legend_elements, loc='upper right',
#               bbox_to_anchor=legend_position, frameon=False, handletextpad=0.4, ncols=4, fontsize=10.5)

    # x-ticks
    dx = 0.2;
    dy = 0.05
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax.set_xticks([i for i in range(len(selected_models))])
    labels = [model_names[m] for m in selected_models]
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)

    ax.set_xlim([-0.4, len(selected_models) - 0.5])
#     ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='y', which='both', labelsize=10)


def plot_discriminator(accuracies, accuracies2, models, model_names, 
                       title="Accuracy in discriminating model-generated data", 
                       figsize=(10.5, 1.2), legendx=1.0,
                       mean_s=None, upper_s=None, lower_s=None):
    fig, ax = plt.subplots(figsize=figsize)
    
    xs = [i - 0.15 for i, _ in enumerate(models)]
    ys = [np.mean(accuracies[model]) * 100 for model in models]
    es = [np.std(accuracies[model]) * 2 * 100 for model in models]
    print(es)
    plt.plot(xs, ys, 'X', color=bcolor, alpha=0.9)
    plt.errorbar(xs, ys, yerr=es, color=bcolor, fmt='.', capsize=2)
    
    if accuracies2 is not None:
        xs = [i + 0.15 for i, _ in enumerate(models)]
        ys = [np.mean(accuracies2[model]) * 100 for model in models]
        es = [np.std(accuracies2[model]) * 2 * 100 for model in models]
        plt.plot(xs, ys, 'X', color=ocolor, alpha=0.9)
        plt.errorbar(xs, ys, yerr=es, color=ocolor, fmt='.', capsize=2)
    
    # plot mean across states
    if mean_s is not None:
        plt.plot([-1, len(models)], [mean_s, mean_s], c=gcolor)
        plt.fill_between([-1, len(models)], [lower_s, lower_s], [upper_s, upper_s], alpha=0.3, color=gcolor)

    dx = 0.2;
    dy = 0.05
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax.set_xticks([i for i in range(len(models))])
    labels = [model_names[m] for m in models]
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)
    ax.set_yticks([50, 60, 70, 80, 90, 100])

    ax.set_xlim([-0.5, len(models) - 0.5])
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=100)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.set_axisbelow(True)
    ax.set_ylim([97.5, 100.2])
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.tick_params(axis='y', which='both', labelsize=10.5)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    legend_elements = []

    if accuracies2 is not None:
        legend_elements.append(plt.Line2D([0], [0], marker='X', color=bcolor,
                                        label='With choice randomization', markersize=6,
                                        markerfacecolor=bcolor, lw=0))
        legend_elements.append(plt.Line2D([0], [0], marker='X', color=ocolor,
                                        label='Without choice randomization', markersize=6,
                                        markerfacecolor=ocolor, lw=0))
        legend_position = (legendx, 1.41)  # Coordinates (x, y) for top left position
        l1 = plt.legend(handles=legend_elements, loc='upper right',
                bbox_to_anchor=legend_position, frameon=False, ncols=2, handletextpad=0, columnspacing=0.7)
    
    if mean_s is not None:
        legend_elements = []
        legend_elements.append(plt.Line2D([0], [0], color=gcolor, 
            label='Discriminating between any U.S. state and rest of ACS census',
                        markersize=6, markerfacecolor=gcolor, lw=2))
        legend_position = (legendx+0.02, 1.26)  # Coordinates (x, y) for top left position
        l2 = plt.legend(handles=legend_elements, loc='upper right',
                bbox_to_anchor=legend_position, frameon=False, ncols=2, columnspacing=0.7)
    
    if accuracies2 is not None:
        ax.add_artist(l1)

    if mean_s is not None:
        ax.add_artist(l2)
    
    ax.set_title(title, fontsize=11, y=1.32)


def plot_similarity_opinions(states, divergence_unadj, divergence, subgroup_ent, models, model_names,
                             xlabel='Entropy of subgroup\'s (U.S. state) responses', 
                             ylabel=r'$\bar{\mathrm{KL}}$(model, subgroup)',
                             title1='Unadjusted responses', title2='Adjusted responses',
                             figsize=(5.5, 2.5)):
    from matplotlib.legend_handler import HandlerTuple, HandlerLine2D
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)

    ax = axs[0]
    p = sb.color_palette("colorblind")

    # xaxis = list([subgroup_unif_mean[s] for s in states])
    xaxis = list([subgroup_ent[s] for s in states])
    x_range = [min(xaxis), max(xaxis)]
    for c, m in zip(p, models):
        yaxis = [divergence_unadj[m][s] for s in states]
        ax.plot(xaxis, yaxis, '.', c=c, markersize=7, alpha=0.4, zorder=-10)
        f = np.poly1d(np.polyfit(xaxis, yaxis, 1))
        ax.plot(x_range, f(x_range), '', c=c, linewidth=2.)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title1)
    ax.xaxis.set_ticks_position('none')
    ax.grid()

    ax = axs[1]
    p = sb.color_palette("colorblind")

    # xaxis = list(subgroup_unif_mean.values())
    x_range = [min(xaxis), max(xaxis)]
    for c, m in zip(p, models):
        yaxis = [divergence[m][s] for s in states]
        ax.plot(xaxis, yaxis, '.', c=c, markersize=7, alpha=0.4, zorder=-10)
        f = np.poly1d(np.polyfit(xaxis, yaxis, 1))
        ax.plot(x_range, f(x_range), '', c=c, linewidth=2.)

    # Legend
    legend_elements = []
    labels_= []

    legend_elements.append(plt.Line2D([0], [0], color='k', marker='.',
                                      markersize=9, alpha=0.7,
                                      markerfacecolor='k', lw=0))
    labels_.append('Subgroup')
    legend_elements.append(plt.Line2D([0], [0], color='k', alpha=0.7,
                                      markersize=11,
                                      markerfacecolor='k', lw=3))
    labels_.append('Trendline')

    legend_elements.append(plt.Line2D([0], [0], color='w', alpha=0.7,
                                      markersize=11,
                                      markerfacecolor='w', lw=3))
    labels_.append('')

    # lines with the names
    for c, m in zip(p, models):
        # dots
        dot = plt.Line2D([0], [0], color=c, marker='.',
                         label='', markersize=8,
                         markerfacecolor=c, lw=0)

        # line 
        line = plt.Line2D([0], [0], color=c, 
                          markersize=11, markerfacecolor=c, lw=2)

        legend_elements.append((dot, line))
        labels_.append(model_names[m])


    legend_labels = [(line, line), (line, line)]
    legend = ax.legend(legend_labels, ['Combined Data 1', 'Combined Data 2'], 
                       handler_map={tuple: HandlerTuple(ndivide=None)})

    legend_position = (1.08, -.2)  # Coordinates (x, y) for top left position
    ax.legend(legend_elements, labels_, loc='upper right',
              bbox_to_anchor=legend_position, frameon=False, ncols=3, handletextpad=.8, columnspacing=0.7,
              handler_map={tuple: HandlerTuple(ndivide=None), 
                           type(line): HandlerLine2D()})

    ax.set_xlabel(xlabel, fontsize=11.8)
    ax.set_title(title2)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_label_coords(-0.1, -.14)
    ax.grid()

    plt.subplots_adjust(wspace=0.05)


def plot_similarity_opinions(states, divergence_unadj, divergence, subgroup_ent, models, model_names,
                             xlabel='Entropy of subgroup\'s (U.S. state) responses', 
                             ylabel=r"$\bar{\mathrm{KL}}$(model, subgroup)",
                             title1='Unadjusted responses', title2='Adjusted responses',
                             figsize=(5.5, 2.5)):
    from matplotlib.legend_handler import HandlerTuple, HandlerLine2D
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)

    ax = axs[0]
    p = sb.color_palette("colorblind")

    # xaxis = list([subgroup_unif_mean[s] for s in states])
    xaxis = list([subgroup_ent[s] for s in states])
    x_range = [min(xaxis), max(xaxis)]
    for c, m in zip(p, models):
        yaxis = [divergence_unadj[m][s] for s in states]
        ax.plot(xaxis, yaxis, '.', c=c, markersize=7, alpha=0.4, zorder=-10)
        f = np.poly1d(np.polyfit(xaxis, yaxis, 1))
        ax.plot(x_range, f(x_range), '', c=c, linewidth=2.)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title1)
    ax.xaxis.set_ticks_position('none')
    ax.grid()

    ax = axs[1]
    p = sb.color_palette("colorblind")

    # xaxis = list(subgroup_unif_mean.values())
    x_range = [min(xaxis), max(xaxis)]
    for c, m in zip(p, models):
        yaxis = [divergence[m][s] for s in states]
        ax.plot(xaxis, yaxis, '.', c=c, markersize=7, alpha=0.4, zorder=-10)
        f = np.poly1d(np.polyfit(xaxis, yaxis, 1))
        ax.plot(x_range, f(x_range), '', c=c, linewidth=2.)

    # Legend
    legend_elements = []
    labels_= []

    legend_elements.append(plt.Line2D([0], [0], color='k', marker='.',
                                      markersize=9, alpha=0.7,
                                      markerfacecolor='k', lw=0))
    labels_.append('Subgroup')
    legend_elements.append(plt.Line2D([0], [0], color='k', alpha=0.7,
                                      markersize=11,
                                      markerfacecolor='k', lw=3))
    labels_.append('Trendline')

    legend_elements.append(plt.Line2D([0], [0], color='w', alpha=0.7,
                                      markersize=11,
                                      markerfacecolor='w', lw=3))
    labels_.append('')

    # lines with the names
    for c, m in zip(p, models):
        # dots
        dot = plt.Line2D([0], [0], color=c, marker='.',
                         label='', markersize=8,
                         markerfacecolor=c, lw=0)

        # line 
        line = plt.Line2D([0], [0], color=c, 
                          markersize=11, markerfacecolor=c, lw=2)

        legend_elements.append((dot, line))
        labels_.append(model_names[m])


    legend_labels = [(line, line), (line, line)]
    legend = ax.legend(legend_labels, ['Combined Data 1', 'Combined Data 2'], 
                       handler_map={tuple: HandlerTuple(ndivide=None)})

    legend_position = (1.08, -.2)  # Coordinates (x, y) for top left position
    ax.legend(legend_elements, labels_, loc='upper right',
              bbox_to_anchor=legend_position, frameon=False, ncols=3, handletextpad=.8, columnspacing=0.7,
              handler_map={tuple: HandlerTuple(ndivide=None), 
                           type(line): HandlerLine2D()})

    ax.set_xlabel(xlabel, fontsize=11.8)
    ax.set_title(title2)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_label_coords(-0.1, -.14)
    ax.grid()

    plt.subplots_adjust(wspace=0.05)
