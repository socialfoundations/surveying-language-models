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


def figure_1b(model_entropies, census_norm_entropies, base_models, model_sizes):
    fig, ax = plt.subplots(1, 2, figsize=(4.6, 1.9), sharey=True)

    u = census_norm_entropies['SEX']
    ax[0].plot([1e7, 1e12], [u - 0.015, u - 0.015], c=gcolor)  # -0.015 to avoid overlap with uniform
    u = census_norm_entropies['FER']
    ax[1].plot([1e7, 1e12], [u, u], c=gcolor)

    msize = 4

    plot_vars = ['SEX', 'FER']
    sizes = [model_sizes[m] for m in base_models]
    for i in range(2):
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

    ax[0].set_ylabel('Entropy of\nmodel rresponses', fontsize=12)
    ax[0].set_title('SEX question')
    ax[1].set_title('FER question')

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
    legend_position = (-0.25, -0.3)
    ax[-1].legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=legend_position, frameon=False, ncols=3, fontsize=10.5, columnspacing=1.1)

    plt.subplots_adjust(wspace=0.1)

def plot_a_bias(a_bias, base_models, model_names, alpha=0.4):
    fig, ax = plt.subplots(figsize=(9, 1.5))

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
    legend_position = (1.0, 1.27)  # Coordinates (x, y) for top left position
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

def plot_abias_vs_entropy(mean_entropy, mean_ordering_bias):
    fig, ax = plt.subplots(figsize=(1.8, 1.4))

    # abias vs entropy
    ax.plot(mean_entropy, mean_ordering_bias, 'D', c=bcolor, markersize=4)

    # legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='D', color=bcolor,
                                      label='Model $m$', markersize=4,
                                      markerfacecolor=bcolor, lw=0))
    legend_position = (1.05, 1.27)  # Coordinates (x, y) for top left position
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=legend_position, frameon=False, handletextpad=-0.1, ncols=3, columnspacing=-0.7)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_ylim([0, 0.6])
    ax.grid()

    ax.set_xlabel('Mean response entropy\n', fontsize=11)
    ax.set_ylabel('Mean A-bias', fontsize=11)
    ax.yaxis.set_label_coords(-0.25, 0.47)


def plot_adjusted_entropy(entropies, census_entropies, models, variables, model_names, figsize, alpha=0.4):
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
    legend_position = (1.015, 1.3)  # Coordinates (x, y) for top left position
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
        labels.append('census')
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
    ax.set_ylabel('Entropy of\nresponses', fontsize=12, labelpad=5, y=0.5)


def plot_kl_unif_census(models, adjusted_kl_census, adjusted_kl_uniform, variables, model_names, alpha=0.6, ymax=4):
    assert len(models) == 6

    fig, axs = plt.subplots(2, 3, figsize=(4.2, 2.5), sharex=True, sharey=True)

    for i in range(3):
        for j in range(2):
            m = models[i + j * 3]
            ax = axs[j, i]
            xaxis = [adjusted_kl_census[v][m] for v in variables]
            yaxis = [adjusted_kl_uniform[v][m] for v in variables]

            ax.plot(xaxis, yaxis, '.', c=bcolor, alpha=alpha, markersize=8)
            ax.set_title(model_names[m], fontsize=10)
            ax.grid()
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')

            ax.set_xticks([i for i in range(ymax+1)])
            ax.set_yticks([i for i in range(ymax+1)])

            ax.plot([0, ymax], [0, ymax], ls="-", c=".75", linewidth=0.4, zorder=-10)

            for spine in ax.spines.values():
                spine.set_edgecolor('0.3')


    ax.set_ylim([-.1, ymax])
    ax.set_xlim([-.1, ymax])

    plt.tight_layout(h_pad=0.2)
    ax = fig.add_subplot(111, frameon=False)

    legend_elements = []
    legend_elements.append((plt.Line2D([0], [0], marker='.', color=bcolor, alpha=1,
                                       label=r'Survey question', markersize=11,
                                       markerfacecolor=bcolor, lw=0)))
    legend_position = (1.04, 1.3)  # Coordinates (x, y) for top left position
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=legend_position, frameon=False, handletextpad=0.5, ncols=3, columnspacing=0.9)

    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("KL(model || census)", fontsize=11)
    plt.ylabel("KL(model || uniform)", fontsize=11, y=.5, labelpad=-3)

def plot_divergence_subgroups(divergence, divergence_uniform, selected_models, model_names, alpha=0.1,
                              ylim=[-0.1, 2.], yticks=[0., 0.5, 1., 1.5, 2.],
                              ylabel=r'$\bar\mathrm{KL}$(model, subgroup)', clabel='Census subgroup'):
    fig, ax = plt.subplots(figsize=(11, 1.5))

    # divergence for each model
    for i, model in enumerate(selected_models):
        ys = list(divergence[model].values())
        xs = [i for _ in ys]
        ax.plot(xs, ys, '.', c=bcolor, alpha=alpha, markersize=12)
        ax.plot(i, divergence_uniform[model], '*', c=rcolor, markersize=9)

    # legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='o', color=bcolor,
                                      label='Census subgroup', markersize=6,
                                      markerfacecolor=bcolor, lw=0))
    legend_elements.append(plt.Line2D([0], [0], marker='*', color=rcolor,
                                      label='Uniformly random responses', markersize=9,
                                      markerfacecolor=rcolor, lw=0))
    legend_position = (1.01, 1.32)  # Coordinates (x, y) for top left position
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=legend_position, frameon=False, handletextpad=0.4, ncols=4, fontsize=10.5)

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
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='y', which='both', labelsize=10)

def plot_discriminator(accuracies, models, model_names, title="Accuracy in discriminating model-generated data"):
    fig, ax = plt.subplots(figsize=(10.5, 1.2))
    plt.plot([i for i, _ in enumerate(models)], [np.mean(accuracies[model]) * 100 for model in models],
             'X', color=bcolor, alpha=0.9)

    dx = 0.2;
    dy = 0.05
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    ax.set_xticks([i for i in range(len(models))])
    labels = [model_names[m] for m in models]
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=10)

    ax.set_xlim([-0.5, len(models) - 0.5])
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=100)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', zorder=-10)
    ax.set_axisbelow(True)
    ax.set_ylim([95, 100])
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.tick_params(axis='y', which='both', labelsize=10.5)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.set_title(title, fontsize=11)


def plot_similarity_opinions(states, subgroup_alignment, alignment_uniform, models, model_names,
                             xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(2.9, 2.4))

    p = sb.color_palette("colorblind")

    xaxis = alignment_uniform
    x_range = [min(xaxis), max(xaxis)]
    for c, m in zip(p, models):
        yaxis = [subgroup_alignment[m][s] for s in states]
        ax.plot(xaxis, yaxis, '.', c=c, markersize=7, alpha=0.4, zorder=-10)
        f = np.poly1d(np.polyfit(xaxis, yaxis, 1))
        ax.plot(x_range, f(x_range), '', c=c, linewidth=2.)

    # Legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], color='k', marker='.',
                                      label='', markersize=8, alpha=0,
                                      markerfacecolor='k', lw=0))
    legend_elements.append(plt.Line2D([0], [0], color='k', alpha=0.,
                                      label='', markersize=11,
                                      markerfacecolor='k', lw=2))

    for c, m in zip(p, models):
        legend_elements.append(plt.Line2D([0], [0], color=c, marker='.',
                                          label='', markersize=8,
                                          markerfacecolor=c, lw=0))
    legend_elements.append(plt.Line2D([0], [0], color='k', marker='.',
                                      label='Subgroup', markersize=9, alpha=0.7,
                                      markerfacecolor='k', lw=0))
    legend_elements.append(plt.Line2D([0], [0], color='k', alpha=0.7,
                                      label='Trendline', markersize=11,
                                      markerfacecolor='k', lw=3))

    for c, m in zip(p, models):
        legend_elements.append(plt.Line2D([0], [0], color=c,
                                          label=model_names[m], markersize=11,
                                          markerfacecolor=c, lw=2))

    legend_position = (1.93, 1.16)  # Coordinates (x, y) for top left position
    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=legend_position, frameon=False, ncols=2, handletextpad=.8, columnspacing=-1.)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=11.8)
    ax.set_title(title)
    ax.yaxis.set_label_coords(-0.2, 0.45)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.grid()