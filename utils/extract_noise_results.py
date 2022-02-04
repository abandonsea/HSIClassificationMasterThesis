#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Set plot style
plt.style.use('bmh')

############
# Set file #
############
PATH = '../../../Results/noise/'

DATASETS = ['paviau', 'indian_pines', 'salinas']
NETWORKS = ['sdmm', 'dffn', 'vscnn', 'sae3ddrn', '3dcrn']
TEST_CASES = ['full', 'reduced_10', 'reduced_05', 'reduced_01']
NOISE_TYPE = ['salt_and_pepper', 'additive_gaussian', 'multiplicative_gaussian',
              'section_mul_gaussian', 'single_section_gaussian']

VALUE_POSITION = 3
TEST_SIZE = 10

section_order = [6, 4, 2, 0, 1, 3, 5, 7]


# Get test results from text file
def get_values(filename):
    results = {}

    current_key = ''
    with open(filename, 'r') as file:
        line = file.readline()
        while line:
            # Check for OA
            if 'amount' in line:
                words = line.split(' ')
                # current_key = str(words[VALUE_POSITION])
                current_key = ''.join(char for char in str(words[VALUE_POSITION]) if char != '\n')

                if current_key not in results:
                    results[current_key] = {'oa': np.array([]), 'aa': np.array([]), 'kappa': np.array([])}
            elif 'OVERALL ACCURACY' in line:
                words = line.split(' ')
                results[current_key]['oa'] = np.append(results[current_key]['oa'], float(words[VALUE_POSITION]))
            # Check for AA
            elif 'AVERAGE ACCURACY' in line:
                words = line.split(' ')
                results[current_key]['aa'] = np.append(results[current_key]['aa'], float(words[VALUE_POSITION]))
            # Check for kappa
            elif 'KAPPA COEFFICIENT' in line:
                words = line.split(' ')
                results[current_key]['kappa'] = np.append(results[current_key]['kappa'], float(words[VALUE_POSITION]))

            # Get next line
            line = file.readline()

    for key in results:
        assert results[key]['oa'].size == results[key]['kappa'].size, 'Wrong list lengths! [1]'
        assert results[key]['aa'].size == results[key]['kappa'].size, 'Wrong list lengths! [2]'

        for noise in results[key]:
            if results[key][noise].size > TEST_SIZE:
                results[key][noise] = results[key][noise][:TEST_SIZE]
            elif results[key][noise].size < TEST_SIZE:
                raise AssertionError

    return results


# Main for running script independently
def main():
    data_dict = {'paviau': 'Pavia University', 'indian_pines': 'Indian Pines', 'salinas': 'Salinas'}
    case_dict = {'full': '80%', 'reduced_10': '10%', 'reduced_05': '5%', 'reduced_01': '1%'}
    noise_dict = {'salt_and_pepper': 'Amount of affected pixels',
                  'additive_gaussian': 'Noise variance with respect to the data\'s',
                  'multiplicative_gaussian': 'Noise variance with respect to the data\'s'}
    net_dict = {'sdmm': 'S-DMM', 'dffn': 'DFFN', 'vscnn': 'VSCNN', 'sae3ddrn': 'SAE-3DDRN', '3dcrn': '3D-CRN'}

    for data in DATASETS:  # Run for 3 datasets
        for case in TEST_CASES:  # Run for 4 test cases
            zero_noise = {}
            for noise in NOISE_TYPE:  # Run for 5 noise types
                nodes = {}
                for net in NETWORKS:  # All network's results will be in the same graphic
                    file = 'noise_' + noise + '.nst'
                    path = PATH + net + '/' + data + '/' + case + '/'
                    filename = path + file

                    results = get_values(filename)

                    nodes[net] = [(noise, results[noise]['oa'].mean()) for noise in results]

                    if noise == 'salt_and_pepper':
                        zero_noise[net] = nodes[net][0]
                    elif noise == 'additive_gaussian' or noise == 'multiplicative_gaussian':
                        nodes[net].insert(0, zero_noise[net])

                # Plot graphs for the simple noise types
                simple_noise_types = ['salt_and_pepper', 'additive_gaussian', 'multiplicative_gaussian']
                if noise in simple_noise_types:
                    # Generate graph for the current values
                    fig, ax = plt.subplots()
                    # fig.suptitle(f'Using {noise} noise')

                    size = 0
                    for key in nodes:
                        labels, values = zip(*nodes[key])
                        if noise == 'salt_and_pepper':
                            labels = [f'{str(100 * float(label))}%' for label in labels]
                        else:
                            labels = [f'{100 * float(label):.0f}%' for label in labels]
                        size = len(labels)

                        ax.plot(labels, values, linewidth=2.0, label=net_dict[key])

                    ax.set(xlim=(0, size-1), xticks=np.arange(0, size),
                           ylim=(0.0, 1.1), yticks=np.arange(0.2, 1.1, 0.2))
                    ax.set(xlabel=noise_dict[noise], ylabel='Average accuracy',
                           title=f'Train split: {case_dict[case]}')
                    ax.legend()
                    # plt.show()
                    plt.savefig(f'{PATH}{data}_{case}_{noise}.png')

                else:
                    # Generate graph for the current values
                    fig, ax = plt.subplots()
                    # fig.suptitle(f'Using {noise} noise')

                    size = 0
                    for key in nodes:
                        labels, values = zip(*nodes[key])
                        labels = [f'S{label}' for label in labels]
                        size = len(labels)

                        aux_labels, aux_values = [], []
                        for i in range(size):
                            aux_labels.append(labels[section_order[i]])
                            aux_values.append(values[section_order[i]])
                        labels, values = aux_labels, aux_values

                        ax.plot(labels, values, linewidth=2.0, label=net_dict[key])

                    ax.set(xlim=(0, size - 1), xticks=np.arange(0, size),
                           ylim=(0.0, 1.1), yticks=np.arange(0.2, 1.1, 0.2))
                    ax.set(xlabel='Section', ylabel='Average accuracy',
                           title=f'Train split: {case_dict[case]}')
                    ax.legend()
                    # plt.show()
                    plt.savefig(f'{PATH}{data}_{case}_{noise}.png')


if __name__ == '__main__':
    main()
