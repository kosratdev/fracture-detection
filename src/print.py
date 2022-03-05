from tabulate import tabulate


def print_table(data):
    print(tabulate(data,
                   headers=['Noise', 'Contrast', 'Edge Detector',
                            'ML Algorithm', 'Accuracy'],
                   tablefmt="fancy_grid"))
