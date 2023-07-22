from tabulate import tabulate


def print_table(data):
    print(tabulate(data,
                   headers=['Dataset',
                            'ML Algorithm', 'Precision', 'Recall', 'Accuracy'],
                   tablefmt="fancy_grid"))
