from tanh import TanH
from step import Step
from relu import ReLU
from sigmoid import Sigmoid
from linear import Linear
from leaky_relu import LeakyRelu
from plot import GraphPlot
import argparse
import numpy as np
import matplotlib.pyplot as plt


def cmd_line_parser():
    parser = argparse.ArgumentParser(description="Plot Activation Functions")

    parser.add_argument('-p', '--plot',
                        help='Plot',
                        choices=[True, False],
                        type=bool,
                        default=False)
    parser.add_argument('-m', '--model',
                         help='Select Model',
                         choices=['linear', 'relu',
                                  'tanh', 'sigmoid',
                                  'step', 'leaky'],
                         type=str,
                         default=None)
    parser.add_argument('-s', '--save',
                         help='Save Model',
                         choices=[True, False],
                         type=bool,
                         default=False)
    return parser.parse_args()

def main():
    args = cmd_line_parser()

    if args.model == 'step':
        func = Step()
    elif args.model == 'relu':
        func = ReLU()
    elif args.model == 'tanh':
        func = TanH()
    elif args.model == 'sigmoid':
        func = Sigmoid()
    elif args.model == 'linear':
        func = Linear()
    elif args.model == 'leaky':
        func = LeakyRelu()
    else:
        print('Invalid model!!! Please see help...')

    x, z = func.function()
    y, z = func.derivative()

    plot = GraphPlot(model=args.model,
                     save_model=args.save,
                     x_val=x, y_val=y, z_val=z)
    plot.plot()


if __name__ == '__main__':
    main()
