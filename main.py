import os
import sys
import time

from utils import inria_data_generator

def main():
    path = sys.argv[1]
    output_path = sys.argv[2]
    data = inria_data_generator(path, output_path)
    data.split_all_images()

if __name__ == '__main__':
    main()
