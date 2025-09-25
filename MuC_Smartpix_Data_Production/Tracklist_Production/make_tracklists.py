import argparse
import subprocess

# user options
parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--number_of_events", help="Total number of events for signal generation", default=10000, type=int)
parser.add_argument("-b", "--bin_size", help="Number of tracks per track list file", default=500, type=int)
parser.add_argument("-f", "--float_precision", help="Float precision for track list files", default=5, type=int)
parser.add_argument("-f", "--flp", help="Direction of sensor, 0 for FE up, 1 for FE down", default=0, type=int)
parser.add_argument("-o", "--overwrite", help="Whether to overwrite existing track list files", default=False, type=bool)
parser.add_argument("-p", "--plot", help="Whether to make diagnostic plots", default=False, type=bool)
ops = parser.parse_args()

