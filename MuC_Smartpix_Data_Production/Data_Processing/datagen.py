import os
import datagensinglefile 
import argparse
import time

parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input_file", help="File name", required=True)
parser.add_argument("-o", "--output_file", help="Tag", required=True)
ops = parser.parse_args()

datagensinglefile.makeParquet(ops.input_file, ops.output_file)