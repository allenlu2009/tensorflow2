# -*- coding: utf-8 -*-
#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import re
import pandas as pd

# set up regular expressions
# use https://regexper.com to visualise these if required
#rx_dict = {
#    'school': re.compile(r'School = (?P<school>.*)\n'),
#    'grade': re.compile(r'Grade = (?P<grade>\d+)\n'),
#    'name_score': re.compile(r'(?P<name_score>Name|Score)'),
#}

#rx_dict = {
#    'learning_rate': re.compile(r'(?P<rate>\d+[.,]?\d*).*hr.*min$'),
#    'grade': re.compile(r'Grade = (?P<grade>\d+)\n'),
#    'name_score': re.compile(r'(?P<name_score>Name|Score)'),
#}


def print_sysinfo(logfile, argvstr, cc='#'):
  print(argvstr)
  date = os.popen('date').read().rstrip()
  pwd  = os.popen('pwd').read().rstrip()
  hostname = os.popen('hostname').read().rstrip()
  user = os.popen('whoami').read().rstrip()
  
  print("{0} Generated {1} by {2}\n{0} {3}:{4}\n{0} {5}\n\n".format(cc,date,user,hostname,pwd,argvstr), file=logfile)
  
  debug = False
  if debug:
    print("{0} Generated {1} by {2}\n{0} {3}:{4}\n{0} {5}\n\n".format(cc,date,user,hostname,pwd,argvstr))

    
def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    number_pattern = '(\d+(?:\.\d+)?)'
    line_pattern = '^\s+%s\s+$' % ('\s+'.join([number_pattern for x in range(10)]))

    match = re.match(line_pattern, line)
    if match:
            print(match.groups())
            return match.groups()
    # if there are no matches
    return None


  
def parse_file(filepath):
    """
    Parse text at given filepath

    Parameters
    ----------
    filepath : str
        Filepath for file_object to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

    """

    #number_pattern = '(\d+(?:\.\d+)?)'
    #number_pattern = '(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+[|k]?'
    #line_pattern = '^\s*%s\.*hr.*min.*$' % ('\s+'.join([number_pattern for x in range(5)]))

    line_pattern = r'^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+[|k]?\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+[|k]?\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+[|k]?\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+[|k]?\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+[|k]?\s+.*hr.*min.*$'

    data = []  # create an empty list to collect the data
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line:
          #print("line: ", line)
          match = re.match(line_pattern, line)
          if match:
            #print("match line: ", line)
            #print(match.groups())
            row = {
              'l_rate': match.group(1),
              'iter': match.group(2),
              'epoch': match.group(3),
              'num': match.group(4),
              'valid_loss': match.group(5),
              'valid_acc': match.group(6),
              'train_loss': match.group(7),
              'train_acc': match.group(8),
              'batch_loss': match.group(9),
              'batch_acc': match.group(10)
              }
            #print(row)
            #return match.groups()

            # append the dictionary to the data list
            data.append(row)

          line = file_object.readline()

        # create a pandas DataFrame from the list of dicts
        print("data: ", data)
        df = pd.DataFrame(data)
        print(df.ndim)
        print(df.shape)
        print(df.dtypes)
        print("data frame: ", df)
        # set the School, Grade, and Student number as the index
        #df.set_index(['epoch', 'valid_loss', 'valid_acc', 'train_loss', 'train_acc'], inplace=True)
        #df.set_index(['epoch'], inplace=True)
        # consolidate df to remove nans
        #df = df.groupby(level=data.index.epoch).first()
        # upgrade Score from float to integer
        df = df.apply(pd.to_numeric, errors='ignore')
    return df


  
def parse(argv):
  argvstr = " ".join(argv)
  
  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-n", "--network", default="alexnet",
        help="name of network")
  ap.add_argument("-d", "--dataset", default="imagenet",
        help="name of dataset")
  ap.add_argument("-l", "--logfile", default="tst.log",
        help="name of logfile")
  ap.add_argument("-o", "--outfile", default="out.log",
        help="name of logfile")
  ap.add_argument("-e", "--epoch", type=int, default=10,
        help="max epoch")
  args = vars(ap.parse_args())

  outfile = open(args["outfile"], 'w', encoding='UTF-8')
  print("args: ",args)
  print_sysinfo(outfile, argvstr)

  df = parse_file(args["logfile"])
  #print(data)
  print(df, file=outfile)


  # plot the accuracies
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(df.epoch, df.batch_acc,
           label="batch_acc")
  plt.plot(df.epoch, df.train_acc,
           label="train_acc")
  plt.plot(df.epoch, df.valid_acc,
           label="valid_acc")
  plt.title("{}: rank-1 and rank-5 accuracy on {}".format(
    args["network"], args["dataset"]))
  plt.xlabel("Epoch #")
  plt.ylabel("Accuracy")
  plt.legend(loc="lower right")

  # plot the losses
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(df.epoch, df.batch_loss,
           label="batch_loss")
  plt.plot(df.epoch, df.train_loss,
           label="train_loss")
  plt.plot(df.epoch, df.valid_loss,
           label="val_loss")
  plt.title("{}: cross-entropy loss on {}".format(args["network"],
                                                  args["dataset"]))
  plt.xlabel("Epoch #")
  plt.ylabel("Loss")
  plt.legend(loc="upper right")
  plt.show()





if __name__ =="__main__":
  debug = False
  #debug = True
  if debug:
    print(sys.argv)
    print(sys.argv[1:]) # ignore script name

  parse(sys.argv)

