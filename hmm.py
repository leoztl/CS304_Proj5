## use 10 files of each digit to train
## author: Yike Guo

from problem3 import *
pathname = "recds_yg/"
filenames = []
for root, dirs, files in os.walk(pathname):
    for name in files:     
        filename = pathname + name
        filenames.append(filename)
            
hmm_digits = get_hmm_digits(filenames)
print(len(hmm_digits))