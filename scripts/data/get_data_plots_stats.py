import os
import sys
import pickle
import matplotlib.pyplot as plt


def get_plot(li, n, title):
    plt.figure(n)
    plt.hist(li, bins=50)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(title)

def get_comps(path, title, outpath):
    with open(path, 'rb') as f:
        li = pickle.load(f)

    print("original length: {}".format(len(li)))

    li_512 = [x for x in li if x < 512]
    li_1024 = [x for x in li if x < 1024]
    li_2048 = [x for x in li if x < 2048]
    li_larger = [x for x in li if x >= 2048]
    li_remove_outliers = [x for x in li if x < 10000]

    print("<512 length: {}".format(len(li_512)))
    print("<1024 length: {}".format(len(li_1024)))
    print("<2048 length: {}".format(len(li_2048)))
    print("<10000 length: {}".format(len(li_remove_outliers)))
    print(">=2048 length: {}".format(len(li_larger)))

    get_plot(li_512, 1, title) 
    get_plot(li_1024, 2, title)
    get_plot(li_2048, 3, title)
    get_plot(li_remove_outliers, 4, title)
    plt.savefig(outpath)
    plt.show()

def run_data(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            name = f.split('.')[0]
            print(name)
            get_comps(os.path.join(root, f), name, "{}_hist.png".format(name))

run_data(sys.argv[1])
