import sys

from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import numpy as np


testy = [
    np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1]),
    np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
    np.array([5, 1, 5, 1, 5, 5, 1, 1, 5, 5, 1, 1, 5]),
    np.array([-1, -1, -1, -5, -5, -3, -4, -2, 1, 2, 2, 1]),
    np.zeros((1, 520)),
    np.arange(0, 521, 1),
    np.eye(7),
    np.dstack([np.eye(7), np.eye(7), np.eye(7)]),
    np.ones((1, 1, 1, 1, 1, 1, 10))
]


def get_info(data):
    data_cp = np.asarray(data.copy()).astype(int)
    s = data_cp.shape
    elementy = 1
    for d in s:
        elementy *= d
    info = [s, elementy]
    return info


def RLE_code(data):
    info = get_info(data)
    flattened = data.flatten()
    coded = []
    init = True
    cr_number = 0
    instances = 1
    for x in range(len(flattened)):
        if init:
            cr_number = flattened[x]
            init = False
        else:
            if flattened[x] == cr_number:
                instances += 1
            else:
                coded.append(instances)
                coded.append(cr_number)
                cr_number = flattened[x]
                instances = 1
    coded.append(int(instances))
    coded.append(cr_number)
    info.append(np.asarray(coded))
    return info

def RLE_decode(data):
    decoded = [None] * data[1]
    coded = data[2]
    x = 0
    n = 0
    el = 1
    while x < data[1]:
        for _ in range(int(coded[n])):
            decoded[x] = coded[el]
            x += 1
        n += 2
        el += 2
    decoded = np.asarray(decoded).reshape(data[0])
    return decoded


def Byte_run_code(data):
    info = get_info(data)
    flattened = data.flatten()
    coded = []
    init = True
    cr_number = 0
    instances = 1
    not_even = False
    not_even_list = []

    for x in range(len(flattened)):
        if init:
            cr_number = flattened[x]
            init = False
        else:
            if flattened[x] == cr_number or len(not_even_list) >= 128:
                if not_even:
                    coded.append(len(not_even_list) - 1)
                    coded.extend(not_even_list)
                    if len(not_even_list) >= 128:
                        if flattened[x] != cr_number:
                            not_even_list = [cr_number]
                            cr_number = flattened[x]
                            instances = 1
                            not_even = True
                        else:
                            not_even_list = []
                            not_even = False
                            instances = 2
                    else:
                        not_even_list = []
                        not_even = False
                        instances = 2
                else:
                    instances += 1
            elif instances > 1 or instances >= 128:
                coded.append(-(instances-1))
                coded.append(cr_number)
                cr_number = flattened[x]
                instances = 1
            elif instances == 1:
                not_even = True
                not_even_list.append(cr_number)
                cr_number = flattened[x]
    if not_even:
        not_even_list.append(cr_number)

    if instances == 1:
        if len(not_even_list) > 0:
            coded.append(len(not_even_list) - 1)
            coded.extend(not_even_list)
            info.append(np.asarray(coded))
        else:
            coded.append(0)
            coded.append(cr_number)
            info.append(np.asarray(coded))
    else:
        coded.append(-(instances-1))
        coded.append(cr_number)
        info.append(np.asarray(coded))
    return info

def Byte_run_decode(data):
    decoded = [None] * data[1]
    coded = data[2]
    x = 0
    n = 0
    el = 1
    while x < data[1]:
        if coded[n] < 0:
            nr_of_elements = -(int(coded[n])) + 1
            for _ in range(nr_of_elements):
                decoded[x] = coded[el]
                x += 1
            n += 2
            el += 2
        else:
            el2 = n + 1
            iter = 0
            while x < data[1] and iter < (int(coded[n]) + 1):
                decoded[x] = coded[el2]
                x += 1
                el2 += 1
                iter += 1
            n = el2
            el = el2 + 1
    decoded = np.asarray(decoded).reshape(data[0])
    return decoded

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj,np.ndarray):
        size=obj.nbytes
    elif isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def main():


    files = ["rysunek_techniczny.jpg", "wzor_dokumentu.jpg", "kolorowe.jpg"]
    methods = [["RLE", RLE_code, RLE_decode], ["Byte run", Byte_run_code, Byte_run_decode]]
    for m in methods:
        print(m[0])
        for t in testy:
            coded = m[1](t)
            decoded = m[2](coded)
            equal = (t == decoded).all()
            if not equal:
                for t, i in enumerate(t == decoded):
                    print(f"{i}: {t}")
            print(f"\t{equal}")

    for f in files:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f)
        print(f)
        img = plt.imread(f"images/{f}")
        ax.imshow(img)
        for m in methods:
            coded = m[1](img)
            decoded = m[2](coded)
            cr = abs(get_size(img.astype(int)))/abs(get_size(coded))
            equal = (decoded == img).all()
            print("\t"+m[0])
            print("\t\t" + f"Czy takie same? - {equal}")
            print(f"\t\tRozmiar przed kompresją -\t{get_size(img)}")
            print(f"\t\tRozmiar po kompresji -\t\t{get_size(coded)}")
            print("\t\t" + f"Współczynnik kompresji - {cr}")
    plt.show()




if __name__ == "__main__":
    main()
