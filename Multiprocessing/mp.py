import multiprocessing as mp
import numpy as np

def get_commons(list_1, list_2):
    return list(set(list_1).intersection(list_2))

def power(list_1):
    power_list = []
    for i in list_1:
        power_list.append(i*i)

    return power_list

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

def main():
    randomNumbers = np.random.randint(1000, size=10_000_000)

    pool = mp.Pool(mp.cpu_count())

    x = split(randomNumbers, 1_000_000)

    print("Splitting done")

    cycles = 10

    for i in range(cycles):
        results = pool.apply(power, args=([x])) 
        print("Done {} of {}".format(i+1, cycles))

    pool.close()

    print(results[:10])

if __name__ == "__main__":
    main()