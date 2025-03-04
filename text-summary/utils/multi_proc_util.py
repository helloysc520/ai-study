import pandas as pd
import numpy as np
from multiprocessing import cpu_count,Pool

'''
多线程处理工具类

'''

cores = cpu_count()
partitions = cores

def parallelize(df,func):

    data_split = np.array_split(df,partitions)

    #初始化线程池
    pool = Pool(cores)

    data = pd.concat(pool.map(func,data_split))

    #关闭线程池,保证不会有新的任务加进来
    pool.close()

    #等待所有线程任务结束
    pool.join()

    return data

if __name__ == '__main__':
    cores = cpu_count()
    print("cpu cores:", cores)


