import numpy as np
import time
import random

class TS:

    def __init__(self, MAX_GEN, length, N, Num, Resource, swapMode, alpha=[], beta=[], ptime=[], weight=[]):
        '''
        MAX_GEN:最大迭代次数
        N:Can_N的長度(前N個最好的)
        Resource:initial資源量
        length：tabu list的長度
        Num: number of jobs(即編碼長度)
        '''
        self.MAX_GEN = MAX_GEN
        self.length = length
        self.N = N
        self.Num = Num
        self.Resource = Resource
        self.swapMode = swapMode
        self.alpha = alpha
        self.beta = beta
        self.sigma = [beta[i]-alpha[i] for i in range(len(alpha))]
        self.ptime = ptime
        self.weight = weight
        self.P_pi = []
        self.neighbor = []  # 鄰居

        self.Ghh = []  # 當前最佳編碼
        self.current_fitness = 0.0  # 當前最佳編碼的fitness
        self.fitness_Ghh_current_list = []  # 當前最佳編碼的fitness list
        self.Ghh_list = []  # 當前最佳編碼的list

        self.bestGh = []  # 最好的编码
        self.best_fitness = 0.0  # 最好編碼的fitness
        self.best_fitness_list = []  # 最好編碼的fitness list

        self.tabu_list = np.random.randint(0, 1, size=(self.length, self.Num)).tolist()  # 初始化禁忌表 (length*Num的二維 0 陣列)

    # 初始解用johnson sequence排，結果存入Ghh
    def InitialSolution(self):
        c = [0] * len(self.alpha)
        type1_index = []
        type2_index = []
        for i in range(len(self.alpha)):
            c[i] =self.beta[i] - self.alpha[i]
            if c[i] > 0:
                type1_index.append(i)
            else:
                type2_index.append(i)
        type1_index.sort(key=lambda type1_index: self.alpha[type1_index])
        type2_index.sort(key=lambda type2_index: self.beta[type2_index], reverse=True)
        sequence = type1_index + type2_index
        sequence = [element+1 for element in sequence]
        self.Ghh = sequence

    # 產生neighbor(1:flip, 2:exchange, 3:hybrid1&2, 4:2-opt)
    def swap(self, swapMode=1):
        if swapMode == 1:
            for i in range(len(self.Ghh)-1):
                temp = self.Ghh.copy()
                temp[i], temp[i+1] = temp[i+1], temp[i]
                self.neighbor.append(temp)
        elif swapMode == 2:
            for i in range(len(self.Ghh)):
                l = random.randint(1, len(self.Ghh)-1)
                i = random.randint(0, len(self.Ghh)-1)
                temp = self.Ghh.copy()
                temp[i:(i+l)] = reversed(temp[i:(i+l)])
                self.neighbor.append(temp)
        elif swapMode == 3:
            pool1 = []
            pool2 = []
            for i in range(len(self.Ghh)-1):
                temp = self.Ghh.copy()
                temp[i], temp[i+1] = temp[i+1], temp[i]
                pool1.append(temp)
            for i in range(len(self.Ghh)):
                l = random.randint(1, len(self.Ghh)-1)
                i = random.randint(0, len(self.Ghh)-1)
                temp = self.Ghh.copy()
                temp[i:(i+l)] = reversed(temp[i:(i+l)])
                pool2.append(temp)

            pool_1 = random.sample(pool1, len(self.Ghh)//2)
            pool_2 = random.sample(pool2, len(self.Ghh)//2)
            self.neighbor = pool_1+pool_2
        elif swapMode == 4:
            for i in range(len(self.Ghh) - 1):
                for j in range(i + 1, len(self.Ghh)):
                    temp = self.Ghh.copy()
                    temp[i], temp[j] = temp[j], temp[i]
                    self.neighbor.append(temp)
        print("neighbor:", self.neighbor)

    # 判断某個編碼是否在tabu_list中
    def judgment(self, GN=[]):
        # GN：判斷swap operation是否在tabu list內. ex: [1, 4]
        flag = 0  # 表示這個編碼不在禁忌表中
        for temp in self.tabu_list:
            temp_reverse = []
            for i in reversed(temp):
                temp_reverse.append(i)
            if GN == temp or GN == temp_reverse:
                flag = 1  # 表示这个編碼在tabu list中
                break
        return flag

    # update tabu_list
    def ChangeTabuList(self, GN=[], flag_=1):
        # GN：要加入tabu_list的swap操作 ex:[1,2]
        # flag_=1表滿足aspiration criterion
        if flag_ == 0:
            self.tabu_list.pop()  # pop出最後一個編碼
            self.tabu_list.insert(0, GN)  # 插入一個新的編碼在最前面
        if flag_ == 1:
            for i, temp in enumerate(self.tabu_list):
                temp_reverse = []
                for j in reversed(temp):
                    temp_reverse.append(j)
                if GN == temp or GN == temp_reverse:
                    self.tabu_list.pop(i)
                    self.tabu_list.insert(0, GN)

    # fitness function
    def fitness(self, GN=[]): # GN:要計算fitness的編碼
        # 累積的WiCi
        value = 0
        current_c = 0
        R = self.Resource
        for i in range(len(GN)):
            index = GN[i]
            if R < self.alpha[index-1]:
                return 999999
            current_c += self.ptime[index-1]
            value += self.weight[index-1]*current_c
            R = R + self.sigma[index-1]
        return value

    def _solver(self):
        self.InitialSolution()
        self.current_fitness = self.fitness(GN=self.Ghh)  # self.Ghh的fitness
        self.Ghh_list.append(self.Ghh.copy())  # update當前最佳編碼的list
        self.fitness_Ghh_current_list.append(self.current_fitness)  # update當前最佳fitness的list

        self.bestGh = self.Ghh  # copy self.Ghh到最好的編碼self.bestGh
        self.best_fitness = self.current_fitness  # 最好的fitness
        self.best_fitness_list.append(self.best_fitness)


        step = 0
        while step <= self.MAX_GEN:
            print("step", step, "", end='')
            self.swap(swapMode=self.swapMode)  # 產生neighbor(每次迭代要清空)
            fitness = []
            for temp in self.neighbor:
                temp_fitness = self.fitness(GN=temp)
                fitness.append(temp_fitness)

            # 將fitness.neighbor　由小到大排序
            fitness_sort = sorted(fitness)
            neighbor_sort = sorted(self.neighbor, key=lambda y: self.fitness(y))
            self.neighbor = []  # 將neighbor清空
            neighbor_sort_N = neighbor_sort[:self.N]  # 選取neighbor中fitness最好的前N個編碼
            fitness_sort_N = fitness_sort[:self.N]  # 選取neighbor中fitness最好的前N個fitness

            # temp從最好的候選解開使跑
            for temp in neighbor_sort_N:
                # 紀錄這個neighbor與Ghh哪邊不相同
                dif = []
                GN = []  # 只要最左和最右的
                for i in range(len(temp)):
                    if self.Ghh[i] != temp[i]:
                        dif.append(self.Ghh[i])
                if len(dif) > 0:
                    GN.append(dif[0])
                    GN.append(dif[-1])

                flag = self.judgment(GN=GN)  # 判断這個swap是否有在tabu_list裡
                # flag==1代表有在tabu list裡
                if flag == 1:
                    # 若符合aspiration criterion
                    if self.fitness(temp) < self.best_fitness:
                        self.current_fitness = self.fitness(temp)
                        self.Ghh = temp
                        self.Ghh_list.append(self.Ghh.copy())  # 更新當前最佳編碼的列表
                        self.fitness_Ghh_current_list.append(self.current_fitness)  # 更新當前的最佳fitness值列表
                        # 更新禁忌表
                        self.ChangeTabuList(GN=GN, flag_=1)
                        self.best_fitness = self.current_fitness
                        self.best_fitness_list.append(self.best_fitness)
                        self.bestGh = temp.copy()  # 更新最好的编碼
                        break
                    else:
                        continue
                else:
                    self.current_fitness = self.fitness(temp)
                    self.Ghh = temp
                    self.Ghh_list.append(self.Ghh.copy())  # update當前最佳編碼的列表
                    self.fitness_Ghh_current_list.append(self.current_fitness)  # update當前的最佳fitness值列表
                    # update tabu list
                    self.ChangeTabuList(GN=GN, flag_=0)
                    if self.current_fitness < self.best_fitness:
                        self.best_fitness = self.current_fitness
                        self.best_fitness_list.append(self.best_fitness)
                        self.bestGh = temp.copy()  # update最好的編碼
                    break
            step += 1
        print(self.best_fitness)
        print(self.bestGh)


if __name__ == '__main__':
    file_location = "data/n50p50ab10V1.2.txt"
    with open(file_location, 'r') as input_data_file:
        input_data = ''.join(input_data_file.readlines())

    lines = input_data.split('\n')

    firstLine = lines[0].split()
    job_count = int(firstLine[0])
    R = int(firstLine[1])  # initial resource
    ptime_i = []
    alpha_i = []
    beta_i = []
    weight_i = []
    for i in range(1, job_count+1):
        line = lines[i]
        parts = line.split()
        alpha_i.append(int(parts[0]))
        beta_i.append(int(parts[1]))
        ptime_i.append(int(parts[2]))
        weight_i.append(int(parts[3]))

    start = time.time()
    ts = TS(MAX_GEN=600, length=7, N=len(alpha_i), Num=len(alpha_i), Resource=R, swapMode=3, alpha=alpha_i, beta=beta_i, ptime=ptime_i, weight=weight_i)
    ts._solver()
    end = time.time()
    print("time: ", end-start, "(sec)")




