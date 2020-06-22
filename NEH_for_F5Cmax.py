import time
import numpy as np
import random
from pandas import read_csv

class FlowShop_Heuristic():

    def __init__(self, Num, p_time):
        self.p_time = p_time
        self.job_count = Num
        self.machine_count = len(p_time)
        self.s1 = []
        self.s2 = []
        self.s3 = []
        self.s4 = []
        self.s5 = []
        self.s6 = []

        # region tabusearch
        self.neighbor = []  # 鄰居

        self.Ghh = []  # 當前最佳編碼
        self.current_fitness = 0.0  # 當前最佳編碼的fitness
        self.fitness_Ghh_current_list = []  # 當前最佳編碼的fitness list
        self.Ghh_list = []  # 當前最佳編碼的list

        self.bestGh = []  # 最好的编码
        self.best_fitness = 0.0  # 最好編碼的fitness
        self.best_fitness_list = []  # 最好編碼的fitness list

        self.tabu_list = np.random.randint(0, 1, size=(self.job_count, self.job_count)).tolist()  # 初始化禁忌表 (length*Num的二維 0 陣列)
        #endregion\

        print(self.p_time)

    #region tabu function
    # 產生neighbor(1:flip, 2:exchange, 3:hybrid1&2, 4:2-opt)
    def swap(self, swapMode=1):
        if swapMode == 1:
            for i in range(len(self.Ghh) - 1):
                temp = self.Ghh.copy()
                temp[i], temp[i + 1] = temp[i + 1], temp[i]
                self.neighbor.append(temp)
        elif swapMode == 2:
            for i in range(len(self.Ghh)):
                l = random.randint(1, len(self.Ghh) - 1)
                i = random.randint(0, len(self.Ghh) - 1)
                temp = self.Ghh.copy()
                temp[i:(i + l)] = reversed(temp[i:(i + l)])
                self.neighbor.append(temp)
        elif swapMode == 3:
            pool1 = []
            pool2 = []
            for i in range(len(self.Ghh) - 1):
                temp = self.Ghh.copy()
                temp[i], temp[i + 1] = temp[i + 1], temp[i]
                pool1.append(temp)
            for i in range(len(self.Ghh)):
                l = random.randint(1, len(self.Ghh) - 1)
                i = random.randint(0, len(self.Ghh) - 1)
                temp = self.Ghh.copy()
                temp[i:(i + l)] = reversed(temp[i:(i + l)])
                pool2.append(temp)

            pool_1 = random.sample(pool1, len(self.Ghh) // 2)
            pool_2 = random.sample(pool2, len(self.Ghh) // 2)
            self.neighbor = pool_1 + pool_2
        elif swapMode == 4:
            for i in range(len(self.Ghh) - 1):
                for j in range(i + 1, len(self.Ghh)):
                    temp = self.Ghh.copy()
                    temp[i], temp[j] = temp[j], temp[i]
                    self.neighbor.append(temp)

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
    #endregion

    # 初始解用johnson sequence排，結果存入Ghh
    def InitialSolution(self):
        total_ptime = []
        index = []
        for i in range(0, self.job_count):
            tmp = 0
            for j in range(0, self.machine_count):
                tmp += self.p_time[j][i]
            total_ptime.append(tmp)
            index.append(i)
        seq = sorted(index, key=lambda index: total_ptime[index], reverse=True)
        seq = [i+1 for i in seq]
        self.s1 = seq
        self.Ghh = self.s1


    # calculate the makespan
    # GN : a job sequence
    def makespan(self, GN=[]):
        times = self.p_time
        # makespan = [[0] * (self.machine_count + 1) for _ in range(0, self.job_count + 1)]
        makespan = [[0] * (self.machine_count + 1) for _ in range(0, len(GN) + 1)]
        for i, job in enumerate(GN):
            for machine in range(0, self.machine_count):
                makespan[i + 1][machine + 1] = max(makespan[i][machine + 1], makespan[i + 1][machine]) + times[machine][job - 1]

        # return makespan[self.job_count][self.machine_count]
        return makespan[len(GN)][self.machine_count]

    def NEH(self):
        # step1
        self.InitialSolution()

        # region step2 & 3
        init = [self.s1[0], self.s1[1]]
        if self.makespan([self.s1[0], self.s1[1]]) > self.makespan([self.s1[1], self.s1[0]]):
            init = [self.s1[1], self.s1[0]]
        tmp = init
        for i in range(2, self.job_count):
            M = 999999999
            ind = 0
            for j in range(0, len(tmp)+1):
                t = tmp.copy()
                t.insert(j, self.s1[i])
                if self.makespan(t) < M:
                    M = self.makespan(t)
                    ind = j
            tmp.insert(ind, self.s1[i])
        self.s2 = tmp
        # endregion

        print("s1:", self.s1)
        print("s1 makespan:", self.makespan(self.s1))
        print("s2:", self.s2)
        print("s2 makespan:", self.makespan(self.s2))

    def SlopeInd(self):
        m = self.machine_count
        A = [0]*(self.job_count+1)
        ind = [i for i in range(self.job_count+1)]
        A[0] = -1

        for job in range(1, self.job_count+1):
            sum = 0
            for i in range(1, m+1):
                sum += (m-(2*i-1))*self.p_time[i-1][job-1]
            sum *= -1
            A[job] = sum
        ind = sorted(ind[1:], key=lambda index: A[index], reverse=True)
        self.s3 = ind

        # region step2&3
        init = [self.s3[0], self.s3[1]]
        if self.makespan([self.s3[0], self.s3[1]]) > self.makespan([self.s3[1], self.s3[0]]):
            init = [self.s3[1], self.s3[0]]
        tmp = init
        for i in range(2, self.job_count):
            M = 999999999
            ind = 0
            for j in range(0, len(tmp) + 1):
                t = tmp.copy()
                t.insert(j, self.s3[i])
                if self.makespan(t) < M:
                    M = self.makespan(t)
                    ind = j
            tmp.insert(ind, self.s3[i])
        self.s4 = tmp
        #endregion

        print("s3:", self.s3)
        print("s3 makespan:", self.makespan(self.s3))
        print("s4:", self.s4)
        print("s4 makespan:", self.makespan(self.s4))

    def ts(self, MAX_GEN, length, N, Num, swapMode):
        MAX_GEN = MAX_GEN
        length = length
        N = N
        Num = Num
        swapMode = swapMode

        self.Ghh = self.s1
        self.current_fitness = self.makespan(GN=self.Ghh)  # self.Ghh的fitness
        self.Ghh_list.append(self.Ghh.copy())  # update當前最佳編碼的list
        self.fitness_Ghh_current_list.append(self.current_fitness)  # update當前最佳fitness的list

        self.bestGh = self.Ghh  # copy self.Ghh到最好的編碼self.bestGh
        self.best_fitness = self.current_fitness  # 最好的fitness
        self.best_fitness_list.append(self.best_fitness)

        step = 0
        while step <= MAX_GEN:
            self.swap(swapMode=swapMode)  # 產生neighbor(每次迭代要清空)
            fitness = []
            for temp in self.neighbor:
                temp_fitness = self.makespan(GN=temp)
                fitness.append(temp_fitness)

            # 將fitness.neighbor　由小到大排序
            fitness_sort = sorted(fitness)
            neighbor_sort = sorted(self.neighbor, key=lambda y: self.makespan(y))
            self.neighbor = []  # 將neighbor清空
            neighbor_sort_N = neighbor_sort[:N]  # 選取neighbor中fitness最好的前N個編碼
            fitness_sort_N = fitness_sort[:N]  # 選取neighbor中fitness最好的前N個fitness

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
                    if self.makespan(temp) < self.best_fitness:
                        self.current_fitness = self.makespan(temp)
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
                    self.current_fitness = self.makespan(temp)
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
        self.s5 = self.bestGh
        # region step2&3
        init = [self.s5[0], self.s5[1]]
        if self.makespan([self.s5[0], self.s5[1]]) > self.makespan([self.s5[1], self.s5[0]]):
            init = [self.s5[1], self.s5[0]]
        tmp = init
        for i in range(2, self.job_count):
            M = 999999999
            ind = 0
            for j in range(0, len(tmp) + 1):
                t = tmp.copy()
                t.insert(j, self.s5[i])
                if self.makespan(t) < M:
                    M = self.makespan(t)
                    ind = j
            tmp.insert(ind, self.s5[i])
        self.s6 = tmp
        #endregion
        print("s5:", self.s5)
        print("s5 makespan:", self.makespan(self.s5))
        print("s6:", self.s6)
        print("s6 makespan:", self.makespan(self.s6))

    def solver(self):
        self.NEH()
        self.SlopeInd()
        self.ts(MAX_GEN=600, length=7, N=self.job_count, Num=self.job_count, swapMode=3)



if __name__ == '__main__':
    df = read_csv('Flowshop_dataset/data_1.csv')
    Num = len(df["index"])
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    for i in range(0, Num):
        p1.append(df["p1"][i])
        p2.append(df["p2"][i])
        p3.append(df["p3"][i])
        p4.append(df["p4"][i])
        p5.append(df["p5"][i])
    p = []
    p.append(p1)
    p.append(p2)
    p.append(p3)
    p.append(p4)
    p.append(p5)

    fsh = FlowShop_Heuristic(Num=Num, p_time=p)
    fsh.solver()
    # print(fsh.makespan([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))






