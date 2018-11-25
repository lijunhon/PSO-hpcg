import os
import numpy as np
#import matplotlib.pyplot as plt
 
class PSO(object):
    def __init__(self, population_size, max_steps):
        self.w = 0.6  # 惯性权重
        self.c1 = self.c2 = 2
        self.population_size = population_size  # 粒子群数量
        self.dim = 4  # 搜索空间的维度
        #self.dim = 2  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.x_bound = [8, 12]  # 解空间范围 TODO 更改这部分，表示nz,nz,ny范围
        self.n_bound = [1, 4]   #TODO 更改这部分，表示进程数范围
        self.core = 24    #机器核心数量 TODO 更改这部分
        self.x1 = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.population_size, self.dim - 1))  # 初始化粒子群位置
        self.x2 = np.random.randint(1,4,size=(self.population_size, 1))  
        self.x = np.hstack((self.x1, self.x2)) #拼接
        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度
        fitness = self.calculate_fitness(self.x, self.core)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmax(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度
        best_pg = np.rint(self.pg)
        print("全局最优gflop:%.5f ,平均gflop:%.5f"%(self.global_best_fitness, np.mean(fitness)))
        print("全局最优nx值:"+str(int(best_pg[0])) + \
                    " ,全局最优ny值:" + str(int(best_pg[1])) + \
                    " ,全局最优nz值:" + str(int(best_pg[2])) + \
                    " ,全局最优进程数:" + str(int(best_pg[3])) + \
                    " ,全局最优线程数:" + str(int(self.core / int(best_pg[3]))))
        print('\n')
 
    def calculate_fitness(self, x, core):
        x = np.rint(x)
        y = np.zeros(len(x))
        for i in range(len(x)):
            print("=========目前的参数==========")
            os.environ['nx'] = str(int(x[i][0]))
            os.environ['ny'] = str(int(x[i][1]))
            os.environ['nz'] = str(int(x[i][2]))
            os.environ['n'] = str(int(x[i][3]))
            os.environ['OMP_NUM_THREADS'] = str(int(core / int(x[i][3])))
            
            print("nx值:"+str(int(x[i][0])) + \
                    " ,ny值:" + str(int(x[i][1])) + \
                    " ,nz值:" + str(int(x[i][2])) + \
                    " ,进程数:" + str(int(x[i][3])) + \
                    " ,线程数:" + str(int(core / int(x[i][3]))))
            os.system('mpirun -n $n ./xhpcg --nx $nx --ny $ny --nz $nz >> a.txt')
            gflop = os.popen('bash result.sh').read().strip('\n')
            y[i] = float(gflop)
            print("第%d个个体计算所得值:%f"%(i,y[i]))
        return y
 
    def evolve(self):
        #fig = plt.figure()
        for step in range(self.max_steps):
            print("第%d轮寻优"%step)
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w*self.v+self.c1*r1*(self.p-self.x)+self.c2*r2*(self.pg-self.x)
            self.x = self.v + self.x
            #plt.clf()
            #plt.scatter(self.x[:, 0], self.x[:, 1], s=30, color='k')
            #plt.xlim(self.x_bound[0], self.x_bound[1])
            #plt.ylim(self.x_bound[0], self.x_bound[1])
            #plt.pause(0.01)
            fitness = self.calculate_fitness(self.x, self.core)
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更大的fitness，所以更新全局最优fitness和位置
            if np.max(fitness) > self.global_best_fitness:
                self.pg = self.x[np.argmax(fitness)]
                self.global_best_fitness = np.max(fitness)
            best_pg = np.rint(self.pg)
            print('全局最优gflop: %.5f, 平均最优gflop: %.5f' % (self.global_best_fitness, np.mean(fitness)))
            print("全局最优nx值:"+str(int(best_pg[0])) + \
                    " ,全局最优ny值:" + str(int(best_pg[1])) + \
                    " ,全局最优nz值:" + str(int(best_pg[2])) + \
                    " ,全局最优进程数:" + str(int(best_pg[3])) + \
                    " ,全局最优线程数:" + str(int(self.core / int(best_pg[3]))))
            print('\n')
 
pso = PSO(2, 10)
pso.evolve()
#plt.show()

