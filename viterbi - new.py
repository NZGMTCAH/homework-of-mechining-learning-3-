# -*- coding: utf-8 -*-
import numpy as np

# 隐马尔可夫模型λ=(A, B, pai)
# A是状态转移概率分布，状态集合Q的大小N=np.shape(A)[0]
# 从下给定A可知：Q={盒1, 盒2, 盒3}, N=3
A = np.array([[0.5, 0.1, 0.4],
              [0.3, 0.5, 0.2],
              [0.2, 0.2, 0.6]])
# B是观测概率分布，观测集合V的大小T=np.shape(B)[1]
# 从下面给定的B可知：V={红，白}，T=2
B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])
# pi是初始状态概率分布，初始状态个数=np.shape(pai)[0]
pi = np.array([[0.2],
                [0.3],
                [0.5]])

# 观测序列
O = np.array([[0],
              [1],
              [0],
              [0],
              [1],
              [0],
              [1],
              [1]])  # 0表示红色，1表示白，就是(红，白，红)观测序列

def hmm_viterbi(A,B,pi,O): #调用维特比算法公式
    T = len(O) #时刻个数
    N = len(A[0]) #状态个数
    delta = [[0]*N for _ in range(T)] #初始值
    psi = [[0]*N for _ in range(T)] #初始状态概率分布
    #step1: init ,初始化
    for i in range(N):
        delta[0][i] = pi[i]*B[i][O[0]]
        psi[0][i] = 0
    #step2: iter 在时刻 t 状态为 i 的所有单条路径中概率最大值
    for t in range(1,T):
        for i in range(N):
            temp,maxindex = 0,0
            for j in range(N):
                res = delta[t-1][j]*A[j][i]
                if res>temp:
                    temp = res #转移率
                    maxindex = j
            delta[t][i] = temp*B[i][O[t]] #在时刻 t 状态为 i 的所有单条路径中概率最大值
            psi[t][i] = maxindex #在时刻 t 状态为 i 的所有单条路径中,概率最大路径的第 t−1 个节点

    #step3: end 计算时刻 t，状态为 i ,最可能的隐藏状态最可能隐藏状态序列出现的概率
    p = max(delta[-1]) #最优路径的概率
    for i in range(N):
        if delta[-1][i] == p:
            i_T = i #最优路径的终点
    #step4：backtrack ,最优路径回溯，最优状态序列
    path = [0]*T
    i_t = i_T
    for t in reversed(range(T-1)):
        i_t = psi[t+1][i_t]
        path[t] = i_t
    path[-1] = i_T
    print(delta)
    print(psi)
    print(path)
    return delta,psi,path

hmm_viterbi(A,B,pi,O)

