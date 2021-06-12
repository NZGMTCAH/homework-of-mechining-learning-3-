import numpy as np
#前向算法
def fore_algorithm(A, B, p_i, o, T, N):
    # 设初值，alpha_1(i) = pt_(i)b_i(o(i))
    alpha = np.zeros((T, N))
    #初始化, αi(1)=πi∗bi(O1),前面的时刻观测到 O1,O2,...,Ot 的概率，
    for i in range(N):
        h = o[0]
        alpha[0][i] = p_i[i] * B[i][h]
    #递推,计算 αi(t)=(∑Nj=1αj(t−1)aji)bi(Ot)
    for t in range(T-1):
        h = o[t+1]
        for i in range(N):
            a = 0
            for j in range(N):
                a += (alpha[t][j] * A[j][i])
            alpha[t+1][i] = a * B[i][h]  #ai(t) 时刻 t，状态为 i ，观测序列为 O1,O2,...,Ot 的概率
    #终止,P(O|λ)=∑Ni=1αi(T)
    P = 0
    for i in range(N):
        P += alpha[T-1][i]  #状态转移，#计算P(O|λ)
    return P, alpha  


#后向算法
def back_algorithm(A, B, p_i, o, T, N):
    #top1 设置初值，beta_t(i)=1，初始化 βi(T)=1
    beta = np.ones((T, N))
    #top2 递推，计算beta(t)，计算 βi(t)=∑Nj=1aijbj(Ot+1)βj(t+1)
    for t in range(T-1):
        t = T - t - 2
        h = o[t + 1]
        h = int(h)

        for i in range(N):
            beta[t][i] = 0
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][h] * beta[t+1][j]   #βi(t) 时刻 t，状态为 i ，观测序列为 Ot+1,Ot+2,...,OT 的概率
 
    #top3 终止，计算P(O|λ)=∑Ni=1πibi(O1)βi(1)，时间复杂度 O(N2T)
    P = 0
    for i in range(N):
        h = o[0]
        h = int(h)
        P += p_i[i] * B[i][h] * beta[0][i] #状态转移，计算P(O|λ)
    return P, beta  #返回P(O|λ)以及概率

if __name__ == "__main__":
    T = 8    #8个时刻
    N = 3   #3 个状态
    #λ(A,B,π)
    A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]] #状态 1 2 3，状态转移概率 A
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]  # red white，发射概率 B
    pi = [0.2, 0.3, 0.5]  #初始概率分布 π
    O = ['红', '白', '红', '红', '白', '红', '白', '白'] #观测序列，
    #hmm问题
    #观测序列 O=O1O2...OT，模型 λ(A,B,π)，计算 P(O|λ)，即计算观测序列的概率
    #观测序列 O=O1O2...OT，模型 λ(A,B,π)，找到对应的状态序列 S
    #观测序列 O=O1O2...OT，找到模型参数 λ(A,B,π)，以最大化 P(O|λ)，
    o = np.zeros(T, np.int)
    for i in range(T):
        if O[i] == '白':
            o[i] = 1
        else:
            o[i] = 0
    PF, alpha = fore_algorithm(A, B, pi, o, T, N) #前向的状态序列 S概率以及观测序列O的概率
    PB, beta = back_algorithm(A, B, pi, o, T, N) #后向的状态序列 S概率以及观测序列O的概率
    print("PF:", PF, "PB:", PB) #前向后向的状态序列概率
    #P(i_4=q_3|O,\lambda) = alpah_4(3)* beta_4(3)
    P = alpha[4-1][3-1] * beta[4-1][3-1]  #使用前后向算法可以计算隐状态，记 γi(t)=P(st=i|O,λ) 表示时刻 t 位于隐状态 i 的概率
    print("前向后向概率计算可得 P(i4=q3|O,lambda)=", P / PF)  #使用前后向算法可以计算隐状态的概率	

