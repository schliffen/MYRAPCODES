#
#
#
import numpy as np
import pandas as pd
import operator
import pickle

data_dir = '/home/ali/CLionProjects/p_01/'

xls = pd.ExcelFile(data_dir + 'Russell_Sim_lambda.xlsx')

data_real = xls.parse(1)
lambdas = xls.parse(2)

# Uparameters
# data_u = pd.read_csv(data_dir + 'New_Sim.csv')
lam1 = lambdas['Lambda_1']
lam2 = lambdas['Lambda_2']
lam3 = lambdas['Lambda_3']
lam4 = lambdas['Lambda_4']
lam5 = lambdas['Lambda_5']
gama = [lambdas['Gamma'][i] for i in range(lambdas['Gamma'].shape[0]) if i % 5 == 0]
gamma_size = len(gama)
dmu_size = 5
lamba = np.array([lam1, lam2, lam3, lam4, lam5])
# ushape = lam1.shape[0]

# lm_1 = [lam1[i] for i in range(ushape) if i % 5 == 0]
# lm_2 = [lam2[i] for i in range(ushape) if i % 5 == 1]
# lm_3 = [lam3[i] for i in range(ushape) if i % 5 == 2]
# lm_4 = [lam4[i] for i in range(ushape) if i % 5 == 3]
# lm_5 = [lam5[i] for i in range(ushape) if i % 5 == 4]
# lm = np.array([lm_1, lm_2, lm_3, lm_4, lm_5])
# shape: dmu x gamma x dimension

def sort_dics(v1, v2):

    dv1 = {}
    dv2 = {}
    for i in range(v1.shape[0]):
        dv1.update({i+1:v1[i]})
        dv2.update({i+1:v2[i]})


    sorted_dv1 = sorted(dv1.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dv2 = sorted(dv2.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_dv1, sorted_dv2

def check_reanking(produced, truth):

    produced = np.array(produced)
    truth = np.array(truth)[:,1:]

    gama_comp = []
    for j1 in range(produced.shape[0]): # this means for each gamma

        tmp_tr = truth[j1]
        tmp_pr = produced[j1]

        tmp_tr, tmp_pr = sort_dics(tmp_tr, tmp_pr)

        pr_dic={}
        tr_dic = {}
        pr_rank_list = []
        tr_rank_list = []
        pr_dmu_list = []
        tr_dmu_list = []

        for i1 in range(len(tmp_pr)):
            if i1 == 0:
                pr_dic.update({tmp_pr[i1][0]: (i1+1, tmp_pr[i1][1]) })
                pr_rank_list.append(i1+1)
                pr_dmu_list.append(tmp_pr[i1][0])

                tr_dic.update({tmp_tr[i1][0]: (i1+1, tmp_tr[i1][1]) })
                tr_rank_list.append(i1+1)
                tr_dmu_list.append(tmp_tr[i1][0])

            else:

                if pr_dic[pr_dmu_list[-1]][1] == tmp_pr[i1][1]:
                    pr_dic.update({tmp_pr[i1][0]: (pr_rank_list[-1], tmp_pr[i1][1]) })
                    pr_rank_list.append(pr_rank_list[-1])
                    pr_dmu_list.append(tmp_pr[i1][0])

                else:
                    pr_dic.update({tmp_pr[i1][0]: (pr_rank_list[-1]+1, tmp_pr[i1][1]) })
                    pr_rank_list.append(pr_rank_list[-1]+1)
                    pr_dmu_list.append(tmp_pr[i1][0])

                if tr_dic[tr_dmu_list[-1]][1] == tmp_tr[i1][1]:
                    tr_dic.update({tmp_tr[i1][0]: (tr_rank_list[-1], tmp_tr[i1][1]) })
                    tr_rank_list.append(tr_rank_list[-1])
                    tr_dmu_list.append(tmp_tr[i1][0])

                else:
                    tr_dic.update({tmp_tr[i1][0]: (tr_rank_list[-1] + 1, tmp_tr[i1][1]) })
                    tr_rank_list.append(tr_rank_list[-1]+1)
                    tr_dmu_list.append(tmp_tr[i1][0])

        # assuming there are only 5 dmus
        dmu_rank_compare = []
        for dmui in range(5):
            if pr_dic[dmui+1][0] == tr_dic[dmui+1][0]:
                dmu_rank_compare.append(1)
            else:
                dmu_rank_compare.append(0)

        gama_comp.append(dmu_rank_compare)

    return gama_comp


def tfunc(lamba,   data_size, data_real):
    sim = []

    for k in range(data_size):
        x1 = np.array([np.random.uniform(12, 15, gamma_size), np.random.uniform(0.21, 0.48, gamma_size)]).transpose(1,0)
        y1 = np.array([np.random.uniform(138, 144, gamma_size), np.random.uniform(21, 22, gamma_size)]).transpose(1,0)
        x2 = np.array([np.random.uniform(10, 17, gamma_size), np.random.uniform(0.1, 0.7, gamma_size)]).transpose(1,0)
        y2 = np.array([np.random.uniform(143, 159, gamma_size), np.random.uniform(28, 35, gamma_size)]).transpose(1,0)
        x3 = np.array([np.random.uniform(4, 12, gamma_size), np.random.uniform(.16, 0.35, gamma_size)]).transpose(1,0)
        y3 = np.array([np.random.uniform(157, 198, gamma_size), np.random.uniform(21, 29, gamma_size)]).transpose(1,0)
        x4 = np.array([np.random.uniform(19, 22, gamma_size), np.random.uniform(0.12, 0.19, gamma_size)]).transpose(1,0)
        y4 = np.array([np.random.uniform(158, 181, gamma_size), np.random.uniform(21, 25, gamma_size)]).transpose(1,0)
        x5 = np.array([np.random.uniform(14, 15, gamma_size), np.random.uniform(.06, 0.09, gamma_size)]).transpose(1,0)
        y5 = np.array([np.random.uniform(157, 161, gamma_size), np.random.uniform(28, 40, gamma_size)]).transpose(1,0)
        x = np.array([x1, x2, x3, x4, x5]).transpose(1,0,2) # gamma X dmu X dim
        y = np.array([y1, y2, y3, y4, y5]).transpose(1,0,2) # gamma X dmu X dim


        dmu = []
        for iter in range(gamma_size):
            dmus = []

            for idmu in range(dmu_size):
                uy = 0
                vx = 0
                for idd in range(dmu_size):
                    uy += (x[iter][idd][0] * lamba[idd][5*iter + idmu])/x[iter][idmu][0] + (x[iter][idd][1] * lamba[idd][5*iter + idmu])/x[iter][idmu][1]
                    vx += (y[iter][idd][0] * lamba[idd][5*iter + idmu])/y[iter][idmu][0] + (y[iter][idd][1] * lamba[idd][5*iter + idmu])/y[iter][idmu][1]

                target = 1. if uy/vx >= 1. else uy/vx
                dmus.append(target)
            dmu.append(dmus)

        dmu = np.array(dmu)
        gama_ranking_compare = check_reanking(dmu, data_real)

        sim.append(gama_ranking_compare)

    return sim


data_size = 10000 # number of simulations

sim_result = tfunc(lamba, data_size, data_real)

sim_res = np.array(sim_result).transpose(2,1,0)

total_res = []

total_sims = sim_res.shape[2]
for i in range(sim_res.shape[0]): # iterate for dmus
    tmp_res = []
    for j in range(sim_res.shape[1]): # iterate over gamma
        #        for k in range(sim_result.shape[2]):

        tmp_res.append(sim_res[i,j].sum()/total_sims)


    total_res.append(tmp_res)

final_res = np.array(total_res).transpose(1,0)
pd.DataFrame(final_res).to_csv("final_results_ex3.csv")
average = final_res.sum(axis=1)/5
pd.DataFrame(average).to_csv("average_ex3.csv")
# with open('total_res.npy', 'wb') as f:
#     pickle.dump(total_res, f)


# checking the results for the simulations
print('done!')





