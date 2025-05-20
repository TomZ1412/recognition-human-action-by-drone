import torch
import numpy as np
import pickle
from collections import OrderedDict
import argparse

weights = [1.2, 0.46877578165654427, -0.05, -0.05, -0.05, 1.2, -0.05, 0.9207139493046791, 1.2, 1.2, -0.05, -0.05, 1.2, -0.05, -0.05, 1.2, 0.03775235783877531, 1.2, 1.2, 1.2, -0.05, 1.2, 1.2]

parser = argparse.ArgumentParser()
parser.add_argument('--mixformer_J_Score', default = './Model_inference/Mix_Former/output/skmixf__V1_J/score.npy')
parser.add_argument('--mixformer_B_Score', default = './Model_inference/Mix_Former/output/skmixf__V1_B/score.npy')
parser.add_argument('--mixformer_JM_Score', default = './Model_inference/Mix_Former/output/skmixf__V1_JM/score.npy')
parser.add_argument('--mixformer_BM_Score', default = './Model_inference/Mix_Former/output/skmixf__V1_BM/score.npy')
parser.add_argument('--mixformer_k2_Score', default = './Model_inference/Mix_Former/output/skmixf__V1_k2/score.npy')
parser.add_argument('--mixformer_k2M_Score', default = './Model_inference/Mix_Former/output/skmixf__V1_k2M/score.npy')
parser.add_argument('--ctrgcn_J3d_Score', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_J_3D/score.npy')
parser.add_argument('--ctrgcn_B3d_Score', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_B_3D/score.npy')
parser.add_argument('--ctrgcn_JM3d_Score', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_JM_3D/score.npy')
parser.add_argument('--ctrgcn_BM3d_Score', default = './Model_inference/Mix_GCN/output/ctrgcn_V1_BM_3D/score.npy')
parser.add_argument('--tdgcn_J2d_Score', default = './Model_inference/Mix_GCN/output/tdgcn_V1_J/score.npy')
parser.add_argument('--tdgcn_B2d_Score', default = './Model_inference/Mix_GCN/output/tdgcn_V1_B/score.npy')
parser.add_argument('--tdgcn_JM2d_Score', default = './Model_inference/Mix_GCN/output/tdgcn_V1_JM/score.npy')
parser.add_argument('--tdgcn_BM2d_Score', default = './Model_inference/Mix_GCN/output/tdgcn_V1_BM/score.npy')
parser.add_argument('--mstgcn_J2d_Score', default = './Model_inference/Mix_GCN/output/mstgcn_V1_J/score.npy')
parser.add_argument('--mstgcn_B2d_Score', default = './Model_inference/Mix_GCN/output/mstgcn_V1_B/score.npy')
parser.add_argument('--mstgcn_JM2d_Score', default = './Model_inference/Mix_GCN/output/mstgcn_V1_JM/score.npy')
parser.add_argument('--mstgcn_BM2d_Score', default = './Model_inference/Mix_GCN/output/mstgcn_V1_BM/score.npy')
parser.add_argument('--tegcn_Score', default = '/home/srt16/code/TE-GCN-main/work_dir/2996/epoch1_test_score.npy')
arg = parser.parse_args()

score=np.zeros((19,4599,155))
score[0]=np.load(arg.mixformer_J_Score)
score[1]=np.load(arg.mixformer_B_Score)
score[2]=np.load(arg.mixformer_JM_Score)
score[3]=np.load(arg.mixformer_BM_Score)
score[4]=np.load(arg.mixformer_k2_Score)
score[5]=np.load(arg.mixformer_k2M_Score)
score[6]=np.load(arg.ctrgcn_J3d_Score)
score[7]=np.load(arg.ctrgcn_B3d_Score)
score[8]=np.load(arg.ctrgcn_JM3d_Score)
score[9]=np.load(arg.ctrgcn_BM3d_Score)
score[10]=np.load(arg.tdgcn_J2d_Score)
score[11]=np.load(arg.tdgcn_B2d_Score)
score[12]=np.load(arg.tdgcn_JM2d_Score)
score[13]=np.load(arg.tdgcn_BM2d_Score)
score[14]=np.load(arg.mstgcn_J2d_Score)
score[15]=np.load(arg.mstgcn_B2d_Score)
score[16]=np.load(arg.mstgcn_JM2d_Score)
score[17]=np.load(arg.mstgcn_BM2d_Score)
score[18]=np.load(arg.tegcn_Score)

final_score=np.zeros((4599,155))
for i in range(19):
    final_score=final_score+score[i]*weights[i]
final_score=final_score/sum(weights)

np.save("pred.npy",final_score)
print(final_score.shape)