# 说明文档

## 一、文件夹目录

### 1. requirements----依赖配置文件
### 2. ./code/Model_inference/Mix_Former(Mix_GCN)/output/../log && ./code/TE-GCN-main/workdir/2996/log  ----训练日志
### 3. ./code/Model_inference/Mix_Former(Mix_GCN)/config/ && ./code/TE-GCN-main/config  ----训练配置文件夹
### 4. code/----源代码

## 二、复现步骤

### 1. 将国赛数据集 (名为*data*的文件夹) 并放在  **code/Process_data** 目录下   
### 2. 安装所需依赖项（含torchlight）
### 3. 进入code/Process_data，运行：
```bash
python process_3d
```
### 5. 进入code目录，运行代码
```bash
cp -r ./Process_data/save_3d_pose Model_inference/Mix_GCN/dataset
cp -r ./Process_data/save_3d_pose Model_inference/Mix_Former/dataset
cp -r ./Process_data/data TE-GCN-main
```
### 6. 进入code/Model_Inference/Mix_GCN目录，运行代码
```bash
python main.py -config ./config/ctrgcn_V1_J_3d.yaml --phase test --save-score True --device 0 
```
```bash
python main.py -config ./config/ctrgcn_V1_B_3d.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/ctrgcn_V1_JM_3d.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/ctrgcn_V1_BM_3d.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mstgcn_V1_B.yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mstgcn_V1_J.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mstgcn_V1_BM.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mstgcn_V1_JM.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/tdgcn_V1_B.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/tdgcn_V1_J.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/tdgcn_V1_BM.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/tdgcn_V1_JM.
yaml --phase test --save-score True --device 0
```
### 7. 进入code/Model_Inference/Mix_Former目录，运行代码
```bash
python main.py -config ./config/mixformer_V1_JM.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mixformer_V1_J.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mixformer_V1_BM.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mixformer_V1_B.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mixformer_V1_k2M.
yaml --phase test --save-score True --device 0
```
```bash
python main.py -config ./config/mixformer_V1_k2.
yaml --phase test --save-score True --device 0
```
### 8.进入code/TE-GCN-main 目录，运行代码
### 9. 进入code目录，运行代码
```bash
python pred.py
```

### 在code/目录下得到 pred.npy 置信度文件
