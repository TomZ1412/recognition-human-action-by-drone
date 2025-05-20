import numpy as np

train_data=np.load("./data/train_joint.npy")
train_label=np.load("./data/train_label.npy")
train_data=np.transpose(train_data,(0,4,2,3,1))

test_data=np.load("./data/val_joint.npy")
test_label=np.load("./data/val_label.npy")


test_data=np.transpose(test_data,(0,4,2,3,1))
np.savez('./save_3d_pose/V2.npz', x_train = train_data, y_train = train_label, x_test = test_data, y_test = test_label)


eval_data=np.load("./data/test_joint.npy")
eval_data=np.transpose(eval_data,(0,4,2,3,1))
eval_label=np.load("./data/test_label.npy")
print(eval_data.shape)
np.savez('./save_3d_pose/V1.npz', x_train = train_data, y_train = train_label, x_test = eval_data, y_test = eval_label)
