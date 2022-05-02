import matplotlib.pyplot as plt
import numpy as  np


train1=[0.06,0.31,0.53,0.71,0.81,0.84,0.87,0.92,0.94,0.95,0.958,0.96,0.9607,0.9615,0.9617]
train2=[0.10,0.23,0.31,0.42,0.54,0.63,0.75,0.83,0.88,0.91,0.93,0.935,0.936,0.937,0.9401]
train3=[0.08,0.19,0.28,0.39,0.48,0.56,0.67,0.75,0.82,0.84,0.87,0.90,0.91,0.9160,0.9201]





x_z=np.arange(0,15,1)



#创建画布
fig, ax = plt.subplots()
#塞入数据
# plt.xlim(0,9000)
ax.plot(x_z,train1, label='add CBAM,Res,D',color="r")
ax.plot(x_z,train2, label='add CBAM,Res',color="b")
ax.plot(x_z,train3, label='None',color="y")


ax.set(xlabel='epoch', ylabel='micro-f1',
       title='comparison')
ax.grid()
ax.legend()
fig.savefig("therr")
plt.show()