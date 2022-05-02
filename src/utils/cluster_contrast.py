import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.xlabel("dimension")
plt.ylabel("NMI")
# 塞入数据
# plt.ylim(0, 0.5)
a_x = [1, 2, 3, 4, 5]
a_y = [0.4110, 0.7159, 0.8391, 0.8412, 0.8422]
# plt.yticks 设置别名
plt.xticks([1, 2, 3, 4, 5], ['32', '64', '128', '256', '512'])
ax.plot(a_x, a_y, label='NMI', color="r")

# ax.plot(x_z, f_loss_16, label='Macro-F1', color="b")


plt.scatter(
    a_x,
    a_y,
    marker="x",
    s=169,
    linewidths=3,
    color="red",
    zorder=10,
)


ax.grid()
ax.legend()
fig.savefig("cluster_contrast")
plt.show()
