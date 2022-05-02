import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.xlabel("dimension")
plt.ylabel("value")
# 塞入数据
# plt.ylim(0, 0.5)
a_x = [1, 2, 3, 4, 5]
a_y = [0.6435, 0.8411, 0.9201, 0.9312, 0.9346]
a_y_2 = [0.6123, 0.8232, 0.9187, 0.9308, 0.9332]
# plt.yticks 设置别名
plt.xticks([1, 2, 3, 4, 5], ['32', '64', '128', '256', '512'])
ax.plot(a_x, a_y, label='Micro-F1', color="r")
ax.plot(a_x, a_y_2, label='Macro-F1', color="b")

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
plt.scatter(
    a_x,
    a_y_2,
    marker="*",
    s=169,
    linewidths=3,
    color="red",
    zorder=10,
)

ax.grid()
ax.legend()
fig.savefig("classfy_contrast")
plt.show()
