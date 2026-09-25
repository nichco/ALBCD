import numpy as np
import matplotlib.pyplot as plt
import os

UCSD_BLUE = "#00629B"
UCSD_GOLD = "#C69214"

fig, ax = plt.subplots(1, 2, figsize=(5.33, 2.25), constrained_layout=True)
ax = ax.flatten()


N = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
monolithic_time = np.array([9.86, 24.22, 112.345, 386.23, 755.009, 1504.773, 2785.930, 3508.666, 15755.439])
ALBCD_time = np.array([40.675, 58.815, 176.175, 265.7, 320.679, 399.415, 565.788, 588.668, 817.812])

# ax[0].semilogy(N, monolithic_time, marker='o', label='Monolithic', color=UCSD_BLUE)
# ax[0].semilogy(N, ALBCD_time, marker='o', label='ALBCD', color=UCSD_GOLD)
ax[0].semilogy(N, monolithic_time, label='Monolithic', linewidth=2, color=UCSD_BLUE)
ax[0].semilogy(N, ALBCD_time, label='ALBCD', linewidth=2, color=UCSD_GOLD)
ax[0].set_xlabel('N')
ax[0].set_xticks([2, 4, 6, 8, 10])
ax[0].set_ylabel('Time (s)')
ax[0].legend(loc='upper left')
ax[0].grid(color='blue', alpha=0.1, axis='y')


N = np.array([2, 4, 6, 8, 10])
monolithic_memory = np.array([11.063, 35.208, 73.571, 126.656, 193.573])
ALBCD_mem = np.array([5.006, 5.33, 5.641, 6.134, 6.657])

# ax[1].plot(N, ALBCD_mem, marker='o', label='ALBCD', color=UCSD_GOLD)
# ax[1].plot(N, monolithic_memory, marker='o', label='Monolithic', color=UCSD_BLUE)
ax[1].semilogy(N, monolithic_memory, label='Monolithic', linewidth=2, color=UCSD_BLUE)
ax[1].semilogy(N, ALBCD_mem, label='ALBCD', linewidth=2, color=UCSD_GOLD)
ax[1].set_xlabel('N')
ax[1].set_xticks([2, 4, 6, 8, 10])
ax[1].set_ylabel('Memory (MB)')
ax[1].legend(loc='upper left')
ax[1].grid(color='blue', alpha=0.1, axis='y')

plt.tight_layout()

HERE = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(HERE, "ucp_scaling.pdf"), bbox_inches='tight')
plt.show()