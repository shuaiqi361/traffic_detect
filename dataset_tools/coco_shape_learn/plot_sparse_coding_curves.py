import os
import numpy as np
import matplotlib.pyplot as plt

# plot the sparse coding curve: reconstruction errors (1 - mIoU) vs. number of basis, for different alpha values
# mask size 28 x 28
n_basis = np.array([32, 64, 128, 256])
c1 = np.array([1 - 0.9147, 1 - 0.9532, 1 - 0.9782, 1 - 0.9931])  # for alpha = 0.01
c2 = np.array([1 - 0.9144, 1 - 0.9529, 1 - 0.9798, 1 - 0.9935])  # for alpha = 0.05
c3 = np.array([1 - 0.9136, 1 - 0.9510, 1 - 0.9754, 1 - 0.9864])  # for alpha = 0.1
c4 = np.array([1 - 0.9092, 1 - 0.9415, 1 - 0.9562, 1 - 0.9656])  # for alpha = 0.2
c5 = np.array([1 - 0.8788, 1 - 0.8991, 1 - 0.9135, 1 - 0.9275])  # for alpha = 0.5

s1 = np.array([97.47, 97.05, 98.00, 98.11])  # sparsity for alpha = 0.01
s2 = np.array([94.93, 92.12, 87.25, 79.61])  # sparsity for alpha = 0.05
s3 = np.array([91.39, 85.26, 74.11, 57.02])  # sparsity for alpha = 0.1
s4 = np.array([83.56, 69.64, 46.65, 30.37])  # sparsity for alpha = 0.2
s5 = np.array([55.57, 34.50, 20.48, 12.91])  # sparsity for alpha = 0.5

# fig = plt.figure()
# plt.plot(n_basis, c1, color='red', marker='+', linewidth=2, markersize=8)
# plt.plot(n_basis, c2, color='green', marker='*', linewidth=2, markersize=8)
# plt.plot(n_basis, c3, color='blue', marker='o', linewidth=2, markersize=8)
# plt.plot(n_basis, c4, color='magenta', marker='x', linewidth=2, markersize=8)
# plt.plot(n_basis, c5, color='c', marker='^', linewidth=2, markersize=8)
# plt.plot(alpha_list, c6 / 256., color='yellow', marker='^', linewidth=2, markersize=6)

# plt.title('Reconstruction error lower bounds')
# plt.legend([r'$\lambda = 0.01$', r'$\lambda = 0.05$', r'$\lambda = 0.1$', r'$\lambda = 0.2$', r'$\lambda = 0.5$'])
# plt.xlabel('Number of basis functions')
# plt.ylabel(r'$1 - mIoU$')
# plt.show()
# plt.savefig(os.path.join(root_path, 'ped_find_alpha_v64_avg_norm.jpg'))


# plot the sparse coding curve: reconstruction errors (1 - mIoU) vs. sparsity weight alpha, for different number of basis functions
# mask size 28 x 28
Cs = np.concatenate([c1[:, None], c2[:, None], c3[:, None], c4[:, None], c5[:, None]], axis=1)
Ss = np.concatenate([s1[:, None], s2[:, None], s3[:, None], s4[:, None], s5[:, None]], axis=1)  # shape: 4, 5

sparsity_weights = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
fig = plt.figure()
plt.plot(sparsity_weights, Ss[0], color='red', marker='+', linewidth=2, markersize=8)
plt.plot(sparsity_weights, Ss[1], color='green', marker='*', linewidth=2, markersize=8)
plt.plot(sparsity_weights, Ss[2], color='blue', marker='o', linewidth=2, markersize=8)
plt.plot(sparsity_weights, Ss[3], color='magenta', marker='x', linewidth=2, markersize=8)

plt.title('Coefficient sparsity')
plt.legend(['32 Basis', '64 Basis', '128 Basis', '256 Basis'])
plt.xlabel('Sparsity weight ' + r'$\lambda$')
plt.ylabel('Coefficient activation percentage')
plt.show()
