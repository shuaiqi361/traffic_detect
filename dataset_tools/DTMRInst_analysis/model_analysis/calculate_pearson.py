import numpy as np
from scipy import stats
import os
import math
import matplotlib.pyplot as plt


def weighted_avg_std(samples, weights):
    """
    :param samples: population samples
    :param weights: weights to be used
    Both are 1d arrays
    :return:
    weighted average and standard deviation
    """
    avg_val = np.average(samples, weights=weights)
    var_val = np.average((samples - avg_val) ** 2., weights=weights)

    return avg_val, np.sqrt(var_val)


def margin_of_error_95(n, std, z_val=1.96):
    """
    The default z-value is for 95% confidence interval
    :param n: number of samples
    :param std: population standard deviation
    :param z_val: Values are taken from standard normal distribution
    :return:
    MoE = z_val * std / sqrt(n),
    where n is the number of samples
    """

    return z_val * std / math.sqrt(n)


# root_folder = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/Res_50_DTCInst_068/digits/square_scramble/arrays'
# baseline_logits_file = os.path.join(root_folder, 'DTM_cls_b16_baseline.npy')
# scrambled_logits_file = os.path.join(root_folder, 'DTM_cls_b16_scrambled_05.npy')
#
#
# baseline_logits = np.load(baseline_logits_file)
# scrambled_logits = np.load(scrambled_logits_file)
#
# pearson_coef, _ = stats.pearsonr(baseline_logits, scrambled_logits)
#
#
# print('Pearson correlation coefficients : ', pearson_coef)


# calculate classwise pearson correlation coefficients
model_name = 'DTM'
num_block = 144
# root_folder = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/Res_50_DTCInst_068/digits/square_scramble/arrays_classwise'
root_folder = '/media/keyi/Data/Research/traffic/detection/AdelaiDet/experiments/' \
              'Res_50_DTMRInst_044/digits_046'

baseline_logits_file = os.path.join(root_folder, '{}_classwise_b{}_baseline.npy'.format(model_name, num_block))
scrambled_logits_file = os.path.join(root_folder, '{}_classwise_b{}_scrambled.npy'.format(model_name, num_block))

min_sample = 10

baseline_logits = np.load(baseline_logits_file, allow_pickle=True).item()
scrambled_logits = np.load(scrambled_logits_file, allow_pickle=True).item()

logits = []
ticks = []
labels = []
counters = []
total_counters = []


c = 1
for k, v in baseline_logits.items():
    total_counters.append(len(baseline_logits[k]))
    if k in scrambled_logits.keys() and len(v) >= min_sample:
        pearson_coef, _ = stats.pearsonr(baseline_logits[k], scrambled_logits[k])
        logits.append(pearson_coef)
        counters.append(len(baseline_logits[k]))
        ticks.append(c)
        c += 1
        labels.append(k)


# print('Pearson correlation coefficients : ', np.mean(logits), np.std(logits), 'from', len(logits), 'categories.')
#
# plt.figure()
# plt.bar(ticks, logits)
#
# plt.xticks(ticks, labels, rotation=85, fontsize=16)
# plt.yticks(fontsize=16)
# plt.title('{} logit coefficient with {} scrambled blocks'.format(model_name, num_block), fontsize=16)
# plt.show()


# # calculate weighted average
# logits = np.array(logits)
# counters = np.array(counters)
# w_avg = np.sum(logits * counters) / np.sum(counters)
# print('Pearson correlation coefficients : ', w_avg, 'from', len(logits), 'categories.')


# Return some info about the population
n_sample = np.sum(counters)
n_class = len(counters)
print('Total original number of objects:', np.sum(total_counters))
print('A total of {} objects are selected from {} categories.'.format(n_sample, n_class))

# calculate average
avg = np.mean(logits)
std = np.std(logits)
moe = margin_of_error_95(n_class, std)
print(avg, std, moe)

# calculate weighted average
w_avg, w_std = weighted_avg_std(logits, counters)
w_moe = margin_of_error_95(n_class, w_std)


print('average: {:.4f}, std: {:.4f}, moe: {:.4f}'.format(avg, std, moe))
print('weighted avg: {:.4f}, w.std: {:.4f}, w.moe: {:.4f}'.format(w_avg, w_std, w_moe))


