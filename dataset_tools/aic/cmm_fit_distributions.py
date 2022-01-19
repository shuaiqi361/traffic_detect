import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt

cmm_errors = [20.74, 15.68, 12.34, 25.12, 21.09, 18.65, 13.99, 25.2, 20.98, 16.53,
              17.04, 10.11, 13.17, 19.87, 20.15, 11.34, 29.48, 20.53, 29.42, 25.41,
              30.80, 32.75, 17.90, 12.56, 24.55, 19.87, 20.51, 22.98, 31.87, 35.42,
              32.90, 40.18, 39.24, 36.54, 26.57, 19.87, 16.29, 27.16, 28.14, 21.44,
              22.08, 23.62, 30.10, 28.87, 26.54]

y = np.array(cmm_errors)
x = np.arange(len(y))
size = len(y)

# plt.hist(y)
# plt.show()
#
# y_df = pd.DataFrame(y, columns=['Data'])
# y_df.describe()

# standardize the data
sc = StandardScaler()
yy = y.reshape(-1, 1)
sc.fit(yy)

y_std = sc.transform(yy)
y_std = y_std.flatten()

dist_names = ['beta',
              'expon',
              'lognorm',
              'chi',
              'genexpon']

# Set up empty lists to store results
chi_square = []
p_values = []
log_ll = []

# Set up 20 bins for chi-square test
# Observed data will be approximately evenly distributed across all bins
percentile_bins = np.linspace(0, 50, 21)
percentile_cutoffs = np.percentile(y_std, percentile_bins)
observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
cum_observed_frequency = np.cumsum(observed_frequency)

# Loop through candidate distributions
for distribution in dist_names:
    # Set up distribution and get fitted distribution parameters
    dist = getattr(scipy.stats, distribution)
    param = dist.fit(y_std)

    # Obtain the KS test P statistic, round it to 5 decimal places
    p = scipy.stats.kstest(y_std, distribution, args=param)[1]
    p = np.around(p, 5)
    p_values.append(p)

    # Get expected counts in percentile bins
    # This is based on a 'cumulative distrubution function' (cdf)
    cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2],
                          scale=param[-1])
    pdf_fitted = dist.pdf(y_std, *param[:-2], loc=param[-2],
                          scale=param[-1])
    log_ll.append(np.mean(np.log(pdf_fitted)))

    expected_frequency = []
    for bin in range(len(percentile_bins) - 1):
        expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)

    # calculate chi-squared
    expected_frequency = np.array(expected_frequency) * size
    cum_expected_frequency = np.cumsum(expected_frequency)
    ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
    chi_square.append(ss)

# Collate results and sort by goodness of fit (best at top)
results = pd.DataFrame()
results['Distribution'] = dist_names
results['chi_square'] = chi_square
results['p_value'] = p_values
results['log_ll'] = log_ll
results.sort_values(['chi_square'], inplace=True)

# Report results
print('\nDistributions sorted by goodness of fit:')
print('----------------------------------------')
print(results)

# Divide the observed data into 20 bins for plotting (this can be changed)
number_of_bins = 20
bin_cutoffs = np.linspace(np.percentile(y, 0), np.percentile(y, 99.99), number_of_bins)

# Create the plot
h = plt.hist(y, bins=bin_cutoffs, color='0.75', edgecolor='white', linewidth=1)

# Get the top three distributions from the previous phase
number_distributions_to_plot = 3
dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

# Create an empty list to store fitted distribution parameters
parameters = []

# Loop through the distributions ot get line fit and paraemters

for dist_name in dist_names:
    # Set up distribution and store distribution paraemters
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y)
    parameters.append(param)

    # Get line for each distribution (and scale to match observed data)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
    scale_pdf = np.trapz(h[0], h[1][:-1]) / np.trapz(pdf_fitted, x)
    pdf_fitted *= scale_pdf

    # Add the line to the plot
    plt.plot(pdf_fitted, label=dist_name)

    # Set the plot x axis to contain 99% of the data
    # This can be removed, but sometimes outlier data makes the plot less clear
    # plt.xlim(0, np.percentile(y, 99.9))

# Add legend and display plot
plt.xlabel('CMM Errors', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Fitted Distributions from CMM Measures', fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Store distribution paraemters in a dataframe (this could also be saved)
dist_parameters = pd.DataFrame()
dist_parameters['Distribution'] = (
    results['Distribution'].iloc[0:number_distributions_to_plot])
dist_parameters['Distribution parameters'] = parameters

# Print parameter results
print('\nDistribution parameters:')
print('------------------------')

for index, row in dist_parameters.iterrows():
    print('\nDistribution:', row[0])
    print('Parameters:', row[1])

