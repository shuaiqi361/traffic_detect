import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# import dash
# import dash_core_components as dcc
# import dash_html_components as html

import plotly.graph_objects as go

# define constants
# mu = 20.65
# sigma = 8.19
#
# x = np.arange(mu - 3 * sigma, mu + 2.5 * sigma, 0.1)  # range of x in spec
# x_all = np.arange(mu - 3 * sigma, mu + 3 * sigma, 0.1)  # entire range of x, both in and out of spec
# # mean = 0, stddev = 1, since Z-transform was calculated
# y = norm.pdf(x, mu, sigma)
# y2 = norm.pdf(x_all, mu, sigma)
#
# # build the plot
# fig, ax = plt.subplots()
# # plt.style.use('fivethirtyeight')
# ax.plot(x_all, y2)
#
# ax.fill_between(x_all, y2, 0, alpha=0.9, color='red')
# ax.fill_between(x, y, 0, alpha=0.8, color='green')
# plt.legend(['Accept', 'Reject'])
#
# ax.set_xlabel('Contour Mapping Measure Error', fontsize=12)
# # ax.set_yticklabels([])
# ax.set_title('MOI 9 LeaveOneOut CMM Error Distribution', fontsize=12)
# plt.grid()
# plt.savefig('/media/keyi/Data/Research/traffic/detection/traffic_detection/dataset_tools/aic/slides/normal_curve.png',
#             dpi=350, bbox_inches='tight')
# plt.show()


# plot t-distribution
# cmm_errors = [20.74, 15.68, 12.34, 25.12, 11.09, 18.65, 13.99, 15.2, 10.98, 16.53,
#               17.04, 10.11, 13.17, 19.87, 20.15, 11.34, 11.48, 20.53, 29.42, 9.41]
# num_sample = len(cmm_errors)
# zeros = [0] * num_sample
#
# mu = np.mean(cmm_errors)
# df = num_sample - 1
# std = np.std(cmm_errors)
#
# ci_95 = (1 - 0.95) / 2.
# ci_99 = (1 - 0.99) / 2.
#
# z_score_95 = std / np.sqrt(num_sample) * 2.093
# z_score_99 = std / np.sqrt(num_sample) * 2.861
# z_score_999 = std / np.sqrt(num_sample) * 3.883
# upper_margin_95 = mu + z_score_95
# upper_margin_99 = mu + z_score_99
# upper_margin_999 = mu + z_score_999
#
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=cmm_errors, y=zeros, marker=dict(
#             color='LightSkyBlue',
#             size=20,
#             line=dict(
#                 color='MediumPurple',
#                 width=2
#             )
#         ), mode='markers', name='CMM Measures'
# ))
# fig.add_trace(go.Scatter(
#     x=[mu], y=[0], mode='markers', marker_size=30, name='CMM mean'
# ))
# fig.add_trace(go.Scatter(
#     x=[upper_margin_95], y=[0], mode='markers', marker_size=30, name='95.0% CI'
# ))
# fig.add_trace(go.Scatter(
#     x=[upper_margin_99], y=[0], mode='markers', marker_size=30, name='99.0% CI'
# ))
# fig.add_trace(go.Scatter(
#     x=[upper_margin_999], y=[0], mode='markers', marker_size=30, name='99.9% CI'
# ))
#
# fig.update_xaxes(showgrid=False)
# fig.update_yaxes(showgrid=False,
#                  zeroline=True, zerolinecolor='black', zerolinewidth=5,
#                  showticklabels=False)
# fig.update_layout(height=500, width=1200, plot_bgcolor='white', title={
#         'text': "CMM LeaveOneOut Error Distribution",
#         'y': 0.9,
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'}, font=dict(
#         size=22
#     ))
# fig.show()



# use k-std to plot the data
cmm_errors = [20.74, 15.68, 12.34, 25.12, 11.09, 18.65, 13.99, 15.2, 10.98, 16.53,
              17.04, 10.11, 13.17, 19.87, 20.15, 11.34, 11.48, 20.53, 29.42, 9.41]
num_sample = len(cmm_errors)
zeros = [0] * num_sample

mu = np.mean(cmm_errors)
df = num_sample - 1
std = np.std(cmm_errors)

upper_margin_1 = mu + 1 * std
upper_margin_2 = mu + 2 * std
upper_margin_3 = mu + 3 * std
upper_margin_4 = mu + 4 * std


fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cmm_errors, y=zeros, marker=dict(
            color='LightSkyBlue',
            size=25,
            line=dict(
                color='MediumPurple',
                width=2
            )
        ), mode='markers', name='CMM Measures'
))
fig.add_trace(go.Scatter(
    x=[mu], y=[0], mode='markers', marker_size=35, name='CMM mean'
))
fig.add_trace(go.Scatter(
    x=[upper_margin_1], y=[0], mode='markers', marker_size=35, name='+1 standard deviation'
))
fig.add_trace(go.Scatter(
    x=[upper_margin_2], y=[0], mode='markers', marker_size=35, name='+2 standard deviation'
))
fig.add_trace(go.Scatter(
    x=[upper_margin_3], y=[0], mode='markers', marker_size=35, name='+3 standard deviation'
))
fig.add_trace(go.Scatter(
    x=[upper_margin_4], y=[0], mode='markers', marker_size=35, name='+4 standard deviation'
))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False,
                 zeroline=True, zerolinecolor='black', zerolinewidth=5,
                 showticklabels=False)
fig.update_layout(height=400, width=1500, plot_bgcolor='white', title={
        'text': "CMM LeaveOneOut Error Distribution",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, font=dict(
        size=25
    ))
fig.show()




