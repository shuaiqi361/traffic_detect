import numpy as np



total_num_frames = 181000

key_frame_time = 0.009
Extraction_time = 0.0024
Detection_time = 0.0138
Tracking_time = 0.0136
Counting_time = 0.0005


print('Current total execution time: ', total_num_frames * (Extraction_time + Detection_time + Tracking_time + Counting_time))

