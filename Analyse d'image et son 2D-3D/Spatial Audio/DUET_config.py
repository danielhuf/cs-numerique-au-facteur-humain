#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

fs = 16000

q_m1 = np.array([-1e-2, 0]) # 1st microphone cartesian coordinates
q_m2 = np.array([1e-2, 0]) # 2nd microphone cartesian coordinates
q_s = np.array([[.5, .3],
               [0, .5],
               [-.5, .3]]).T # source cartesian coordinates, shape (2, 3)


d1 = np.linalg.norm(q_m1[:,np.newaxis] - q_s, axis=0) # sources-to-1st microphone distance
d2 = np.linalg.norm(q_m2[:,np.newaxis] - q_s, axis=0) # sources-to-2nd microphone distance

c = 344 # sound velocity in m/s

a = d1/d2 # inter-microphone level ratio
delta_sec = (d2 - d1)/c # time difference of arrival in seconds
delta = delta_sec*fs
