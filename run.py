#!/usr/bin/env python3
#
# usage: python3 run.py Airport.bmp

from PIL import Image
import numpy as np
import os
import sys

filename = sys.argv[1]
basename = os.path.splitext(os.path.basename(filename))[0]

X = np.array(Image.open(filename))

# u: left-singular vector
# s: singular value
# v: right-singular vector
u, s, v = np.linalg.svd(X)

os.mkdir('images')

for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
    ur = u[:, :k - 1]
    sr = np.diag(s[:k - 1])
    vr = v[:k - 1, :]
    Xk = np.dot(np.dot(ur, sr), vr)

    image = Image.fromarray(Xk)
    image = image.convert('RGB')
    image.save(f'images/{basename}-{k}.jpg')
