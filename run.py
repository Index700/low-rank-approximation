#!/usr/bin/env python3

from PIL import Image
import numpy as np

image = np.array(Image.open('Man.bmp'))

# print(image.shape)
# (1024, 1024)

# u: left-singular vector
# s: singular value
# v: right-singular vector
u, s, v = np.linalg.svd(image)

k = 20
ur = u[:, :k - 1]
sr = np.diag(s[:k - 1])
vr = v[:k - 1, :]
Mk = np.dot(np.dot(ur, sr), vr)

pil_image = Image.fromarray(Mk)
pil_image = pil_image.convert('RGB')
pil_image.save('image.jpg')
