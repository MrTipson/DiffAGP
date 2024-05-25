import matplotlib.pyplot as plt
import os

import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T

def plot_output(render, filename):
	plt.figure(figsize=(5, 5))
	plt.imshow(mi.util.convert_to_bitmap(render))
	plt.axis('off')
	plt.savefig(filename)
	plt.close()

res = 256
lr = 1e-1
epochs = 100
folder = "test"
if os.path.isfile(folder):
	raise f"File with name {folder} already exists"
elif not os.path.isdir(folder):
	os.mkdir(folder)

scene = mi.load_file("scene.xml", res=str(res))
params = mi.traverse(scene)

gt = dr.clip(mi.render(scene, params),0,1)
plot_output(gt, f"{folder}/gt.png")
params["env.radiance.value"] = 0

opt = mi.ad.Adam(lr=lr, uniform=True)
opt['guard.position'] = params['guard.position']

for it in range(epochs):
	params.update(opt)

	img = dr.clip(mi.render(scene, params, seed=it),0,1)

	plot_output(img, f"{folder}/render.png")
	if it % 10 == 0:
		plot_output(img, f"{folder}/{it}.png")

	# L2 Loss
	plot_output(dr.sqr(gt - img), f"{folder}/loss.png")
	loss = dr.mean(dr.sqr(gt - img))
	
	dr.backward(loss)
	opt.step()

	print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}: Position = {np.round(params['guard.position'], 2)}", flush=True)
	if loss[0] < 1e-4:
		break
