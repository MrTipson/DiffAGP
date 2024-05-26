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

res = 128
lr = 10e-1
epochs = 100
folder = "test"
if os.path.isfile(folder):
	raise f"File with name {folder} already exists"
elif not os.path.isdir(folder):
	os.mkdir(folder)

from tooth_scene import create_scene
guardcount = 1
scene_dict = create_scene(spp=32, res=res)
scene = mi.load_dict(scene_dict)
params = mi.traverse(scene)

gt = dr.clip(mi.render(scene, params),0,1)
plot_output(gt, f"{folder}/gt.png")
params["env.radiance.value"] = 0

opt = mi.ad.Adam(lr=lr, uniform=True)
opt['guard0.position'] = params['guard0.position']

light_intensity = 15
minloss, miniter = dr.inf, None
for it in range(epochs):
	params.update(opt)

	local_loss = []
	for i in range(guardcount):
		params[f"guard{i}.intensity.value"] = 0
	for i in range(guardcount):
		params[f"guard{i}.intensity.value"] = light_intensity 
		local_loss.append(1 - dr.clip(mi.render(scene, params, seed=it),0,1))
		with dr.suspend_grad():
			if i == 0:
				global_loss = local_loss[-1]*1
			else:
				global_loss *= local_loss[-1]
		params[f"guard{i}.intensity.value"] = 0
	for i in range(guardcount):
		params[f"guard{i}.intensity.value"] = light_intensity
	
	img = 1 - global_loss

	plot_output(img, f"{folder}/render.png")
	if it % 10 == 0:
		plot_output(img, f"{folder}/{it}.png")

	# L2 Loss
	#loss = dr.sqr(gt - img)
	loss = 0
	plot_output(global_loss*gt, f"{folder}/loss.png")
	for i in range(guardcount):
		loss += dr.mean(local_loss[i]*global_loss*gt)
	loss /= guardcount
	
	dr.backward(loss)
	opt.step()
	for i in range(guardcount):
		opt[f"guard{i}.position"][1] = params[f"guard{i}.position"][1]

	if loss[0]/minloss < 0.95:
		minloss = loss[0]
		miniter = it

	if it - miniter > 10:
		minloss = loss[0]
		miniter = it
		print(*np.array(params[f'guard{guardcount-1}.position']).flatten())
		scene_dict[f'guard{guardcount}'] = {
			'type': 'point',
			'intensity': {'type': 'spectrum', 'value': 0},	
			'position': (*np.array(params[f'guard{guardcount-1}.position']).flatten(),) 
		}
		for i in range(guardcount):
			scene_dict[f'guard{i}']['position'] = (*np.array(params[f'guard{i}.position']).flatten(),)
		guardcount += 1
		scene = mi.load_dict(scene_dict)
		params = mi.traverse(scene)
		params["env.radiance.value"] = 0
		for i in range(guardcount):
			opt[f'guard{i}.position'] = params[f'guard{i}.position']

	print(f"Iteration {1+it:03d}: Loss = {loss[0]:6f}")
	print(f"Positions = {[list(np.round(params[f'guard{i}.position'], 2).flatten()) for i in range(guardcount)]}", flush=True)
	if loss[0] < 1e-4:
		break
