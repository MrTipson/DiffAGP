import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Polygon and guards setup
th = 0.75 # tooth height
n_teeth = 5
polygonFrom = torch.tensor([*[[x, 0 if x % 2 == 0 else th] for x in range(n_teeth*2+1)], [n_teeth*2,1], [0,1]])
polygonTo = torch.roll(polygonFrom, -1, 0)

guards = torch.tensor([[[0.25,0.25]], [[4,0.25]]], requires_grad=True)
guard_states = [] # for animation

optimizer = torch.optim.AdamW([guards], lr=1e-2)

for _ in tqdm(range(1000)):
	optimizer.zero_grad()

	p1 = polygonFrom-guards
	p2 = polygonTo-guards
	crossp = p1[:,:,0]*p2[:,:,1] - p1[:,:,1]*p2[:,:,0]
	visibility = torch.relu(-crossp)
	loss = torch.sum(torch.cumprod(visibility, dim=0) + 0.5*visibility)

	loss.backward()
	optimizer.step()

	guard_states.append(guards.detach().squeeze(1).numpy().copy())

fig = plt.figure()
plt.plot(*zip(*polygonFrom.numpy(), polygonFrom[0]), c="black")
scplot = plt.scatter([],[])

def animate(i):
	scplot.set_offsets(guard_states[i])
	return scplot,

ani = FuncAnimation(fig, animate, interval=20, frames=len(guard_states), blit=True)
ani.save('animation.gif', writer=PillowWriter(fps=30))
