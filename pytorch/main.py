import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Polygon and guards setup
polygonFrom = torch.tensor([[0,0], [0,1], [2,1], [2,0], [1,0.5]])
polygonTo = torch.roll(polygonFrom, -1, 0)

guards = torch.tensor([[0.25,0.25]], requires_grad=True)
guard_states = [] # for animation

def project(a,b,c):
	# < 0: c is on the right side of a->b
	# > 0: c is on the left side of a->b
	return (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])

optimizer = torch.optim.AdamW([guards], lr=1e-2)

for _ in tqdm(range(100)):
	optimizer.zero_grad()

	loss = torch.sum(torch.relu(-project(polygonFrom, guards, polygonTo)))

	loss.backward()
	optimizer.step()

	guard_states.append(guards.detach().numpy().copy())

fig = plt.figure()
plt.plot(*zip(*polygonFrom.numpy(), polygonFrom[0]), c="black")
scplot = plt.scatter([],[])

def animate(i):
	scplot.set_offsets(guard_states[i])
	return scplot,

ani = FuncAnimation(fig, animate, interval=20, frames=len(guard_states), blit=True)
ani.save('animation.gif', writer=PillowWriter(fps=20))
