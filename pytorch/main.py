import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

'''
polygon .. list of points (x,y) in counter clockwise order
guards  .. list of points (x,y)
'''
def optimize_AGP(polygon, guards):
	polygonFrom = torch.tensor(polygon, dtype=float)
	polygonTo = torch.roll(polygonFrom, -1, 0)
	guards = torch.tensor([[g] for g in guards], requires_grad=True, dtype=float)
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

	print("Saving animation...")
	fig, ax = plt.subplots()
	plt.plot(*zip(*polygonFrom.numpy(), polygonFrom[0]), c="black")
	scplot = ax.scatter([],[])

	def animate(i):
		ax.set_title(i)
		scplot.set_offsets(guard_states[i])
		return scplot,

	ani = FuncAnimation(fig, animate, interval=20, frames=len(guard_states), blit=True)
	ani.save('animation.gif', writer=PillowWriter(fps=30))

# Polygon and guards setup
if __name__ == "__main__":
	from examples import load_sawteeth, load_z, load_e1, load_e2
	optimize_AGP(*load_e2())