import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

'''
polygon .. list of points (x,y) in counter clockwise order
guards  .. list of points (x,y)
'''
def optimize_AGP(polygon, guards, a=0.1):
	polygonFrom = torch.tensor(polygon, dtype=float)
	polygonTo = torch.roll(polygonFrom, -1, 0)
	guards = torch.tensor([[g] for g in guards], requires_grad=True, dtype=float)
	guard_states = [] # for animation

	optimizer = torch.optim.AdamW([guards], lr=1e-2)

	progress_bar = tqdm(range(2000))
	for it in progress_bar:
		optimizer.zero_grad()

		p1 = polygonFrom-guards
		p2 = polygonTo-guards
		crossp = p1[:,:,0]*p2[:,:,1] - p1[:,:,1]*p2[:,:,0]
		visibility = torch.relu(-crossp)
		insideness_loss = torch.sum(torch.cumprod(visibility, dim=0) + visibility)

		p1 = (guards-polygonFrom).unsqueeze(0)
		p2 = (polygonFrom.unsqueeze(1)-polygonFrom.unsqueeze(0)).unsqueeze(1)
		p3 = (guards-polygonTo).unsqueeze(0)
		p4 = (polygonFrom.unsqueeze(1)-polygonTo.unsqueeze(0)).unsqueeze(1)
		crossp1 = p1[:,:,:,0]*p2[:,:,:,1] - p1[:,:,:,1]*p2[:,:,:,0]
		crossp2 = p3[:,:,:,0]*p4[:,:,:,1] - p3[:,:,:,1]*p4[:,:,:,0]
		p1,p2,p3,p4 = p1,p3,p2,p4 # we omit negation since it cancels out
		crossp3 = p1[:,:,:,0]*p2[:,:,:,1] - p1[:,:,:,1]*p2[:,:,:,0]
		crossp4 = p3[:,:,:,0]*p4[:,:,:,1] - p3[:,:,:,1]*p4[:,:,:,0]
		obstructed = torch.relu(-crossp1*crossp2)*torch.relu(-crossp3*crossp4)

		obstruction_loss = torch.sum(torch.prod(torch.log(1+torch.sum(obstructed, axis=2)), dim=1))

		loss = a*obstruction_loss + insideness_loss
		progress_bar.set_description(f"{loss:.2f}, {obstruction_loss:.2f}")

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
	ani.save('animation.gif', writer=PillowWriter(fps=50))

# Polygon and guards setup
if __name__ == "__main__":
	from examples import load_sawteeth, load_z, load_e1, load_e2
	optimize_AGP(*load_e2())