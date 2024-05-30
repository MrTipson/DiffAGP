def load_sawteeth(teeth_height=0.75, teeth_number=5, max_height=1):
	polygon = []
	for x in range(teeth_number*2+1):
		if x % 2 == 0:
			polygon.append([x,0])
		else:
			polygon.append([x,teeth_height])
	polygon.append([teeth_number*2, max_height])
	polygon.append([0, max_height])
	guards = [[0.25,0.25], [4,0.25]]
	return polygon, guards

def load_z():
	polygon = [[0,0], [2,-1], [-2,-3], [2,-5], [1,-3], [3,0]]
	guards = [[0.25,0.25], [4,0.25]]
	return polygon, guards

# Solving the art gallery problem using gradient descent Fig. 2.12
def load_e1():
	polygon = [[1.83,-0.5], [3.52, -1.9], [6.44, -2.73], [7.82, -0.16], [8, 2.4], [9.6,1.09],
			[9.15,3.69], [6.32, 5.48], [3.31, 3.89], [2.3, 2.61], [4.88, 3],
			[2.98, 1.68], [1.87, 0.44], [4.75, 0.1]]
	guards = [[2,0], [2,0]]
	return polygon, guards

# Solving the art gallery problem using gradient descent Fig. 3.12
def load_e2():
	polygon = [[6.46,2.96], [7.58, -2.05], [4.78, -1.56], [7.62, -3.13], [11.18, -0.5], [11.39,3.24],
			[9.16,5.97], [6.08, 4.97], [3.34, 6.51], [3.33, 5.03], [3.37, 4.62],
			[4.06, 2.5], [1.76, 3.02], [4.13, 0.23]]
	guards = [[2,0], [2,0]]
	return polygon, guards