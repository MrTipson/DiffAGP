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