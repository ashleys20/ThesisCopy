import math

def get_direction(curr, prev):
    return [curr[0] - prev[0], curr[1]-prev[1]]

#distance formula that takes 2 tuples and returns value
def dist_formula(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)