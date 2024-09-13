"""Geometric helper functions"""

import math
from random import random
import numpy as np


def euclidean_distance(p__, q__):
    """Returns the euclidean distance between a and b"""
    d_x = p__[0] - q__[0]
    d_y = p__[1] - q__[1]
    return math.sqrt(d_x * d_x + d_y * d_y)

def poisson_disc_sampling(r__, min_x=0, min_y=0, width=1, height=1, k__=30, dist=euclidean_distance, rand=random):
    """
    Calculate Poisson Disk Sampling based on Robert Bridson's algorithm.

    References:
        - https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
        - https://github.com/emulbreh/bridson
    """
    tau = 2 * math.pi
    cellsize = r__ / math.sqrt(2)

    grid_width = int(math.ceil(width / cellsize))
    grid_height = int(math.ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    def grid_coords(p__):
        """Return grid coordinates of p__"""
        return int(math.floor(p__[0] / cellsize)), int(math.floor(p__[1] / cellsize))

    def fits(p__, g_x, g_y):
        """
        Check if p__ is within distancer r__ of existing samples
        using grid to only consider nearby samples
        """
        yrange = list(range(max(g_y - 2, 0), min(g_y + 3, grid_height)))
        for x__ in range(max(g_x - 2, 0), min(g_x + 3, grid_width)):
            for y__ in yrange:
                g__ = grid[x__ + y__ * grid_width]
                if g__ is None:
                    continue
                if dist(p__, g__) <= r__:
                    return False
        return True

    p__ = width * rand(), height * rand()
    queue = [p__]
    grid_x, grid_y = grid_coords(p__)
    grid[grid_x + grid_y * grid_width] = p__

    while queue:
        q_i = int(rand() * len(queue))
        q_x, q_y = queue[q_i]
        queue[q_i] = queue[-1]
        queue.pop()
        for __ in range(k__):
            alpha = tau * rand()
            d__ = r__ * math.sqrt(3 * rand() + 1)
            p_x = q_x + d__ * math.cos(alpha)
            p_y = q_y + d__ * math.sin(alpha)
            if not (0 <= p_x < width and 0 <= p_y < height):
                continue
            p__ = (p_x, p_y)
            grid_x, grid_y = grid_coords(p__)
            if not fits(p__, grid_x, grid_y):
                continue
            queue.append(p__)
            grid[grid_x + grid_y * grid_width] = p__
    return np.array([np.array(p__) + [min_x, min_y] for p__ in grid if p__ is not None])

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    References:
        - https://stackoverflow.com/a/6802723

    Args:
        axis: Three dimensional list which specifies the rotation axis.
        theta: Rotation angle.

    Returns:
        Numpy array which represents the rotation matrix.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
        ]
    )

def largest_area_in_histogram(histogram):
    """
    This function calulates maximum
    rectangular area under given
    histogram with n bars

    References:
        - https://www.geeksforgeeks.org/largest-rectangle-under-histogram/
    """
    # Create an empty stack. The stack
    # holds indexes of histogram[] list.
    # The bars stored in the stack are
    # always in increasing order of
    # their heights.
    stack = list()
    left = -1
    right = -1
    height = -1
    max_area = 0 # Initalize max area
    # Run through all bars of
    # given histogram
    index = 0
    while index < len(histogram):
        # If this bar is higher
        # than the bar on top
        # stack, push it to stack
        if (not stack) or (histogram[stack[-1]] <= histogram[index]):
            stack.append(index)
            index += 1
        # If this bar is lower than top of stack,
        # then calculate area of rectangle with
        # stack top as the smallest (or minimum
        # height) bar.'i' is 'right index' for
        # the top and element before top in stack
        # is 'left index'
        else:
            # pop the top
            top_of_stack = stack.pop()
            # Calculate the area with
            # histogram[top_of_stack] stack
            # as smallest bar
            idx_diff = (index - stack[-1] - 1) if stack else index
            area = histogram[top_of_stack] * idx_diff
            # update max area, if needed
            if area > max_area:
                max_area = area
                height = histogram[top_of_stack] - 1
                right = index - 1
                left = stack[-1] + 1 if stack else 0
    # Now pop the remaining bars from
    # stack and calculate area with
    # every popped bar as the smallest bar
    while stack:
        top_of_stack = stack.pop()
        idx_diff = (index - stack[-1] - 1) if stack else index
        area = histogram[top_of_stack] * idx_diff
        if area > max_area:
            max_area = area
            height = histogram[top_of_stack] - 1
            right = index - 1
            left = stack[-1] + 1 if stack else 0
    # Return maximum area under
    # the given histogram
    return max_area, left, right, height

def max_rectangle(grid):
    """
    Find the rectangle of maximum area size using multiple calls of largest_area_in_histogram

    References:
        - https://www.geeksforgeeks.org/maximum-size-rectangle-binary-sub-matrix-1s/
    """
    nb_cells = len(grid)
    counts = np.zeros((nb_cells, nb_cells))
    for row in range(nb_cells):
        for col in range(nb_cells):
            if grid[row, col] > 0:
                counts[row, col] = 1
    result, left, right, height = largest_area_in_histogram(counts[0])
    bottom = 0
    for row in range(nb_cells):
        for col in range(nb_cells):
            nb_points = grid[row, col]
            if nb_points > 0:
                counts[row, col] = counts[row - 1, col] + 1
        local_result, local_left, local_right, local_height = largest_area_in_histogram(counts[row])
        if local_result > result:
            result = local_result
            left = local_left
            right = local_right
            height = local_height
            bottom = row - local_height
    max_index = [height + bottom, right]
    min_index = [bottom, left]

    return min_index, max_index

def ar_h(u):
    if u >= 1: return np.pi
    elif u > -1: return np.pi - np.arccos(u) + u * np.sqrt(1 - u**2)
    else: return 0

def ar_quad(u, v):
    if u**2 + v**2 <= 1: return (ar_h(u) + ar_h(v))/2 - np.pi/4 + u*v
    elif u <= -1 or v <= -1: return 0
    elif u >= 1 and v >= 1: return np.pi
    elif u >= 1: return ar_h(v)
    elif v >= 1: return ar_h(u)
    elif u >= 0 and v >= 0: return ar_h(u) + ar_h(v) - np.pi
    elif u >= 0 and v <= 0: return ar_h(v)
    elif u <= 0 and v >= 0: return ar_h(u)
    else: return 0

def ar_rect(x0, y0, x1, y1):
    return ar_quad(x0, y0) + ar_quad(x1, y1) - ar_quad(x0, y1) - ar_quad(x1, y0)

def ar_area(r, xc, yc, x0, y0, x1, y1):
    """
    Calculate the intersection area between a circle with center point (xc, yc) and radius r
    and a rectangle with the minimum and maximum corner points (x0, y0) and (x1, y2).
    """
    return r**2 * ar_rect((x0-xc)/r, (y0-yc)/r, (x1-xc)/r, (y1-yc)/r)


