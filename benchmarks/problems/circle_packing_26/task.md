You are an expert mathematician specializing in circle packing problems and computational geometry. The best known result for the sum of radii when packing 26 circles in a unit square is 2.635.

Key directions to explore:
1. The optimal arrangement likely involves variable-sized circles
2. A pure hexagonal arrangement may not be optimal due to edge effects
3. The densest known circle packings often use a hybrid approach
4. The optimization routine is critically important - simple physics-based models with carefully tuned parameters
5. Consider strategic placement of circles at square corners and edges
6. Adjusting the pattern to place larger circles at the center and smaller at the edges
7. The math literature suggests special arrangements for specific values of n
8. You can use the scipy optimize package (e.g. LP or SLSQP) to optimize the radii given center locations and constraints

Be creative and try to find a new solution better than the best known result.

The program must define run_packing(), returning (centers, radii, sum_of_radii) with centers of shape (26, 2) and radii of shape (26,). Circles must lie inside the unit square and must not overlap.

EXECUTION ENVIRONMENT:
The program is run with Python 3.11, numpy and scipy, on a single CPU core, with a hard time limit of 60 seconds. Any search or optimization it performs must finish well within that limit, or the program scores zero.
