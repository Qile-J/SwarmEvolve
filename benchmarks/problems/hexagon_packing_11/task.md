SETTING:
You are an expert computational geometer and optimization specialist focusing on hexagon packing problems.
Your task is to evolve a constructor function that generates an optimal arrangement of exactly 11 unit regular hexagons within a larger regular hexagon, maximizing 1/outer_hex_side_length (equivalently minimizing the outer hexagon's side length).

PROBLEM CONTEXT:
- Target: Beat the current state-of-the-art benchmark of 1/outer_hex_side_length = 1/3.930092 ≈ 0.2544
- Constraint: All 11 inner hexagons must be unit regular hexagons (side length = 1) that are fully contained within the outer hexagon with no overlaps
- Mathematical formulation: For hexagon i at position (xi, yi) with rotation θi:
  * Non-overlap: All pairs of inner hexagons must be disjoint
  * Containment: All vertices of inner hexagons must lie within the outer hexagon
  * Objective: maximize 1/R where R is the outer hexagon side length

PERFORMANCE METRICS:
1. **inv_outer_hex_side_length**: 1/outer_hex_side_length (PRIMARY OBJECTIVE - maximize)
2. **combined_score**: inverse_side_length / 0.2544 (progress toward beating SOTA)
3. **eval_time**: Execution time for full evaluation

EXECUTION ENVIRONMENT:
The program is run with Python 3.11, numpy and scipy, on a single CPU core, with a hard time limit of 60 seconds. Any search or optimization it performs must finish well within that limit, or the program scores zero.
