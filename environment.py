"""Physical environment: floor plan, fuel, fire and smoke fields, path planning.

All arrays are indexed [x, y] with y = row of FLOOR_PLAN (row 0 at the top).
The fields are updated synchronously once per model step, like the nutrient
field of the tumor model.
"""

from collections import deque
from functools import lru_cache

import numpy as np

# '#' wall, '.' floor, 'B' lab bench (blocks movement, burns),
# 'C' chemical cabinet (blocks movement, burns strongly, possible ignition),
# 'E' exit, 'S' extinguisher station / responder base.
FLOOR_PLAN = (
    "########################################",
    "#C.................#..................C#",
    "#..BBBBB..BBBBB....#...BBBBB..BBBBB....#",
    "#..................#...................#",
    "#..BBBBB..BBBBB....#...BBBBB..BBBBB....#",
    "#..................#...................#",
    "#..BBBBB..BBBBB....#...BBBBB..BBBBB....#",
    "#.................C#...................#",
    "#######..###############..##############",
    "E......................................E",
    "E..................S...................E",
    "#######..###############..##############",
    "#C.................#...................#",
    "#..BBBBB..BBBBB....#...BBBBB..BBBBB....#",
    "#..................#...................#",
    "#..BBBBB..BBBBB....#...BBBBB..BBBBB....#",
    "#..................#...................#",
    "#..BBBBB..BBBBB....#...BBBBB..BBBBB....#",
    "#.................C#..................C#",
    "########################################",
)
CORRIDOR_ROWS = (9, 10)
FUEL_BY_TILE = {"#": 0.0, ".": 0.3, "B": 1.0, "C": 2.0, "E": 0.0, "S": 0.0}
MOORE = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
VON_NEUMANN = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Environment:
    """Fire, fuel and smoke dynamics on the floor plan."""

    def __init__(self, p_spread=0.25, fire_growth=0.08, burn_rate=0.015,
                 smoke_emission=0.25, smoke_diffusion=0.5, smoke_substeps=4,
                 smoke_vent=0.03, floor_plan=FLOOR_PLAN):
        assert all(len(r) == len(floor_plan[0]) for r in floor_plan), "ragged floor plan"
        self.floor_plan = floor_plan
        self.width, self.height = len(floor_plan[0]), len(floor_plan)
        self.tiles = np.array([list(r) for r in floor_plan]).T
        self.walkable = np.isin(self.tiles, [".", "E", "S"])
        self.air = self.tiles != "#"
        self.fuel = np.vectorize(FUEL_BY_TILE.get)(self.tiles).astype(float)
        self.fire = np.zeros((self.width, self.height))
        self.smoke = np.zeros((self.width, self.height))
        self.ever_burned = np.zeros((self.width, self.height), dtype=bool)

        self.p_spread = p_spread
        self.fire_growth = fire_growth
        self.burn_rate = burn_rate
        self.smoke_emission = smoke_emission
        self.smoke_diffusion = smoke_diffusion
        self.smoke_substeps = smoke_substeps
        self.smoke_vent = smoke_vent

        self.exits = [tuple(map(int, c)) for c in np.argwhere(self.tiles == "E")]
        self.stations = [tuple(map(int, c)) for c in np.argwhere(self.tiles == "S")]
        self.cabinets = [tuple(map(int, c)) for c in np.argwhere(self.tiles == "C")]
        a = np.pad(self.air.astype(float), 1)
        self._air_neighbours = self._shift_sum(a)
        # Static fields (ignore fire): cached per goal
        self._static_cache = {}

    # ------------------------------------------------------------ geometry
    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _shift_sum(self, padded):
        w, h = self.width, self.height
        return sum(padded[1 + dx:1 + dx + w, 1 + dy:1 + dy + h] for dx, dy in VON_NEUMANN)

    def room_of(self, pos):
        """0-3 = labs (NW, NE, SW, SE), 4 = corridor. Used for state abstraction."""
        x, y = pos
        if y in CORRIDOR_ROWS or (y in (8, 11)):
            return 4
        return (0 if y < 9 else 2) + (0 if x < 19 else 1)

    # ------------------------------------------------------------ dynamics
    def ignite(self, cell, intensity=0.5):
        self.fire[cell] = intensity
        self.ever_burned[cell] = True

    def update(self, rng):
        burning = self.fire > 0
        self.fire[burning & (self.fuel > 0)] += self.fire_growth
        self.fire[burning & (self.fuel <= 0)] *= 0.7
        np.clip(self.fire, 0, 1, out=self.fire)
        self.fire[self.fire < 0.05] = 0.0

        # Spread to the 4 neighbours: P = 1 - prod(1 - p * I_neighbour), scaled by target fuel
        no_ignite = np.ones_like(self.fire)
        padded = np.pad(self.fire, 1)
        for dx, dy in VON_NEUMANN:
            no_ignite *= 1 - self.p_spread * padded[1 + dx:1 + dx + self.width, 1 + dy:1 + dy + self.height]
        p_ignite = (1 - no_ignite) * np.minimum(1.0, self.fuel)
        new = (self.fire == 0) & (self.fuel > 0) & (rng.random(self.fire.shape) < p_ignite)
        self.fire[new] = 0.2

        self.fuel = np.maximum(0.0, self.fuel - self.burn_rate * self.fire)
        self.ever_burned |= self.fire > 0

        # Smoke: emission, diffusion through air only (walls block), venting
        self.smoke += self.smoke_emission * self.fire
        has_n = self._air_neighbours > 0
        for _ in range(self.smoke_substeps):
            s = np.where(self.air, self.smoke, 0.0)
            avg = np.divide(self._shift_sum(np.pad(s, 1)), self._air_neighbours,
                            out=np.zeros_like(s), where=has_n)
            self.smoke = np.where(self.air, s + self.smoke_diffusion * (avg - s) * has_n, 0.0)
        self.smoke *= 1 - self.smoke_vent
        self.smoke[self.tiles == "E"] *= 0.5
        np.clip(self.smoke, 0, 1, out=self.smoke)

    # ------------------------------------------------------------ queries
    def fire_within(self, pos, r):
        x, y = pos
        return bool(self.fire[max(0, x - r):x + r + 1, max(0, y - r):y + r + 1].any())

    def fire_cells_within(self, pos, r):
        x, y = pos
        x0, y0 = max(0, x - r), max(0, y - r)
        cells = np.argwhere(self.fire[x0:x + r + 1, y0:y + r + 1] > 0) + [x0, y0]
        return [tuple(map(int, c)) for c in cells]

    def burning_cells(self):
        return [tuple(map(int, c)) for c in np.argwhere(self.fire > 0)]

    # ------------------------------------------------------------ path planning
    def distance_field(self, sources, avoid_fire=False):
        """Multi-source BFS over walkable cells (8-neighbour). Sources may be
        non-walkable (e.g. a burning cabinet): agents then stop next to them.
        Fields that ignore fire never change, so they are cached."""
        if not avoid_fire:
            key = tuple(sorted(sources))
            if key not in self._static_cache:
                if len(self._static_cache) > 4000:
                    self._static_cache.clear()
                self._static_cache[key] = self._bfs(key, None)
            return self._static_cache[key]
        return self._bfs(sources, self.fire > 0.3)

    def _bfs(self, sources, blocked):
        dist = np.full((self.width, self.height), np.inf)
        q = deque()
        for s in sources:
            dist[s] = 0
            q.append(s)
        walk = self.walkable if blocked is None else self.walkable & ~blocked
        w, h = self.width, self.height
        while q:
            x, y = q.popleft()
            d = dist[x, y] + 1
            for dx, dy in MOORE:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and walk[nx, ny] and dist[nx, ny] == np.inf:
                    dist[nx, ny] = d
                    q.append((nx, ny))
        return dist
