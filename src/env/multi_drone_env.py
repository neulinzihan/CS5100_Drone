import gym
import numpy as np
import random
from gym import spaces

class MultiDroneCoverageEnv(gym.Env):
    """
    A multi-agent environment for drone coverage on an NxN grid.
    Each drone sees only its own position (row, col) plus a coverage fraction
    (if desired). The difference here: each drone receives a truly local reward,
    i.e., the number of newly covered cells contributed by that drone alone.
    
    We define local_reward[i] = | coverage_i - union_{j != i} coverage_j |,
    normalized by total grid cells if we wish. This means the drone only
    gets credit for coverage that wouldn't exist if it weren't there.
    """

    def __init__(self, 
                 grid_size=10, 
                 num_drones=3,
                 coverage_radii=None,
                 max_steps=100,
                 partial_coverage=False,
                 normalize_rewards=True):
        """
        Args:
            grid_size (int): NxN grid dimension
            num_drones (int): how many drones
            coverage_radii (list or None): coverage radius for each drone
            max_steps (int): max steps in an episode
            partial_coverage (bool): if True, each drone's obs might contain local info
                                     (not implemented here, just a placeholder).
            normalize_rewards (bool): if True, reward is normalized by grid_size^2
        """
        super(MultiDroneCoverageEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.coverage_radii = coverage_radii if coverage_radii else [1]*num_drones
        self.max_steps = max_steps
        self.partial_coverage = partial_coverage
        self.normalize_rewards = normalize_rewards
        
        # Single-drone action space: 5 possible moves [up, down, left, right, stay].
        self.single_action_space = spaces.Discrete(5)
        
        # Multi-agent action space: a Dict of {drone_id: Discrete(5)}.
        self.action_space = spaces.Dict({
            i: self.single_action_space for i in range(self.num_drones)
        })

        # Single-drone observation: (row, col, coverage_frac)
        # coverage_frac is global coverage fraction for illustration
        # (If you want “truly local” observation, remove or modify this.)
        high = np.array([grid_size-1, grid_size-1, 1.0], dtype=np.float32)
        low  = np.array([0, 0, 0.0], dtype=np.float32)
        self.single_observation_space = spaces.Box(
            low=low, 
            high=high, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Multi-agent observation space
        self.observation_space = spaces.Dict({
            i: self.single_observation_space for i in range(self.num_drones)
        })
        
        self.reset()

    def reset(self):
        self.steps_taken = 0
        # Random initial positions
        self.drone_positions = []
        for _ in range(self.num_drones):
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            self.drone_positions.append((r, c))
        
        return self._get_obs()

    def step(self, action_dict):
        """
        action_dict: {drone_id: action} with each action in [0..4].
        We move each drone, compute coverage, then assign local rewards.
        """
        self.steps_taken += 1
        
        # Move drones
        new_positions = []
        for i in range(self.num_drones):
            a = action_dict[i]
            r, c = self.drone_positions[i]
            if a == 0:   # UP
                r = max(r-1, 0)
            elif a == 1: # DOWN
                r = min(r+1, self.grid_size-1)
            elif a == 2: # LEFT
                c = max(c-1, 0)
            elif a == 3: # RIGHT
                c = min(c+1, self.grid_size-1)
            # 4 = STAY does nothing
            new_positions.append((r,c))
        self.drone_positions = new_positions
        
        # Compute coverage for each drone individually
        coverage_sets = []
        for idx, (r,c) in enumerate(self.drone_positions):
            coverage_i = self._compute_coverage_for_one(r, c, self.coverage_radii[idx])
            coverage_sets.append(coverage_i)
        
        # Compute union coverage of all drones (to check done condition).
        coverage_all = set()
        for s in coverage_sets:
            coverage_all |= s  # set union
        coverage_frac = len(coverage_all) / (self.grid_size * self.grid_size)
        
        # Now compute local reward for each drone i:
        #   coverage_{-i} = union of all coverage_j for j != i
        #   local_reward[i] = | coverage_i - coverage_{-i} |
        # i.e. how many cells are newly covered exclusively by drone i
        reward_dict = {}
        for i in range(self.num_drones):
            coverage_minus_i = set()
            for j in range(self.num_drones):
                if j != i:
                    coverage_minus_i |= coverage_sets[j]
            
            # Drone i's newly contributed coverage:
            newly_contributed = coverage_sets[i] - coverage_minus_i
            local_reward = float(len(newly_contributed))
            if self.normalize_rewards:
                local_reward /= float(self.grid_size*self.grid_size)
            
            reward_dict[i] = local_reward
        
        # Done if coverage_frac >= 1.0 or step limit reached
        done = (coverage_frac >= 1.0 or self.steps_taken >= self.max_steps)
        
        obs_dict = self._get_obs(coverage_frac=coverage_frac)
        info = {}
        return obs_dict, reward_dict, done, info

    def _compute_coverage_for_one(self, r, c, radius):
        """Compute coverage set for a single drone at (r,c) with coverage radius."""
        covered = set()
        for rr in range(self.grid_size):
            for cc in range(self.grid_size):
                dist = np.sqrt((r - rr)**2 + (c - cc)**2)
                if dist <= radius:
                    covered.add((rr, cc))
        return covered

    def _get_obs(self, coverage_frac=None):
        """
        For each drone, return an array [row, col, coverage_frac].
        coverage_frac = global coverage fraction for demonstration,
                        though it's not truly local info. 
        """
        if coverage_frac is None:
            # If called at reset time, compute coverage fraction from positions
            coverage_all = set()
            for i,(r,c) in enumerate(self.drone_positions):
                coverage_all |= self._compute_coverage_for_one(r, c, self.coverage_radii[i])
            coverage_frac = len(coverage_all)/(self.grid_size*self.grid_size)
        
        obs_dict = {}
        for i, (r,c) in enumerate(self.drone_positions):
            obs_dict[i] = np.array([r, c, coverage_frac], dtype=np.float32)
        return obs_dict

    def render(self, mode='human'):
        """
        Minimal textual rendering:
        - '*' for covered cells
        - 'Di' for drone i
        """
        # 1) union coverage for the entire grid
        coverage_all = set()
        coverage_sets = []
        for idx, (r,c) in enumerate(self.drone_positions):
            cov = self._compute_coverage_for_one(r, c, self.coverage_radii[idx])
            coverage_sets.append(cov)
            coverage_all |= cov
        
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for (rr, cc) in coverage_all:
            grid[rr][cc] = "*"
        
        for i, (r,c) in enumerate(self.drone_positions):
            grid[r][c] = f"D{i}"
        
        for row in grid:
            print(" ".join(row))
        print()

    def close(self):
        pass
import gym
import numpy as np
import random
from gym import spaces

class MultiDroneCoverageEnv(gym.Env):
    """
    A multi-agent environment for drone coverage on an NxN grid.
    Each drone sees only its own position (row, col) plus a coverage fraction
    (if desired). The difference here: each drone receives a truly local reward,
    i.e., the number of newly covered cells contributed by that drone alone.
    
    We define local_reward[i] = | coverage_i - union_{j != i} coverage_j |,
    normalized by total grid cells if we wish. This means the drone only
    gets credit for coverage that wouldn't exist if it weren't there.
    """

    def __init__(self, 
                 grid_size=10, 
                 num_drones=3,
                 coverage_radii=None,
                 max_steps=100,
                 partial_coverage=False,
                 normalize_rewards=True):
        """
        Args:
            grid_size (int): NxN grid dimension
            num_drones (int): how many drones
            coverage_radii (list or None): coverage radius for each drone
            max_steps (int): max steps in an episode
            partial_coverage (bool): if True, each drone's obs might contain local info
                                     (not implemented here, just a placeholder).
            normalize_rewards (bool): if True, reward is normalized by grid_size^2
        """
        super(MultiDroneCoverageEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.coverage_radii = coverage_radii if coverage_radii else [1]*num_drones
        self.max_steps = max_steps
        self.partial_coverage = partial_coverage
        self.normalize_rewards = normalize_rewards
        
        # Single-drone action space: 5 possible moves [up, down, left, right, stay].
        self.single_action_space = spaces.Discrete(5)
        
        # Multi-agent action space: a Dict of {drone_id: Discrete(5)}.
        self.action_space = spaces.Dict({
            i: self.single_action_space for i in range(self.num_drones)
        })

        # Single-drone observation: (row, col, coverage_frac)
        # coverage_frac is global coverage fraction for illustration
        # (If you want “truly local” observation, remove or modify this.)
        high = np.array([grid_size-1, grid_size-1, 1.0], dtype=np.float32)
        low  = np.array([0, 0, 0.0], dtype=np.float32)
        self.single_observation_space = spaces.Box(
            low=low, 
            high=high, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Multi-agent observation space
        self.observation_space = spaces.Dict({
            i: self.single_observation_space for i in range(self.num_drones)
        })
        
        self.reset()

    def reset(self):
        self.steps_taken = 0
        # Random initial positions
        self.drone_positions = []
        for _ in range(self.num_drones):
            r = np.random.randint(0, self.grid_size)
            c = np.random.randint(0, self.grid_size)
            self.drone_positions.append((r, c))
        
        return self._get_obs()

    def step(self, action_dict):
        """
        action_dict: {drone_id: action} with each action in [0..4].
        We move each drone, compute coverage, then assign local rewards.
        """
        self.steps_taken += 1
        
        # Move drones
        new_positions = []
        for i in range(self.num_drones):
            a = action_dict[i]
            r, c = self.drone_positions[i]
            if a == 0:   # UP
                r = max(r-1, 0)
            elif a == 1: # DOWN
                r = min(r+1, self.grid_size-1)
            elif a == 2: # LEFT
                c = max(c-1, 0)
            elif a == 3: # RIGHT
                c = min(c+1, self.grid_size-1)
            # 4 = STAY does nothing
            new_positions.append((r,c))
        self.drone_positions = new_positions
        
        # Compute coverage for each drone individually
        coverage_sets = []
        for idx, (r,c) in enumerate(self.drone_positions):
            coverage_i = self._compute_coverage_for_one(r, c, self.coverage_radii[idx])
            coverage_sets.append(coverage_i)
        
        # Compute union coverage of all drones (to check done condition).
        coverage_all = set()
        for s in coverage_sets:
            coverage_all |= s  # set union
        coverage_frac = len(coverage_all) / (self.grid_size * self.grid_size)
        
        # Now compute local reward for each drone i:
        #   coverage_{-i} = union of all coverage_j for j != i
        #   local_reward[i] = | coverage_i - coverage_{-i} |
        # i.e. how many cells are newly covered exclusively by drone i
        reward_dict = {}
        for i in range(self.num_drones):
            coverage_minus_i = set()
            for j in range(self.num_drones):
                if j != i:
                    coverage_minus_i |= coverage_sets[j]
            
            # Drone i's newly contributed coverage:
            newly_contributed = coverage_sets[i] - coverage_minus_i
            local_reward = float(len(newly_contributed))
            if self.normalize_rewards:
                local_reward /= float(self.grid_size*self.grid_size)
            
            reward_dict[i] = local_reward
        
        # Done if coverage_frac >= 1.0 or step limit reached
        done = (coverage_frac >= 1.0 or self.steps_taken >= self.max_steps)
        
        obs_dict = self._get_obs(coverage_frac=coverage_frac)
        info = {}
        return obs_dict, reward_dict, done, info

    def _compute_coverage_for_one(self, r, c, radius):
        """Compute coverage set for a single drone at (r,c) with coverage radius."""
        covered = set()
        for rr in range(self.grid_size):
            for cc in range(self.grid_size):
                dist = np.sqrt((r - rr)**2 + (c - cc)**2)
                if dist <= radius:
                    covered.add((rr, cc))
        return covered

    def _get_obs(self, coverage_frac=None):
        """
        For each drone, return an array [row, col, coverage_frac].
        coverage_frac = global coverage fraction for demonstration,
                        though it's not truly local info. 
        """
        if coverage_frac is None:
            # If called at reset time, compute coverage fraction from positions
            coverage_all = set()
            for i,(r,c) in enumerate(self.drone_positions):
                coverage_all |= self._compute_coverage_for_one(r, c, self.coverage_radii[i])
            coverage_frac = len(coverage_all)/(self.grid_size*self.grid_size)
        
        obs_dict = {}
        for i, (r,c) in enumerate(self.drone_positions):
            obs_dict[i] = np.array([r, c, coverage_frac], dtype=np.float32)
        return obs_dict

    def render(self, mode='human'):
        """
        Minimal textual rendering:
        - '*' for covered cells
        - 'Di' for drone i
        """
        # 1) union coverage for the entire grid
        coverage_all = set()
        coverage_sets = []
        for idx, (r,c) in enumerate(self.drone_positions):
            cov = self._compute_coverage_for_one(r, c, self.coverage_radii[idx])
            coverage_sets.append(cov)
            coverage_all |= cov
        
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for (rr, cc) in coverage_all:
            grid[rr][cc] = "*"
        
        for i, (r,c) in enumerate(self.drone_positions):
            grid[r][c] = f"D{i}"
        
        for row in grid:
            print(" ".join(row))
        print()

    def close(self):
        pass
