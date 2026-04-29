import random
import pygame
import math
import os
from modules.utils import config
from modules.base_agent import BaseAgent
from scenarios.features.mona.full_simulation.sim.task import task_colors

# Load agent configuration (Scenario Specific)
work_rate = config['agents']['work_rate']

# Load behavior tree
behavior_tree_xml = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/{config['agents']['behavior_tree_xml']}"

# Battery drain: 10% per 1000 pixels moved
_BATTERY_DRAIN_PER_PX = 3.0 / 1000.0

class Agent(BaseAgent):
    def __init__(self, agent_id, position, tasks_info, rotation=0, seed=None, initial_battery=None):
        super().__init__(agent_id, position, tasks_info, rotation)
        self.work_rate = work_rate

        self.task_amount_done = 0.0

        # Battery: use fixed value if provided, otherwise random 40~90%
        if initial_battery is not None:
            self.battery = float(initial_battery)
        else:
            rng = random.Random(seed) if seed is not None else random
            self.battery = rng.uniform(40.0, 90.0)
        self._prev_distance = 0.0

    def update(self):
        super().update()
        delta = self.distance_moved - self._prev_distance
        self.battery = max(0.0, self.battery - delta * _BATTERY_DRAIN_PER_PX)
        self._prev_distance = self.distance_moved

    def draw(self, screen):
        """Draw agent with circle and directional triangle."""
        # 1. Circle
        pygame.draw.circle(
            screen, (0, 0, 0),
            (int(self.position.x), int(self.position.y)),
            40,
            width=4
        )
        # 2. Triangle
        size = 10
        angle = self.rotation

        # Calculate the triangle points based on the current position and angle
        p1 = pygame.Vector2(self.position.x + size * math.cos(angle), self.position.y + size * math.sin(angle))
        p2 = pygame.Vector2(self.position.x + size * math.cos(angle + 2.5), self.position.y + size * math.sin(angle + 2.5))
        p3 = pygame.Vector2(self.position.x + size * math.cos(angle - 2.5), self.position.y + size * math.sin(angle - 2.5))

        self.update_color()
        pygame.draw.polygon(screen, self.color, [p1, p2, p3])

    def update_color(self):
        _ST_COLORS = {
            0: (30, 100, 220),   # Super Task 0 → blue
            1: (220, 50, 50),    # Super Task 1 → red
        }
        st_id = getattr(self, 'assigned_super_task_id', None)
        self.color = _ST_COLORS.get(st_id, (0, 0, 0))  # unassigned → black


