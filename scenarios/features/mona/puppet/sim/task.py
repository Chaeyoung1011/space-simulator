import pygame
import random
from modules.utils import config, generate_task_colors
dynamic_task_generation = config['tasks'].get('dynamic_task_generation', {})
max_generations = dynamic_task_generation.get('max_generations', 0) if dynamic_task_generation.get('enabled', False) else 0
tasks_per_generation = dynamic_task_generation.get('tasks_per_generation', 0) if dynamic_task_generation.get('enabled', False) else 0

task_colors = generate_task_colors(config['tasks']['quantity'] + tasks_per_generation*max_generations)

from modules.base_task import BaseTask

_fixed_amounts = config['tasks'].get('fixed_amounts', [])

class Task(BaseTask):
    def __init__(self, task_id, position):
        super().__init__(task_id, position)
        if task_id < len(_fixed_amounts):
            self.amount = float(_fixed_amounts[task_id])
        else:
            self.amount = random.uniform(config['tasks']['amounts']['min'], config['tasks']['amounts']['max'])
        self.radius = self.amount / config['simulation']['task_visualisation_factor']
        self.color = task_colors.get(self.task_id, (0, 0, 0))  # Default to black if task_id not found


    def draw(self, screen):
        self.radius = self.amount / config['simulation']['task_visualisation_factor']        
        if not self.completed:
            r = max(int(self.radius), 10)   # minimum 10 px for visibility
            x, y = int(self.position[0]), int(self.position[1])
            shape = getattr(self, 'draw_shape', 'circle')
            if shape == 'square':
                pygame.draw.rect(screen, self.color,
                                 pygame.Rect(x - r, y - r, r * 2, r * 2))
            else:
                pygame.draw.circle(screen, self.color, (x, y), r)

    def draw_task_id(self, screen):
        if not self.completed:
            text_surface = self.font.render(f"task_id {self.task_id}: {self.amount:.2f}", True, (250, 250, 250))
            screen.blit(text_surface, (self.position[0], self.position[1]))


