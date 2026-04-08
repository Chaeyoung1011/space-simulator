from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from modules.base_sim import BaseSim
from modules.utils import ResultSaver, config, generate_positions
from scenarios.features.mona.full_simulation.sim.task import Task
from scenarios.features.mona.full_simulation.sim.agent import Agent


def generate_tasks(task_quantity=None, task_id_start=0, seed=None):
    if task_quantity is None:
        task_quantity = config['tasks']['quantity']
    task_locations = config['tasks']['locations']

    tasks_positions = generate_positions(task_quantity,
                                        task_locations['x_min'],
                                        task_locations['x_max'],
                                        task_locations['y_min'],
                                        task_locations['y_max'],
                                        radius=task_locations['non_overlap_radius'],
                                        seed=seed)

    tasks = [Task(idx + task_id_start, pos) for idx, pos in enumerate(tasks_positions)]
    return tasks


def generate_agents(tasks_info, seed=None):
    agent_quantity = config['agents']['quantity']
    agent_locations = config['agents']['locations']
    fixed_positions = config['agents'].get('fixed_positions', [])
    fixed_positions = [tuple(p) for p in fixed_positions]  # list → tuple
    fixed_angles = config['agents'].get('fixed_angles', [])  # radians

    num_fixed  = min(len(fixed_positions), agent_quantity)
    num_random = agent_quantity - num_fixed

    if num_random > 0:
        random_positions = generate_positions(
                                      num_random,
                                      agent_locations['x_min'],
                                      agent_locations['x_max'],
                                      agent_locations['y_min'],
                                      agent_locations['y_max'],
                                      radius=agent_locations['non_overlap_radius'],
                                      seed=seed)
    else:
        random_positions = []

    agents_positions = fixed_positions[:num_fixed] + random_positions

    agents = []
    for idx, pos in enumerate(agents_positions):
        angle = fixed_angles[idx] if idx < len(fixed_angles) else 0
        agents.append(Agent(idx, pos, tasks_info, rotation=angle))
    return agents


class Sim(BaseSim):
    def __init__(self, config):
        super().__init__(config)

        # Set `generate_tasks` function for dynamic task generation
        self.generate_tasks = generate_tasks
        
        # Set data recording
        self.result_saver = ResultSaver(config)

        # Initialise
        self.reset()

    def reset(self):
        super().reset()

        # Initialize agents and tasks
        self.tasks = generate_tasks(seed=self.seed)
        self.agents = generate_agents(self.tasks, seed=self.seed)
        
        # Initialize data recording
        self.data_records = []
        self.convergence_records = []

    def save_results(self):
        # Save gif
        if self.save_gif and self.rendering_mode == "Screen":        
            self.recording = False
            print("Recording stopped.")
            self.result_saver.save_gif(self.frames)          
     

        # Save time series data
        if self.save_timewise_result_csv:
            csv_file_path = self.result_saver.save_to_csv("timewise", self.data_records, ['time', 'wall_clock', 'agents_total_distance_moved', 'agents_total_task_amount_done', 'remaining_tasks', 'tasks_total_amount_left'])
            self.result_saver.plot_timewise_result(csv_file_path)
        
        # Save agent-wise data            
        if self.save_agentwise_result_csv:        
            variables_to_save = ['agent_id', 'task_amount_done', 'distance_moved']
            agentwise_results = self.result_saver.get_agentwise_results(self.agents, variables_to_save)                        
            csv_file_path = self.result_saver.save_to_csv('agentwise', agentwise_results, variables_to_save)
            
            self.result_saver.plot_boxplot(csv_file_path, variables_to_save[1:])

        # Save convergence data
        if self.save_timewise_result_csv and self.convergence_records:
            conv_csv = self.result_saver.save_to_csv('convergence', self.convergence_records, ['time', 'convergence_pct'])
            self._plot_convergence_result(conv_csv)

        # Save yaml
        if self.save_config_yaml:
            self.result_saver.save_config_yaml()

    def record_convergence_result(self):
        total = len(self.agents)
        if total == 0:
            self.convergence_records.append([self.simulation_time, 0.0])
            return

        # 각 agent의 bundle 전체 task_id 수집
        all_bundles = [
            [task.task_id for task in agent.planned_tasks]
            for agent in self.agents
        ]

        # 전체 bundle에서 각 task_id 등장 횟수
        all_task_ids = [tid for bundle in all_bundles for tid in bundle]
        counts = Counter(all_task_ids)

        # bundle이 비어있거나 bundle 내 모든 task가 중복 없으면 수렴된 agent
        converged_count = sum(
            1 for bundle in all_bundles
            if len(bundle) == 0 or all(counts[tid] == 1 for tid in bundle)
        )

        # 개별 agent 수렴 비율 (enable_global_convergence 여부와 무관)
        converged_pct = converged_count / total * 100.0

        self.convergence_records.append([self.simulation_time, converged_pct])

    def _plot_convergence_result(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        plt.figure(figsize=(10, 4))
        plt.plot(df['time'], df['convergence_pct'], color='steelblue')
        plt.xlabel('Time (s)')
        plt.ylabel('% of Converged Agents')
        enable_global_convergence = config['decision_making']['CBBA'].get('enable_global_convergence', False)
        title = ('System Global Convergence Over Time' if enable_global_convergence
                 else 'Agent Convergence Over Time (based on full bundle uniqueness)')
        plt.title(title)
        plt.ylim(0, 105)
        plt.grid(True)
        plt.tight_layout()
        img_path = csv_file_path.replace('.csv', '.png')
        plt.savefig(img_path)
        plt.close()

    def record_timewise_result(self):
        agents_total_distance_moved = sum(agent.distance_moved for agent in self.agents)
        agents_total_task_amount_done = sum(agent.task_amount_done for agent in self.agents)
        remaining_tasks = len([task for task in self.tasks if not task.completed])
        tasks_total_amount_left = sum(task.amount for task in self.tasks)
        
        self.data_records.append([
            self.simulation_time,
            self.wall_clock_elapsed,
            agents_total_distance_moved,
            agents_total_task_amount_done,
            remaining_tasks,
            tasks_total_amount_left
        ])        
                  