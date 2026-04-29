from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import pygame
import socket, json, threading
from modules.base_sim import BaseSim
from modules.utils import ResultSaver, config, generate_positions
from scenarios.features.mona.puppet.sim.task import Task
from scenarios.features.mona.puppet.sim.agent import Agent
from scenarios.features.mona.puppet.sim.super_task import SuperTask


def generate_tasks(task_quantity=None, task_id_start=0, seed=None,
                   fixed_positions_override=None, fixed_amounts_override=None):
    if task_quantity is None:
        task_quantity = config['tasks']['quantity']
    task_locations = config['tasks']['locations']

    fixed_positions = fixed_positions_override if fixed_positions_override is not None \
                      else config['tasks'].get('fixed_positions', [])
    fixed_positions = [tuple(p) for p in fixed_positions]

    num_fixed  = min(len(fixed_positions), task_quantity)
    num_random = task_quantity - num_fixed

    if num_random > 0:
        random_positions = generate_positions(num_random,
                                              task_locations['x_min'],
                                              task_locations['x_max'],
                                              task_locations['y_min'],
                                              task_locations['y_max'],
                                              radius=task_locations['non_overlap_radius'],
                                              seed=seed)
    else:
        random_positions = []

    task_positions = fixed_positions[:num_fixed] + random_positions
    tasks = [Task(idx + task_id_start, pos) for idx, pos in enumerate(task_positions)]

    # Apply fixed amounts if provided
    if fixed_amounts_override is not None:
        for idx, task in enumerate(tasks):
            if idx < len(fixed_amounts_override):
                task.amount = float(fixed_amounts_override[idx])
                task.radius = task.amount / config['simulation']['task_visualisation_factor']

    return tasks


def _setup_agent_groups(agents: list, super_tasks: list) -> None:
    """Inject super-task references into each agent (all agents see all super tasks)."""
    for agent in agents:
        agent.super_tasks_info             = super_tasks   # shared list reference
        agent.super_task_message_to_share  = {}
        agent.super_task_messages_received = []
        agent.super_task_dwell_complete    = False


# Visual style per super-task index: (draw_shape, color)
_ST_TASK_VISUAL = {
    0: ('square', (30,  100, 220)),   # ST_0 members → blue square
    1: ('circle', (220,  50,  50)),   # ST_1 members → red circle
}

def generate_super_tasks(tasks: list) -> list:
    """Build SuperTask objects from the `super_tasks.groups` config section.

    Each group entry is a list of task indices (0-based within *tasks*).
    Also stamps each member task with its visual shape/colour.
    Returns an empty list if `super_tasks` is absent from config.
    """
    from scenarios.features.mona.puppet.sim.super_task import size_for_count
    groups = config.get('super_tasks', {}).get('groups', [])

    # Parse all groups first to compute a shared (max) size
    parsed = []
    for grp in groups:
        if isinstance(grp, dict):
            indices    = grp.get('tasks', [])
            max_agents = grp.get('max_agents', None)
        else:
            indices    = grp
            max_agents = None
        member_tasks = [tasks[i] for i in indices if i < len(tasks)]
        parsed.append((indices, max_agents, member_tasks))

    shared_size = max(size_for_count(len(m)) for _, _, m in parsed if m) if parsed else None

    super_tasks = []
    for st_id, (indices, max_agents, member_tasks) in enumerate(parsed):
        if member_tasks:
            super_tasks.append(SuperTask(st_id, member_tasks, max_agents=max_agents, size=shared_size))
            shape, color = _ST_TASK_VISUAL.get(st_id, ('circle', (100, 100, 100)))
            for t in member_tasks:
                t.draw_shape = shape
                t.color      = color
    return super_tasks


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
    fixed_batteries = config['agents'].get('fixed_batteries', [])

    agents = []
    for idx, pos in enumerate(agents_positions):
        angle = fixed_angles[idx] if idx < len(fixed_angles) else 0
        initial_battery = fixed_batteries[idx] if idx < len(fixed_batteries) else None
        battery_seed = (seed * 1000 + idx) if seed is not None else None
        agents.append(Agent(idx, pos, tasks_info, rotation=angle, seed=battery_seed,
                            initial_battery=initial_battery))
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

        # WhyCon UDP listener — receives real MONA positions and updates sim agents
        t = threading.Thread(target=self._listen_whycon_udp, daemon=True)
        t.start()

        # Initialize agents and tasks
        self.tasks = generate_tasks(seed=self.seed)
        self.agents = generate_agents(self.tasks, seed=self.seed)

        # Initialize super tasks
        self.super_tasks = generate_super_tasks(self.tasks)

        # Inject group / super-task info into each agent
        _setup_agent_groups(self.agents, self.super_tasks)
        # Initialize data recording
        self.data_records = []
        self.battery_records = {agent.agent_id: [] for agent in self.agents}

        # on-completion generation state
        self._all_arrived_wall = None
        dynamic_task_generation = config.get('tasks', {}).get('dynamic_task_generation', {})
        self._completion_delay = dynamic_task_generation.get('delay_seconds', 1.0)
        self._arrive_threshold = config.get('tasks', {}).get('threshold_done_by_arrival', 1.0)

    def _all_agents_arrived(self):
        """모든 agent가 assigned task에 도착했는지 확인 (IsArrivedAtTarget 조건과 동일)."""
        if not self.agents:
            return False
        for agent in self.agents:
            task_id = agent.assigned_task_id
            if task_id is None:
                return False
            task = next((t for t in agent.tasks_info if t.task_id == task_id), None)
            if task is None:
                return False
            dist = (pygame.Vector2(task.position) - agent.position).length()
            if dist >= task.radius + self._arrive_threshold:
                return False
        return True

    def generate_tasks_if_needed(self):
        """Override: 모든 agent가 IsArrivedAtTarget 상태를 delay_seconds 초 유지하면 새 task 생성."""
        if self.generation_count >= self.max_generations:
            return

        if self._all_agents_arrived():
            if self._all_arrived_wall is None:
                self._all_arrived_wall = self.wall_clock_elapsed

            elapsed = self.wall_clock_elapsed - self._all_arrived_wall
            if elapsed >= self._completion_delay:
                seed = self.seed + self.generation_count + 1 if self.seed is not None else None
                dynamic_cfg = config.get('tasks', {}).get('dynamic_task_generation', {})
                generations_cfg = dynamic_cfg.get('generations', None)
                if generations_cfg and self.generation_count < len(generations_cfg):
                    gen_cfg = generations_cfg[self.generation_count]
                    dyn_positions = gen_cfg.get('fixed_positions', None)
                    dyn_amounts   = gen_cfg.get('fixed_amounts', None)
                else:
                    dyn_positions = dynamic_cfg.get('fixed_positions', None)
                    dyn_amounts   = dynamic_cfg.get('fixed_amounts', None)
                new_tasks = self.generate_tasks(
                    task_quantity=self.tasks_per_generation,
                    task_id_start=0,
                    seed=seed,
                    fixed_positions_override=dyn_positions,
                    fixed_amounts_override=dyn_amounts,
                )

                # Replace tasks in-place (agents share the same list reference)
                self.tasks.clear()
                self.tasks.extend(new_tasks)
                self.generation_count += 1
                self._all_arrived_wall = None

                # Rebuild super tasks so they reference the new task objects
                self.super_tasks = generate_super_tasks(self.tasks)
                _setup_agent_groups(self.agents, self.super_tasks)

                for agent in self.agents:
                    agent.task_amount_done = 0.0
                    agent.assigned_task_id = None
                    agent.planned_tasks = []
                    agent.blackboard = {}
                    self._reset_agent_decision_maker(agent)

                if self.rendering_mode != "None":
                    print(f"[{self.simulation_time:.2f}] Replaced with {self.tasks_per_generation} new tasks (Generation {self.generation_count}). Agents reset.")
        else:
            self._all_arrived_wall = None

    def _reset_agent_decision_maker(self, agent):
        """BT 트리를 순회하여 decision_maker(CBBA 등)를 새로 초기화."""
        def walk(node):
            if hasattr(node, 'decision_maker'):
                node.decision_maker.__init__(agent)
            if hasattr(node, 'children'):
                for child in node.children:
                    walk(child)
        if hasattr(agent, 'tree') and agent.tree is not None:
            walk(agent.tree)

    def draw_tasks(self):
        """Draw super tasks (background layer) then regular tasks on top."""
        for st in self.super_tasks:
            st.draw(self.screen)
            st.draw_id(self.screen)
        super().draw_tasks()

    def _listen_whycon_udp(self):
        """WhyCon이 보내는 실제 MONA 위치를 받아 시뮬레이터 에이전트 위치에 반영."""
        udp_port = int(getattr(self, "config", {}).get("mona", {}).get("udp_port", 9999))
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", udp_port))
        print(f"[Env] WhyCon UDP listening on 0.0.0.0:{udp_port}")
        while True:
            try:
                data, _ = sock.recvfrom(2048)
                msg = json.loads(data)
                agent_id = int(msg.get("agent_id", 0))
                x = float(msg["x"])
                y = float(msg["y"])
                yaw = msg.get("yaw", None)
                if 0 <= agent_id < len(self.agents):
                    ag = self.agents[agent_id]
                    if hasattr(ag, "set_position"):
                        ag.set_position(x, y, yaw)
            except Exception as e:
                print(f"[WhyCon UDP Error] {e}")

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

        # Save battery graphs
        if self.save_timewise_result_csv and self.battery_records:
            self._plot_battery_graphs()

        # Save yaml
        if self.save_config_yaml:                
            self.result_saver.save_config_yaml()   

    def _plot_battery_graphs(self):
        colors = plt.cm.tab10.colors  # up to 10 distinct colors
        base = self.result_saver.result_file_path.rsplit('.', 1)[0]

        # Collect all records into a flat list for CSV
        rows = []
        for agent in self.agents:
            for (wall_clock, distance_moved, battery) in self.battery_records.get(agent.agent_id, []):
                rows.append({
                    'agent_id':       agent.agent_id,
                    'wall_clock':     wall_clock,
                    'distance_moved': distance_moved,
                    'battery':        battery,
                })

        # Save CSV
        csv_path = base + '_battery.csv'
        df = pd.DataFrame(rows, columns=['agent_id', 'wall_clock', 'distance_moved', 'battery'])
        df.to_csv(csv_path, index=False)
        print(f"[Battery] Saved: {csv_path}")

        # Graph 1: battery % vs distance moved (px)
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        for i, agent in enumerate(self.agents):
            agent_df = df[df['agent_id'] == agent.agent_id]
            if agent_df.empty:
                continue
            ax1.plot(agent_df['distance_moved'], agent_df['battery'],
                     color=colors[i % len(colors)], label=f'Agent {agent.agent_id}')
        ax1.set_xlabel('Distance Moved (px)')
        ax1.set_ylabel('Battery (%)')
        ax1.set_title('Battery vs Distance Moved')
        ax1.set_ylim(0, 105)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True)
        fig1.tight_layout()
        path1 = base + '_battery_vs_distance.png'
        fig1.savefig(path1)
        plt.close(fig1)
        print(f"[Battery] Saved: {path1}")

        # Graph 2: battery % vs wall-clock time (s)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for i, agent in enumerate(self.agents):
            agent_df = df[df['agent_id'] == agent.agent_id]
            if agent_df.empty:
                continue
            ax2.plot(agent_df['wall_clock'], agent_df['battery'],
                     color=colors[i % len(colors)], label=f'Agent {agent.agent_id}')
        ax2.set_xlabel('Wall Clock Time (s)')
        ax2.set_ylabel('Battery (%)')
        ax2.set_title('Battery vs Wall Clock Time')
        ax2.set_ylim(0, 105)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)
        fig2.tight_layout()
        path2 = base + '_battery_vs_time.png'
        fig2.savefig(path2)
        plt.close(fig2)
        print(f"[Battery] Saved: {path2}")

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
         
        # Record per-agent battery data
        for agent in self.agents:
            self.battery_records[agent.agent_id].append(
                (self.wall_clock_elapsed, agent.distance_moved, agent.battery)
            )        
                  