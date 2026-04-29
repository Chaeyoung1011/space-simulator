import random
import time
import pygame
import importlib
from modules.base_bt_nodes import BTNodeList, Status, Node, Sequence, Fallback, ReactiveSequence, ReactiveFallback, SyncAction, SyncCondition
from plugins.grape.grape import GRAPE

# BT Node List
CUSTOM_ACTION_NODES = [
    'GatherLocalInfo',
    'AssignSuperTask',
    'RebalanceGroups',
    'AssignTask',
    'MoveToTarget',
    'ExecuteTask',
    'Explore',
    'Idle',
]

CUSTOM_CONDITION_NODES = [
    'IsTaskCompleted',
    'IsArrivedAtTarget',
]

# Remove base-registered duplicates then extend
BTNodeList.ACTION_NODES    = [n for n in BTNodeList.ACTION_NODES    if n not in CUSTOM_ACTION_NODES]
BTNodeList.CONDITION_NODES = [n for n in BTNodeList.CONDITION_NODES if n not in CUSTOM_CONDITION_NODES]
BTNodeList.ACTION_NODES.extend(CUSTOM_ACTION_NODES)
BTNodeList.CONDITION_NODES.extend(CUSTOM_CONDITION_NODES)


# Scenario-specific config
from modules.utils import config
target_arrive_threshold = config['tasks']['threshold_done_by_arrival']
task_locations = config['tasks']['locations']
sampling_freq = config['simulation']['sampling_freq']
sampling_time = 1.0 / sampling_freq
agent_max_random_movement_duration = config.get('agents', {}).get('random_exploration_duration', None)
use_rotation_shim = config.get('agents', {}).get('use_rotation_shim', False)
super_task_dwell_required       = config.get('tasks',  {}).get('super_task_dwell_seconds', 1.0)

_dm_plugin_path = config.get('decision_making', {}).get('plugin')
if _dm_plugin_path and '.' in _dm_plugin_path:
    _module_path, _class_name = _dm_plugin_path.rsplit('.', 1)
    _phase2_class = getattr(importlib.import_module(_module_path), _class_name)
else:
    _phase2_class = None


def _move(agent, target):
    if use_rotation_shim:
        agent.follow_rotation_shim(target)
    else:
        agent.follow(target)


class GatherLocalInfo(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._update)

    def _update(self, agent, blackboard):
        blackboard['local_tasks_info'] = {
            t.task_id: t for t in agent.get_tasks_nearby(with_completed_task=False)
        }

        # CBBA messages (also populates agents_nearby)
        blackboard['local_agents_info'] = agent.local_message_receive()

        # GRAPE messages (super task channel)
        super_msgs = []
        for other in agent.agents_nearby:
            if other.agent_id != agent.agent_id:
                msg = getattr(other, 'super_task_message_to_share', {})
                if msg:
                    super_msgs.append(msg)
        agent.super_task_messages_received = super_msgs

        # Expose non-completed super tasks
        blackboard['super_tasks_info'] = {
            st.task_id: st
            for st in getattr(agent, 'super_tasks_info', [])
            if not st.completed
        }

        return Status.SUCCESS


class AssignSuperTask(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._update)
        self.decision_maker = GRAPE(agent)
        self._converge_start = None
        self._was_converged  = False

    def _update(self, agent, blackboard):
        # Already converged (blackboard latch)
        if blackboard.get('super_task_converged', False):
            self._was_converged = True
            return Status.SUCCESS

        # New generation detected: reset GRAPE state
        if self._was_converged:
            self._was_converged  = False
            self._converge_start = None
            self.decision_maker.reset()
            agent.super_task_message_to_share  = {}
            agent.super_task_messages_received = []

        # Keep agent still while converging
        agent.velocity     = pygame.Vector2(0, 0)
        agent.acceleration = pygame.Vector2(0, 0)

        super_tasks_info = blackboard.get('super_tasks_info', {})
        if not super_tasks_info:
            return Status.SUCCESS  # no super tasks configured → skip phase

        # Swap to GRAPE message channel
        orig_msg = agent.message_to_share
        orig_rcv = agent.messages_received
        agent.message_to_share  = getattr(agent, 'super_task_message_to_share', {})
        agent.messages_received = getattr(agent, 'super_task_messages_received', [])

        grape_bb = dict(blackboard)
        grape_bb['local_tasks_info'] = super_tasks_info
        assigned_id = self.decision_maker.decide(grape_bb)

        # Save GRAPE outgoing message; restore regular channel
        agent.super_task_message_to_share = agent.message_to_share
        agent.message_to_share  = orig_msg
        agent.messages_received = orig_rcv

        blackboard['assigned_super_task_id'] = assigned_id
        agent.assigned_super_task_id         = assigned_id

        # Check global convergence (all agents assigned for N sec)
        all_agents   = list(getattr(agent, 'agents_info', None) or [])
        all_assigned = all(getattr(a, 'assigned_super_task_id', None) is not None for a in all_agents)

        if not all_assigned:
            self._converge_start = None
            return Status.RUNNING

        now = time.time()
        if self._converge_start is None:
            self._converge_start = now
        if now - self._converge_start >= super_task_dwell_required:
            blackboard['super_task_converged'] = True
            self._converge_start = None

            if agent.agent_id == all_agents[0].agent_id:
                try:
                    lowest = min(all_agents, key=lambda a: getattr(a, 'battery', None) if getattr(a, 'battery', None) is not None else 100.0)
                    closest_st_id = min(super_tasks_info.values(), key=lambda st: (st.center - lowest.position).length()).task_id if super_tasks_info else None
                    batt = getattr(lowest, 'battery', None)
                    batt_str = f"{batt:.1f}%" if batt is not None else "Unknown (100.0%)"
                    print(f"\n{'='*50}\n[GRAPE Debug] Allocation Result")
                    print(f"1. Lowest battery agent: Agent {lowest.agent_id} (Battery: {batt_str})")
                    print(f"2. Closest SuperTask to Agent {lowest.agent_id}: ST{closest_st_id}")
                    print(f"3. Assigned SuperTask: ST{getattr(lowest, 'assigned_super_task_id', None)}")
                    print('='*50)
                except Exception as e:
                    print(f"[GRAPE Debug] Error: {e}")

            return Status.SUCCESS

        return Status.RUNNING

    def halt(self):
        pass  # preserve convergence timer across reactive ticks


class RebalanceGroups(SyncAction):
    """GRAPE 수렴 후 그룹별 (agents vs tasks) 균형 조정.
    surplus 그룹의 가장 배터리 높은 에이전트를 deficit 그룹으로 이동.
    한 번만 실행되고 blackboard에 latch."""

    def __init__(self, name, agent):
        super().__init__(name, self._update)

    def _update(self, agent, blackboard):
        if blackboard.get('rebalance_done', False):
            return Status.SUCCESS

        super_tasks = blackboard.get('super_tasks_info', {})
        if not super_tasks:
            blackboard['rebalance_done'] = True
            return Status.SUCCESS

        all_agents = list(getattr(agent, 'agents_info', None) or [])

        def agent_count(st_id):
            return sum(1 for a in all_agents if getattr(a, 'assigned_super_task_id', None) == st_id)

        def task_count(st):
            return sum(1 for t in st.tasks if not t.completed)

        # surplus가 사라질 때까지 반복
        while True:
            surplus_st = next(
                (st for st in super_tasks.values() if agent_count(st.task_id) > task_count(st)),
                None
            )
            if surplus_st is None:
                break

            deficit_st = max(
                (st for st in super_tasks.values() if st.task_id != surplus_st.task_id),
                key=lambda st: task_count(st) - agent_count(st.task_id),
                default=None,
            )
            if deficit_st is None:
                break

            # surplus ST의 에이전트 중 배터리 가장 높은 에이전트 이동
            surplus_agents = [a for a in all_agents
                              if getattr(a, 'assigned_super_task_id', None) == surplus_st.task_id]
            if not surplus_agents:
                break

            highest = max(surplus_agents,
                          key=lambda a: getattr(a, 'battery', None) or 100.0)
            highest.assigned_super_task_id = deficit_st.task_id

        # 자기 blackboard도 업데이트 (이 에이전트가 이동한 경우 반영)
        blackboard['assigned_super_task_id'] = getattr(agent, 'assigned_super_task_id', None)
        blackboard['rebalance_done'] = True
        return Status.SUCCESS


class AssignTask(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._update)
        self.decision_maker = _phase2_class(agent)

    def _update(self, agent, blackboard):
        assigned_super_task_id = blackboard.get('assigned_super_task_id')
        if assigned_super_task_id is None:
            return Status.FAILURE

        super_task = blackboard.get('super_tasks_info', {}).get(assigned_super_task_id)
        if super_task is None:
            return Status.FAILURE

        tasks_list = {t.task_id: t for t in super_task.tasks if not t.completed}
        if not tasks_list:
            return Status.FAILURE

        scoped_bb = dict(blackboard)
        scoped_bb['local_tasks_info'] = tasks_list

        assigned_id = self.decision_maker.decide(scoped_bb)

        if assigned_id is None or assigned_id not in tasks_list:
            agent.set_planned_tasks([])
            return Status.FAILURE

        agent.set_assigned_task_id(assigned_id)
        blackboard['assigned_task_id'] = assigned_id
        agent.set_planned_tasks([tasks_list[assigned_id]])
        return Status.SUCCESS


class IsTaskCompleted(SyncCondition):
    def __init__(self, name, agent):
        super().__init__(name, self._update)

    def _update(self, agent, blackboard):
        assigned_task_id = blackboard.get('assigned_task_id')
        if assigned_task_id is None:
            return Status.RUNNING

        task = agent.tasks_info[assigned_task_id]
        if task.completed is True:
            blackboard['assigned_task_id'] = None
            return Status.SUCCESS
        return Status.FAILURE


class IsArrivedAtTarget(SyncCondition):
    def __init__(self, name, agent):
        super().__init__(name, self._update)

    def _update(self, agent, blackboard):
        assigned_task_id = blackboard.get('assigned_task_id')
        if assigned_task_id is None:
            raise ValueError(f"[{self.name}] Error: No assigned_task_id found in the blackboard!")

        distance = (agent.tasks_info[assigned_task_id].position - agent.position).length()
        if distance < agent.tasks_info[assigned_task_id].radius + target_arrive_threshold:
            return Status.SUCCESS
        return Status.FAILURE


class MoveToTarget(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._update)

    def _update(self, agent, blackboard):
        assigned_task_id = blackboard.get('assigned_task_id')
        if assigned_task_id is None:
            raise ValueError(f"[{self.name}] Error: No assigned_task_id found in the blackboard!")

        _move(agent, agent.tasks_info[assigned_task_id].position)
        return Status.RUNNING


class ExecuteTask(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._update)

    def _update(self, agent, blackboard):
        assigned_task_id = blackboard.get('assigned_task_id')
        if assigned_task_id is None:
            raise ValueError(f"[{self.name}] Error: No assigned_task_id found in the blackboard!")

        agent.tasks_info[assigned_task_id].reduce_amount(agent.work_rate)
        agent.update_task_amount_done(agent.work_rate)
        _move(agent, agent.tasks_info[assigned_task_id].position)
        return Status.RUNNING


class Explore(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._update)
        self.random_move_time = float('inf')
        self.random_waypoint = (0, 0)

    def _update(self, agent, blackboard):
        if self.random_move_time > agent_max_random_movement_duration:
            self.random_waypoint = (
                random.randint(task_locations['x_min'], task_locations['x_max']),
                random.randint(task_locations['y_min'], task_locations['y_max'])
            )
            self.random_move_time = 0

        self.random_move_time += sampling_time
        _move(agent, self.random_waypoint)
        return Status.RUNNING

    def halt(self):
        self.random_move_time = float('inf')

class Idle(SyncAction):
    def __init__(self, name, agent):
        super().__init__(name, self._update)

    def _update(self, agent, blackboard):
        agent.velocity = pygame.Vector2(0, 0)
        agent.acceleration = pygame.Vector2(0, 0)
        # ★ rotation update 블록을 건너뛰게 함 (atan2(0,0)=0 회전 방지)
        agent._use_rotation_shim = True
        return Status.RUNNING