"""
MONA Robot Client Module

UDP client for communicating with MONA robots.
Sends motion commands (G commands) to ESP32-based robots.

Protocol:
    - "G <angle_deg> <distance_mm>\\n" : Move command
    - "STOP\\n" : Stop command
"""
import json
import socket
import math
import threading
from typing import Tuple, Optional


class BatteryReceiver:
    """Singleton UDP listener that receives battery data from all MONA robots.

    Arduino sends: {"battery": 85.2, "pulses": 12345}  →  port 5005
    Source IP identifies which robot sent the packet.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, listen_port: int, ip_to_agent_id: dict):
        self._ip_to_agent_id = ip_to_agent_id   # {robot_ip: agent_id}
        self._battery_data: dict = {}            # {agent_id: battery%}
        self._data_lock = threading.Lock()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('', listen_port))
        self._sock.settimeout(0.5)

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[BatteryReceiver] Listening on UDP port {listen_port}")

    # ── Public API ──────────────────────────────────────────────

    def get_battery(self, agent_id: int) -> Optional[float]:
        """Return latest battery % for agent_id, or None if not yet received."""
        with self._data_lock:
            return self._battery_data.get(agent_id, None)

    def close(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass

    # ── Background recv loop ─────────────────────────────────────

    def _recv_loop(self):
        while self._running:
            try:
                data, addr = self._sock.recvfrom(256)
                ip = addr[0]
                payload = json.loads(data.decode('utf-8'))
                batt = float(payload['battery'])
                agent_id = self._ip_to_agent_id.get(ip)
                if agent_id is not None:
                    with self._data_lock:
                        self._battery_data[agent_id] = batt
            except socket.timeout:
                pass
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, config: dict) -> 'BatteryReceiver':
        """Return the shared singleton, creating it on first call."""
        with cls._lock:
            if cls._instance is None:
                mona_cfg = config.get('mona', {})
                listen_port = int(mona_cfg.get('battery_listen_port', 5005))
                ip_to_agent_id = {
                    robot['host']: int(robot['agent_id'])
                    for robot in mona_cfg.get('robots', [])
                }
                cls._instance = cls(listen_port, ip_to_agent_id)
            return cls._instance


class MonaClient:
    """
    UDP client for MONA robot communication.
    
    Sends fire-and-forget UDP packets to control robot movement.
    The robot's Arduino firmware handles the actual motion control.
    """
    
    DEFAULT_PX_TO_MM = 1.0
    DEFAULT_DISTANCE_SCALE = 1.0

    def __init__(self, host: str, port: int, 
                 px_to_mm: float = None, 
                 distance_scale: float = None):
        """
        Initialize UDP client for a MONA robot.
        
        Args:
            host: Robot IP address
            port: Robot UDP port
            px_to_mm: Pixel to millimeter conversion factor
            distance_scale: Additional distance scaling factor
        """
        self.host = host
        self.port = int(port)
        self.px_to_mm = float(px_to_mm or self.DEFAULT_PX_TO_MM)
        self.distance_scale = float(distance_scale or self.DEFAULT_DISTANCE_SCALE)
        
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.is_connected = True  # UDP is connectionless, always "connected"

    # ==================== Public API ====================

    def send_g_to(self, current_position: Tuple[float, float],
                  current_heading: float,
                  target_position: Tuple[float, float]) -> None:
        """
        Send a G command to move robot towards target.
        
        Args:
            current_position: Current (x, y) in pixels
            current_heading: Current heading in radians
            target_position: Target (x, y) in pixels
        """
        angle_deg, distance_mm = self._compute_motion(
            current_position, current_heading, target_position
        )
        self._send_g_command(angle_deg, distance_mm)

    def send_stop(self) -> None:
        """Send stop command to robot."""
        self._send_packet(b"STOP\n")

    def close(self) -> None:
        """Close the UDP socket."""
        if self._socket:
            self._socket.close()
            self._socket = None
            self.is_connected = False

    # ==================== Motion Calculation ====================

    def _compute_motion(self, current_xy: Tuple[float, float],
                        current_heading: float,
                        target_xy: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute angle and distance to target.
        
        Args:
            current_xy: Current position (x, y) in pixels
            current_heading: Current heading in radians
            target_xy: Target position (x, y) in pixels
            
        Returns:
            Tuple of (angle_degrees, distance_mm)
        """
        # Calculate vector to target
        dx = target_xy[0] - current_xy[0]
        dy = target_xy[1] - current_xy[1]
        
        # Convert distance to mm
        distance_px = math.hypot(dx, dy)
        distance_mm = distance_px * self.px_to_mm * self.distance_scale
        
        # Calculate angle difference
        desired_heading = math.atan2(dy, dx)
        angle_diff = self._normalize_angle(desired_heading - current_heading)
        angle_deg = math.degrees(angle_diff)
        
        return angle_deg, distance_mm

    # ==================== Network Communication ====================

    def _send_g_command(self, angle_deg: float, distance_mm: float) -> None:
        """Send formatted G command."""
        payload = f"G {angle_deg:.2f} {distance_mm:.1f}\n"
        self._send_packet(payload.encode())
        # print(f"[UDP->{self.host}:{self.port}] {payload.strip()}")

    def _send_packet(self, data: bytes) -> None:
        """Send raw UDP packet."""
        try:
            self._socket.sendto(data, (self.host, self.port))
        except Exception as e:
            print(f"[MonaClient] UDP send failed: {e}")

    # ==================== Utility ====================

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # ==================== Factory Method ====================

    @classmethod
    def from_config(cls, agent_id: int, config: dict) -> 'MonaClient':
        """
        Create MonaClient from configuration dictionary.
        
        Args:
            agent_id: Agent ID to find robot config for
            config: Root configuration dictionary
            
        Returns:
            Configured MonaClient instance
        """
        mona_cfg = config.get('mona', {})
        robots = mona_cfg.get('robots', [])
        
        # Find robot config for this agent
        robot_cfg = None
        for robot in robots:
            if robot.get('agent_id') == agent_id:
                robot_cfg = robot
                break
        
        # Use found config or fall back to global config
        cfg = robot_cfg or mona_cfg
        
        return cls(
            host=cfg.get('host', '127.0.0.1'),
            port=int(cfg.get('port', 8080)),
            px_to_mm=mona_cfg.get('px_to_mm', cls.DEFAULT_PX_TO_MM),
            distance_scale=mona_cfg.get('distance_scale', cls.DEFAULT_DISTANCE_SCALE)
        )