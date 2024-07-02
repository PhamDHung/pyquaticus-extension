#DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause
from pyquaticus.envs.pyquaticus import PyQuaticusEnvBase
import colorsys
import copy
import itertools
import math
import random
from collections import defaultdict
from typing import Optional, Dict, Hashable, List

import numpy as np
import pygame
from gymnasium.utils import seeding
from pygame import SRCALPHA, draw
from pygame.math import Vector2
from pygame.transform import rotozoom

from pyquaticus.config import config_dict_std, ACTION_MAP
from pyquaticus.structs import Team, RenderingPlayer, Flag, CircleObstacle, PolygonObstacle
from pyquaticus.utils.pid import PID
from pyquaticus.utils.utils import (
    angle180,
    clip,
    get_rot_angle,
    mag_bearing_to,
    mag_heading_to_vec,
    rc_intersection,
    reflect_vector,
    rot2d,
    vec_to_mag_heading,
)

class PyQuaticusEnv(PyQuaticusEnvBase):
    """
    ### Description.
    This environment simulates a game of capture the flag with agent dynamics based on MOOS-IvP
    (https://oceanai.mit.edu/ivpman/pmwiki/pmwiki.php?n=IvPTools.USimMarine#section5).


    ### Rewards

    Reward functions will be learned using an inverse reinforcement learning algorithm (D-REX)

    ### Starting State (need to update)

    Each flag is placed at 0° latitude and 1/8 horizontal world size distance away from the back
    wall of its respective team's territory.

    If random_init is False, agents are spawned facing the scrimmage line at 0° latitude and
    1/4 horizontal world size distance back from the scrimmage line.

    If random_init is True, agents are spawned facing the scrimmage line at 0° latitude and
    equidistant from the scrimmage line (this distance is variable). Then, they are shifted a variable
    distance to either the left or right (same direction for both agents) with the constraints that
    they cannot start inside the flag_keepout zone or behind their flag.

    ### Arguments

    ```
    gym.make('pyquaticus')
    ```
    team_size: number of agents per team
    num_flags: number of flags per team
    reward_config: a dictionary configuring the reward structure (see rewards.py)
    config_dict: a dictionary configuring the environment (see config_dict_std above)
    render_mode: String for how to render the game
    render_agent_ids: Flag for displaying IDs next to agents.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        team_size: int = 1,
        num_flags: int = 1,
        reward_config: dict = None,
        config_dict=config_dict_std,
        render_mode: Optional[str] = None,
        render_agent_ids: Optional[bool] = False
    ):
        super().__init__(team_size, num_flags, config_dict=config_dict)

        self.render_mode = render_mode
        self.render_ids = render_agent_ids

        # set variables from config
        self.set_config_values(self.config_dict)

        self.dones = {}
        self.reset_count = 0

        self.learning_iteration = 0

        self.seed()

        self.current_time = 0

        self.num_blue = self.team_size
        self.num_red = self.team_size

        self.agents = [agent_id for agent_id in self.players.keys()]
        self.possible_agents = self.agents[:]
        self.agents_of_team = {team: self.get_players_of_team(team) for team in Team}
        assert len(self.agents_of_team[Team.BLUE_TEAM]) == len(
            self.agents_of_team[Team.RED_TEAM]
        )
    
        # Create a PID controller for each agent
        if self.render_mode:
            dt = 1/self.render_fps
        else:
            dt = self.tau

        # Setup Rewards
        self.reward_config = {} if reward_config is None else reward_config
        self._pid_controllers = {}
        self.params = dict()
        self.prev_params = dict()

        for player_id in self.agents:
            # Setup PID for each player
            self._pid_controllers[player_id] = {
                "speed": PID(dt=dt, kp=1.0, ki=0.0, kd=0.0, integral_max=0.07),
                "heading": PID(dt=dt, kp=0.35, ki=0.0, kd=0.07, integral_max=0.07),
            }

            # Setup reward config
            if player_id not in self.reward_config:
                self.reward_config[player_id] = None
            
            # Initialize per-agent parameters
            self.params[player_id] = dict()
            self.prev_params[player_id] = dict()

            # Set tagging cooldown
            self.players[player_id].tagging_cooldown = self.tagging_cooldown

        # pygame screen
        self.screen = None
        self.clock = None
        self.isopen = False

    def create_players(self) -> Dict[Hashable, RenderingPlayer]:
        players = dict()
        for i in range(0, self.team_size):
            players[i] = RenderingPlayer(i, Team.BLUE_TEAM, (self.agent_radius * self.pixel_size), self.config_dict, self.render_mode)
        for i in range(self.team_size, 2 * self.team_size):
            players[i] = RenderingPlayer(i, Team.RED_TEAM, (self.agent_radius * self.pixel_size), self.config_dict, self.render_mode)
        return players

    def create_flags(self) -> Dict[Team, List[Flag]]:
        flags = dict()
        for team in Team:
            for _ in range(self.num_flags):
                self.flags[team].append(Flag(team))
        return flags

    def step(self, raw_action_dict):
        """
        Steps the environment forward in time by `tau`, applying actions.

        Args:
            raw_action_dict: Actions IDs from discrete action space for agents to apply

        Returns:
            New observation after stepping

            Some reward

            Indicator for whether or not the env is done/truncated

            Additional info (not currently used)
        """
        if self.state is None:
            raise Exception("Call reset before using step method.")

        # set the time
        self.current_time += self.sim_speedup_factor * self.tau
        # self.state["current_time"] = self.current_time
        self.get_state()["current_time"] = self.current_time
        
        if not set(raw_action_dict.keys()) <= set(self.players):
            raise ValueError(
                "Keys of action dict should be player ids but got"
                f" {raw_action_dict.keys()}"
            )

        for player in self.players.values():
            if player.tagging_cooldown != self.tagging_cooldown:
                # player is still under a cooldown from tagging, advance their cooldown timer, clip at the configured tagging cooldown
                player.tagging_cooldown = self._min(
                    (player.tagging_cooldown + self.sim_speedup_factor * self.tau), self.tagging_cooldown
                )

        self.flag_collision_bool = np.zeros(self.num_agents)

        action_dict = self._to_speed_heading(raw_action_dict)
        if self.render_mode:
            for _i in range(self.num_renders_per_step):
                for _j in range(self.sim_speedup_factor):
                    self._move_agents(action_dict, 1/self.render_fps)
                self._render()
        else:
            for _ in range(self.sim_speedup_factor):
                self._move_agents(action_dict, self.tau)

        # agent and flag capture checks and more
        self._check_pickup_flags()
        self._check_agent_captures()
        self._check_flag_captures()
        if not self.teleport_on_tag:
            self._check_untag()
        self._set_dones()
        self._get_dist_to_obstacles()

        if self.message and self.render_mode:
            print(self.message)
        rewards = {agent_id: self.compute_rewards(agent_id) for agent_id in self.players}
        obs = {agent_id: self.state_to_obs(agent_id, self.normalize) for agent_id in raw_action_dict}
        info = {}

        terminated = False
        truncated = False
        if self.dones["__all__"]:
            if self.dones["blue"] or self.dones["red"]:
                terminated = True
            else:
                truncated = True
        terminated = {agent: terminated for agent in raw_action_dict}
        truncated = {agent: truncated for agent in raw_action_dict}

        return obs, rewards, terminated, truncated, info

    def _move_agents(self, action_dict, dt):
        """Moves agents in the space according to the specified speed/heading in `action_dict`."""
        for player in self.players.values():
            pos_x = player.pos[0]
            pos_y = player.pos[1]
            flag_loc = self.flags[int(player.team)].home

            if player.team == Team.BLUE_TEAM:
                player.on_own_side = pos_x >= self.scrimmage
            else:
                player.on_own_side = pos_x <= self.scrimmage

            # convert desired_speed   and  desired_heading to
            #         desired_thrust  and  desired_rudder
            # requested heading is relative so it directly maps to the heading error
            
            if player.is_tagged and not self.teleport_on_tag:
                flag_home = self.flags[int(player.team)].home
                _, heading_error = mag_bearing_to(player.pos, flag_home, player.heading)
                desired_speed = self.config_dict["max_speed"]
            else:
                desired_speed, heading_error = action_dict[player.id]

            # desired heading is relative to current heading
            speed_error = desired_speed - player.speed
            desired_speed = self._pid_controllers[player.id]["speed"](speed_error)
            desired_rudder = self._pid_controllers[player.id]["heading"](heading_error)

            desired_thrust = player.thrust + self.speed_factor * desired_speed

            desired_thrust = clip(desired_thrust, -self.max_thrust, self.max_thrust)
            desired_rudder = clip(desired_rudder, -self.max_rudder, self.max_rudder)

            # propagate vehicle speed
            raw_speed = np.interp(
                desired_thrust, self.thrust_map[0, :], self.thrust_map[1, :]
            )
            new_speed = min(
                raw_speed * 1 - ((abs(desired_rudder) / 100) * self.turn_loss),
                self.max_speed,
            )
            if (new_speed - player.speed) / dt > self.max_acc:
                new_speed = new_speed + self.max_acc * dt
            elif (player.speed - new_speed) / dt > self.max_dec:
                new_speed = new_speed - self.max_dec * dt

            # propagate vehicle heading
            raw_d_hdg = desired_rudder * (self.turn_rate / 100) * dt
            thrust_d_hdg = raw_d_hdg * (1 + (abs(desired_thrust) - 50) / 50)
            if desired_thrust < 0:
                thrust_d_hdg = -thrust_d_hdg

            # if not moving, then can't turn
            if (new_speed + player.speed) / 2.0 < 0.5:
                thrust_d_hdg = 0.0
            new_heading = angle180(player.heading + thrust_d_hdg)

            vel = mag_heading_to_vec(new_speed, new_heading)

            # If the player hits a boundary, return them to their original starting position and skip
            # to the next agent.
            player_hit_obstacle = False
            for obstacle in self.obstacles:
                collision = obstacle.detect_collision((pos_x, pos_y), radius = self.agent_radius)
                if collision is True:
                    player_hit_obstacle = True
                    break
            if player_hit_obstacle is True or not (
                (self.agent_radius <= pos_x <= self.world_size[0] - self.agent_radius)
                and (
                    self.agent_radius <= pos_y <= self.world_size[1] - self.agent_radius
                )
            ):
                if player.team == Team.RED_TEAM:
                    self.game_score['blue_tags'] += 1
                else:
                    self.game_score['red_tags'] += 1
                if player.has_flag:
                    # If they have a flag, return the flag to it's home area
                    self.flags[int(not int(player.team))].reset()
                    # self.state["flag_taken"][int(not int(player.team))] = 0

                    if player.team == Team.RED_TEAM:
                        self.red_team_flag_pickup = False
                    else:
                        self.blue_team_flag_pickup = False
                self.get_state()["agent_oob"][player.id] = 1
                if self.teleport_on_tag:
                    player.reset()
                else:
                    if self.tag_on_wall_collision:
                        self.state["agent_tagged"][player.id] = 1
                        player.is_tagged = True
                    player.rotate(copy.deepcopy(player.prev_pos))
                continue
            else:
                self.state["agent_oob"][player.id] = 0

            # check if agent is in keepout region
            ag_dis_2_flag = self.get_distance_between_2_points(
                np.asarray([pos_x, pos_y]), np.asarray(flag_loc)
            )
            if (
                ag_dis_2_flag < self.flag_keepout
                and not self.get_state()["flag_taken"][int(player.team)]
                and self.flag_keepout > 0.
            ):
                self.flag_collision_bool[player.id] = True

                ag_pos = np.array([pos_x, pos_y])
                ag_pos = rc_intersection(
                    np.array([ag_pos, player.prev_pos]),
                    np.asarray(flag_loc),
                    self.flag_keepout,
                )  # point where agent center first intersected with keepout zone
                ag_vel = reflect_vector(ag_pos, vel, np.asarray(flag_loc))

                crd_ref_angle = get_rot_angle(np.asarray(flag_loc), ag_pos)
                vel_ref = rot2d(ag_vel, -crd_ref_angle)
                vel_ref[1] = 0. # convention is that vector pointing from keepout intersection to flag center is y' axis in new reference frame

                vel = rot2d(vel_ref, crd_ref_angle)
                pos_x = ag_pos[0]
                pos_y = ag_pos[1]

            if new_speed > 0.1:
                # only rely on vel if speed is large enough to recover heading
                new_speed, new_heading = vec_to_mag_heading(vel)
            # propagate vehicle position
            hdg_rad = math.radians(player.heading)
            new_hdg_rad = math.radians(new_heading)
            avg_speed = (new_speed + player.speed) / 2.0
            s = math.sin(new_hdg_rad) + math.sin(hdg_rad)
            c = math.cos(new_hdg_rad) + math.cos(hdg_rad)
            avg_hdg = math.atan2(s, c)
            # Note: sine/cos swapped because of the heading / angle difference
            new_ag_pos = [
                pos_x + math.sin(avg_hdg) * avg_speed * dt,
                pos_y + math.cos(avg_hdg) * avg_speed * dt,
            ]

            if player.has_flag:
                flg_idx = not int(player.team)
                self.flags[flg_idx].pos = list(new_ag_pos)

            player.prev_pos = player.pos
            player.pos = np.asarray(new_ag_pos)
            player.speed = clip(new_speed, 0.0, self.max_speed)
            player.heading = angle180(new_heading)
            player.thrust = desired_thrust

    def _check_pickup_flags(self):
        """Updates player states if they picked up the flag."""
        for player in self.players.values():
            team = int(player.team)
            other_team = int(not team)
            other_team_enum = Team.RED_TEAM if player.team == Team.BLUE_TEAM else Team.BLUE_TEAM
            if not (player.has_flag or self.get_state()["flag_taken"][other_team]) and (not player.is_tagged):
                for flag in self.get_flags_of_team(other_team_enum):    
                    flag_pos = flag.pos
                    distance_to_flag = self.get_distance_between_2_points(
                        player.pos, flag_pos
                    )

                    if distance_to_flag < self.catch_radius:
                        player.has_flag = True
                        flag.taken = True
                        self.get_state()["flag_taken"][other_team] = 1
                        if player.team == Team.BLUE_TEAM:
                            self.game_score['blue_grabs'] += 1
                            self.blue_team_flag_pickup = True
                        else:
                            self.game_score['red_grabs'] += 1
                            self.red_team_flag_pickup = True
                            break

    def _check_untag(self):
        """Untags the player if they return to their own flag."""
        for player in self.players.values():
            team = int(player.team)
            flag_homes = [flag.home for flag in self.get_flags_of_team(team)]
            for flag_home in flag_homes:
                distance_to_flag = self.get_distance_between_2_points(
                    player.pos, flag_home
                )
                if distance_to_flag < self.catch_radius and player.is_tagged:
                    self.get_state()["agent_tagged"][player.id] = 0
                    player.is_tagged = False

    def _check_agent_captures(self):
        """Updates player states if they tagged another player."""
        # Reset capture state if teleport_on_tag is true
        if self.teleport_on_tag:
            self.get_state()["agent_tagged"] = [0] * self.num_agents
            for player in self.players.values():
                player.is_tagged = False

        self.get_state()["agent_captures"] = [None] * self.num_agents
        for player in self.players.values():
            # Only continue logic check if player tagged someone if it's on its own side and is untagged.
            if player.on_own_side and (
                player.tagging_cooldown == self.tagging_cooldown
            ) and not player.is_tagged:
                for other_player in self.players.values():
                    o_team = int(other_player.team)
                    # Only do the rest if the other player is NOT on sides and they are not on the same team.
                    if (
                        not other_player.is_tagged
                        and not other_player.on_own_side
                        and other_player.team != player.team
                    ):
                        dist_between_agents = self.get_distance_between_2_points(
                            player.pos, other_player.pos
                        )
                        if (
                            dist_between_agents > 0.0
                            and dist_between_agents < self.catch_radius
                        ):
                            # If we get here, then `player` tagged `other_player` and we need to reset `other_player`
                            if other_player.team == Team.RED_TEAM:
                                self.game_score['blue_tags'] += 1
                            else:
                                self.game_score['red_tags'] += 1
                            self.get_state()["agent_tagged"][other_player.id] = 1
                            other_player.is_tagged = True
                            self.get_state()["agent_captures"][player.id] = other_player.id
                            if other_player.has_flag:
                                # If the player with the flag was tagged, the flag also needs to be reset.
                                if other_player.team == Team.BLUE_TEAM:
                                    self.blue_team_flag_pickup = False
                                else:
                                    self.red_team_flag_pickup = False
                                self.get_state()["flag_taken"][int(player.team)] = 0
                                for flag in self.get_flags_of_team(player.team):
                                    if np.allclose(flag.pos, other_player.pos):
                                        flag.reset()
                                other_player.has_flag = False

                            # Only if config["teleport_on_capture"] == True
                            if self.teleport_on_tag:
                                buffer_sign = (
                                    1.0
                                    if self.flags[o_team].home[0] < self.scrimmage
                                    else -1.0
                                )
                                other_player.pos[0] = self.flags[o_team].home[
                                    0
                                ] + buffer_sign * (self.flag_keepout + 0.1)
                                other_player.pos[1] = self.flags[o_team].home[1]
                                other_player.speed = 0.0
                                other_player.on_own_side = True

                            # Set players tagging cooldown
                            player.tagging_cooldown = 0.0

                            # Break loop (should not be allowed to tag again during current timestep)
                            break

    def _check_flag_captures(self):
        """Updates states if a player captured a flag."""
        # these are false except at the instance that the flag is captured
        self.blue_team_flag_capture = False
        self.red_team_flag_capture = False
        for player in self.players.values():
            if player.on_own_side and player.has_flag:
                if player.team == Team.BLUE_TEAM:
                    self.blue_team_flag_capture = True
                    self.game_score['blue_captures'] += 1
                    self.blue_team_flag_pickup = False
                    other_team = Team.RED_TEAM
                else:
                    self.red_team_flag_capture = True
                    self.game_score['red_captures'] += 1
                    self.red_team_flag_pickup = False
                    other_team = Team.BLUE_TEAM
                player.has_flag = False
                flags = self.get_flags_of_team(other_team)
                for flag in flags:
                    if np.isclose(player.pos, flag.pos):
                        flag.reset()
                        break
                self.get_state()["flag_taken"][other_team] = False
                break

    def get_full_state_info(self):
        """Return the full state."""
        return self.get_state()

    def current_iter_value(self, iter_value):
        """Returns the learning iteration."""
        self.learning_iteration = iter_value

    def set_config_values(self, config_dict):
        """
        Sets initial configuration parameters for the environment.

        Args:
            config_dict: The provided configuration. If a key is missing, it is replaced
            with the standard configuration value.
        """
        # set variables from config
        super().set_config_values(config_dict)
        self.pixel_size = config_dict.get("pixel_size", config_dict_std["pixel_size"])
        self.agent_radius = config_dict.get(
            "agent_radius", config_dict_std["agent_radius"]
        )
        self.flag_radius = self.agent_radius  # agent and flag radius will be the same
        self.catch_radius = config_dict.get(
            "catch_radius", config_dict_std["catch_radius"]
        )
        self.flag_keepout = config_dict.get(
            "flag_keepout", config_dict_std["flag_keepout"]
        )
        self.own_side_accel = config_dict.get(
            "own_side_accel", config_dict_std["own_side_accel"]
        )
        self.opp_side_accel = config_dict.get(
            "opp_side_accel", config_dict_std["opp_side_accel"]
        )
        self.wall_bounce = config_dict.get(
            "wall_bounce", config_dict_std["wall_bounce"]
        )
        self.tau = config_dict.get("tau", config_dict_std["tau"])
        self.sim_speedup_factor = config_dict.get("sim_speedup_factor", config_dict_std["sim_speedup_factor"])
        self.max_time = config_dict.get("max_time", config_dict_std["max_time"])
        self.max_screen_size = config_dict.get(
            "max_screen_size", config_dict_std["max_screen_size"]
        )
        self.random_init = config_dict.get(
            "random_init", config_dict_std["random_init"]
        )
        self.save_traj = config_dict.get("save_traj", config_dict_std["save_traj"])
        self.render_fps = config_dict.get("render_fps", config_dict_std["render_fps"])
        self.normalize = config_dict.get("normalize", config_dict_std["normalize"])
        self.teleport_on_tag = config_dict.get("teleport_on_tag", config_dict_std["teleport_on_tag"])
        self.tag_on_wall_collision = config_dict.get("tag_on_wall_collision", config_dict_std["tag_on_wall_collision"])

        # MOOS Dynamics Parameters
        self.speed_factor = config_dict.get(
            "speed_factor", config_dict_std["speed_factor"]
        )
        self.thrust_map = config_dict.get("thrust_map", config_dict_std["thrust_map"])
        self.max_thrust = config_dict.get("max_thrust", config_dict_std["max_thrust"])
        self.max_rudder = config_dict.get("max_rudder", config_dict_std["max_rudder"])
        self.turn_loss = config_dict.get("turn_loss", config_dict_std["turn_loss"])
        self.turn_rate = config_dict.get("turn_rate", config_dict_std["turn_rate"])
        self.max_acc = config_dict.get("max_acc", config_dict_std["max_acc"])
        self.max_dec = config_dict.get("max_dec", config_dict_std["max_dec"])

        if config_dict.get(
            "suppress_numpy_warnings", config_dict_std["suppress_numpy_warnings"]
        ):
            # Suppress numpy warnings to avoid printing out extra stuff to the console
            np.seterr(all="ignore")

        if self.render_mode:
            # Pygame Orientation Vector
            self.UP = Vector2((0.0, 1.0))

            # arena
            self.arena_offset = 20  # pixels
            self.border_width = 4  # pixels
            self.a2a_line_width = 3 #pixels

            self.arena_width = self.world_size[0] * self.pixel_size
            self.arena_height = self.world_size[1] * self.pixel_size

            # check that world size (pixels) does not exceed the screen dimensions
            world_screen_err_msg = (
                "Specified world_size {} exceeds the maximum size {} in at least one"
                " dimension".format(
                    [
                        round(2*self.arena_offset + self.pixel_size * self.world_size[0]),
                        round(2*self.arena_offset + self.pixel_size * self.world_size[1])
                    ], 
                    self.max_screen_size
                )
            )
            assert (
                2*self.arena_offset + self.pixel_size * self.world_size[0] <= self.max_screen_size[0]
                and 2*self.arena_offset + self.pixel_size * self.world_size[1] <= self.max_screen_size[1]
            ), world_screen_err_msg

        # check that world dimensions (pixels) are even
        world_even_err_msg = (
            "Specified world_size {} has at least one dimension that is not even"
            .format(self.world_size)
        )
        assert (self.world_size[0] * self.pixel_size) % 2 == 0 and (
            self.world_size[1] * self.pixel_size
        ) % 2 == 0, world_even_err_msg

        self.screen_size = [self.pixel_size * d for d in self.world_size]

        # check that time between frames (1/render_fps) is not larger than timestep (tau)
        frame_rate_err_msg = (
            "Specified frame rate ({}) creates time intervals between frames larger"
            " than specified timestep ({})".format(self.render_fps, self.tau)
        )
        assert 1 / self.render_fps <= self.tau, frame_rate_err_msg

        self.num_renders_per_step = int(self.render_fps * self.tau)

        # check that time warp is a integer >= 1 and only active if rendering is on
        if self.sim_speedup_factor < 1:
            print("Warning: sim_speedup_factor must be an integer >= 1! Defaulting to 1.")
            self.sim_speedup_factor = 1

        if type(self.sim_speedup_factor) != int:
            self.sim_speedup_factor = int(np.round(self.sim_speedup_factor))
            print(f"Warning: Converted sim_speedup_factor to integer: {self.sim_speedup_factor}")

        # check that agents and flags properly fit within world
        agent_flag_err_msg = (
            "Specified agent_radius ({}), flag_radius ({}), and flag_keepout ({})"
            " create impossible initialization based on world_size({})".format(
                self.agent_radius, self.flag_radius, self.flag_keepout, self.world_size
            )
        )
        horizontal_fit = (
            2 * self.agent_radius + 2 * self.flag_keepout < self.world_size[0] / 2
        )
        vertical_fit = self.flag_keepout + self.agent_radius < self.world_size[1] / 2
        assert horizontal_fit and vertical_fit, agent_flag_err_msg

    def get_distance_between_2_points(self, start: np.array, end: np.array) -> float:
        """
        Convenience method for returning distance between two points.

        Args:
            start: Starting position to measure from
            end: Point to measure to
        Returns:
            The distance between `start` and `end`
        """
        return np.linalg.norm(np.asarray(start) - np.asarray(end))

    def _set_dones(self):
        """Check all of the end game conditions."""
        # Check if all flags of one team are captured
        if self.game_score["red_captures"] >= self.max_score:
            self.dones["red"] = True
            self.dones["__all__"] = True
            self.message = "Red Wins! Blue Loses"

        elif self.game_score["blue_captures"] >= self.max_score:
            self.dones["blue"] = True
            self.dones["__all__"] = True
            self.message = "Blue Wins! Red Loses"

        elif self.state["current_time"] >= self.max_time:
            self.dones["__all__"] = True
            self.message = "Game Over. No Winner"

    def update_params(self, agent_id):
        # Important Note: Be sure to deep copy anything other than plain-old-data, e.g.,
        # lists from self.state
        # Otherwise it will point to the same object and prev_params/params will be identical
        agent = self.players[agent_id]
        obs = self.state_to_obs(agent.id, False)
        self.params[agent.id]["team"] = agent.team
        self.params[agent.id]["capture_radius"] = self.catch_radius
        self.params[agent.id]["agent_id"] = agent.id
        self.params[agent.id]["agent_oob"] = copy.deepcopy(self.state["agent_oob"])

        # Obstacle Distance/Bearing
        for i, obstacle in enumerate(self.state["dist_to_obstacles"][agent.id]):
            self.params[agent.id][f"obstacle_{i}_distance"] = obstacle[0]
            self.params[agent.id][f"obstacle_{i}_bearing"] = obstacle[1]

        if agent.team == Team.RED_TEAM:
            # Game Events
            self.params[agent.id]["num_teammates"] = self.num_red
            self.params[agent.id]["num_opponents"] = self.num_blue
            self.params[agent.id]["team_flag_pickup"] = self.red_team_flag_pickup
            self.params[agent.id]["team_flag_capture"] = self.red_team_flag_capture
            self.params[agent.id]["opponent_flag_pickup"] = self.blue_team_flag_pickup
            self.params[agent.id]["opponent_flag_capture"] = self.blue_team_flag_capture
            # Elements
            self.params[agent.id]["team_flag_home"] = self.get_distance_between_2_points(
                    agent.pos, copy.deepcopy(self.state["flag_home"][1])
                )
            self.params[agent.id]["team_flag_bearing"] = obs["own_home_bearing"]
            self.params[agent.id]["team_flag_distance"] = obs["own_home_distance"]
            self.params[agent.id]["opponent_flag_bearing"] = obs[
                "opponent_home_bearing"
            ]
            self.params[agent.id]["opponent_flag_distance"] = obs[
                "opponent_home_distance"
            ]
        else:
            # Game Events
            self.params[agent.id]["num_teammates"] = self.num_blue
            self.params[agent.id]["num_opponents"] = self.num_red
            self.params[agent.id]["team_flag_pickup"] = self.blue_team_flag_pickup
            self.params[agent.id]["team_flag_capture"] = self.blue_team_flag_capture
            self.params[agent.id]["opponent_flag_pickup"] = self.red_team_flag_pickup
            self.params[agent.id]["opponent_flag_capture"] = self.red_team_flag_capture
            # Elements
            self.params[agent.id]["team_flag_home"] = self.get_distance_between_2_points(
                    agent.pos, copy.deepcopy(self.state["flag_home"][0])
                )
            self.params[agent.id]["team_flag_bearing"] = obs["own_home_bearing"]
            self.params[agent.id]["team_flag_distance"] = obs["own_home_distance"]
            self.params[agent.id]["opponent_flag_bearing"] = obs[
                "opponent_home_bearing"
            ]
            self.params[agent.id]["opponent_flag_distance"] = obs[
                "opponent_home_distance"
            ]
        self.params[agent.id]["num_players"] = len(self.players)
        self.params[agent.id]["speed"] = agent.speed
        self.params[agent.id]["tagging_cooldown"] = (
            not agent.tagging_cooldown >= 10.0
        )
        self.params[agent.id]["thrust"] = agent.thrust
        self.params[agent.id]["has_flag"] = agent.has_flag
        self.params[agent.id]["on_own_side"] = agent.on_own_side
        self.params[agent.id]["heading"] = agent.heading
        # Distances to boundaries
        self.params[agent.id]["wall_0_bearing"] = obs["wall_0_bearing"]
        self.params[agent.id]["wall_0_distance"] = obs["wall_0_distance"]
        self.params[agent.id]["wall_1_bearing"] = obs["wall_1_bearing"]
        self.params[agent.id]["wall_1_distance"] = obs["wall_1_distance"]
        self.params[agent.id]["wall_2_bearing"] = obs["wall_2_bearing"]
        self.params[agent.id]["wall_2_distance"] = obs["wall_2_distance"]
        self.params[agent.id]["wall_3_bearing"] = obs["wall_3_bearing"]
        self.params[agent.id]["wall_3_distance"] = obs["wall_3_distance"]
        self.params[agent.id]["wall_distances"] =  self._get_dists_to_boundary()[agent.id]
        self.params[agent.id]["agent_captures"] = copy.deepcopy(self.state["agent_captures"])
        self.params[agent.id]["agent_tagged"] = copy.deepcopy(self.state["agent_tagged"])
        own_team = agent.team
        other_team = Team.BLUE_TEAM if own_team == Team.RED_TEAM else Team.RED_TEAM
        # Add Teamate and Opponent Information
        for team in [own_team, other_team]:
            dif_agents = filter(lambda a: a.id != agent.id, self.agents_of_team[team])
            for i, dif_agent in enumerate(dif_agents):
                entry_name = f"teammate_{i}" if team == own_team else f"opponent_{i}"
                status = "teammate" if team == own_team else "opponent"
                # bearing relative to the bearing to you
                self.params[agent.id][f"{status}_{dif_agent.id}_bearing"] = obs[
                    (entry_name, "bearing")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_distance"] = obs[
                    (entry_name, "distance")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_relative_heading"] = (
                    obs[(entry_name, "relative_heading")]
                )
                self.params[agent.id][f"{status}_{dif_agent.id}_speed"] = obs[
                    (entry_name, "speed")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_has_flag"] = obs[
                    (entry_name, "has_flag")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_on_side"] = obs[
                    (entry_name, "on_side")
                ]
                self.params[agent.id][f"{status}_{dif_agent.id}_tagging_cooldown"] = (
                    obs[(entry_name, "tagging_cooldown")]
                )

    def compute_rewards(self, agent_id):
        if self.reward_config[agent_id] is None:
            return 0
        # Update Prev Params
        self.prev_params[agent_id] = copy.deepcopy(self.params[agent_id])
        # Update Params
        self.update_params(agent_id)
        if self.prev_params[agent_id] == {}:
            self.prev_params[agent_id] = copy.deepcopy(self.params[agent_id])
        # Get reward based on the passed in reward function
        return self.reward_config[agent_id](
            agent_id, self.params[agent_id], self.prev_params[agent_id]
        )

    def _reset_dones(self):
        """Resets the environments done indicators."""
        dones = {}
        dones["red"] = False
        dones["blue"] = False
        dones["__all__"] = False
        return dones

    def reset(self, seed=None, return_info=False, options: Optional[dict] = None):
        """
        Resets the environment so that it is ready to be used.

        Args:
            seed (optional): Starting seed.
            options: Additonal options for resetting the environment (for now it just contains normalize)
        """
        super().reset(seed-seed, options=options)

        flag_locations = self.flags.values()

        if return_info:
            raise DeprecationWarning("return_info has been deprecated by PettingZoo -- https://github.com/Farama-Foundation/PettingZoo/pull/890")

        self.dones = self._reset_dones()

        agent_positions, agent_spd_hdg, agent_on_sides = self._generate_agent_starts(
            np.array(flag_locations)
        )

        self.reset_state({
            "agent_position": agent_positions,
            "prev_agent_position": copy.deepcopy(agent_positions),
            "agent_spd_hdg": agent_spd_hdg,
            "agent_has_flag": np.zeros(self.num_agents),
            "agent_on_sides": agent_on_sides,
            "flag_home": copy.deepcopy(flag_locations),
            "flag_locations": flag_locations,
            "flag_taken": np.zeros(2),
            "current_time": 0.0,
            "agent_captures": [
                None
            ] * self.num_agents,  # whether this agent tagged something
            "agent_tagged": [0] * self.num_agents,  # if this agent was tagged
            "agent_oob": [0] * self.num_agents,  # if this agent went out of bounds
            "dist_to_obstacles": dict()
        })
        for agent_id in self.agents:
            self.state["dist_to_obstacles"][agent_id] = [(0, 0)] * len(self.obstacles)

        for k in self.game_score:
            self.game_score[k] = 0

        self._determine_team_wall_orient()

        self.blue_team_flag_pickup = False
        self.red_team_flag_pickup = False
        self.blue_team_score = 0
        self.blue_team_score_prev = 0
        self.red_team_score = 0
        self.red_team_score_prev = 0
        self.message = ""
        self.current_time = 0
        self.reset_count += 1
        reset_obs = {agent_id: self.state_to_obs(agent_id, self.normalize) for agent_id in self.agents}

        if self.render_mode:
            self._render()
        return reset_obs

    def _generate_agent_starts(self, flag_locations):
        """
        Generates starting positions for all players based on flag locations.

        If `random_init` is `True`, then agent positons are generated randomly using their home
        flag and random offsets.

        If `random_init` is `False`, then agent positions are programmatically generated using the formula
        init_x = world_size[x] / 4
        init_y = (world_size[y] / (team_size + 1)) * ((player_id % team_size) + 1)

        This allows players to be equidistant vertically from one another and all be in a straight line 1/4 of the way
        from the boundary.

        Args:
            flag_locations: The locations of all flags.

        Returns
        -------
            Initial player positions
            Initial player orientations
            Initial player velocities (always 0)
            Intial player on_sides (also always True)

        """
        r = self.agent_radius

        agent_locations = []
        agent_spd_hdg = []
        agent_on_sides = []

        if self.random_init:
            flags_separation = self.get_distance_between_2_points(
                flag_locations[0], flag_locations[1]
            )

        for player in self.players.values():
            if self.random_init:
                max_agent_separation = flags_separation - 2 * self.flag_keepout

                # starting center point between two agents
                agent_shift = np.random.choice((-1.0, 1.0)) * np.random.uniform(
                    0, 0.5 * (0.5 * max_agent_separation + r) - r
                )

                # adjust agent max and min separation ranges based on shifted center point
                max_agent_separation = 2 * (
                    max_agent_separation / 2 - np.abs(agent_shift)
                )
                min_agent_separation = max(4 * r, 2 * (np.abs(agent_shift) + r))

                # initial agent separation
                np.random.uniform(
                    min_agent_separation, max_agent_separation
                )
                if player.team == Team.RED_TEAM:
                    other_team = Team.BLUE_TEAM
                    player.init_heading = 90
                else:
                    player.init_heading = -90
                    other_team = Team.RED_TEAM

                orientation = (
                    self.get_flags_of_team(other_team)[0]
                    - self.get_flags_of_team(player.team)[0]
                ) / flags_separation
                init_pos = flag_locations[int(player.team)] + orientation * (
                    flags_separation / 2
                )
                agent_shift = agent_shift * orientation
                init_pos += agent_shift
                player.home = init_pos
            else:
                if player.team == Team.RED_TEAM:
                    init_x_pos = self.world_size[0] / 4
                    player.heading = 90
                else:
                    init_x_pos = self.world_size[0] - self.world_size[0] / 4
                    player.heading = -90

                init_y_pos = (self.world_size[1] / (self.team_size + 1)) * (
                    (player.id % self.team_size) + 1
                )
                player.home = [init_x_pos, init_y_pos]
            player.reset()
            agent_locations.append(player.pos)
            agent_spd_hdg.append([player.speed, player.heading])
            agent_on_sides.append(True)

        return agent_locations, agent_spd_hdg, agent_on_sides

    def _get_dists_to_boundary(self):
        """
        Returns a list of numbers of length self.num_agents
        where each number is the corresponding agents min distance to a boundary wall.
        """
        distances_to_walls = defaultdict(dict)
        for player in self.players.values():
            i = player.id
            x_pos = player.pos[0]
            y_pos = player.pos[1]

            distances_to_walls[i]["left_dist"] = x_pos
            distances_to_walls[i]["right_dist"] = self.world_size[0] - x_pos
            distances_to_walls[i]["bottom_dist"] = y_pos
            distances_to_walls[i]["top_dist"] = self.world_size[1] - y_pos

        return distances_to_walls

    def _get_dists_between_agents(self):
        """Returns dictionary of distances between agents indexed by agent numbers."""
        agt_to_agt_vecs = {}

        for player in self.players.values():
            i = player.id
            agt_to_agt_vecs[i] = {}
            i_pos = player.pos
            for other_player in self.players.values():
                j = other_player.id
                j_pos = other_player.pos
                agt_to_agt_vecs[i][j] = [j_pos[0] - i_pos[0], j_pos[1] - i_pos[1]]

        return agt_to_agt_vecs

    def _get_dist_to_flags(self):
        """Returns a dictionary mapping observation keys to 2d arrays."""
        flag_vecs = {}

        for player in self.players.values():
            flag_vecs[player.id] = {}
            team_flags = player.team
            player_pos = player.pos
            # This should easily scale to N teams, but the keys will need to be updated
            # so they don't overwrite each other.
            for team in Team:
                if team != player.team:
                    opponent_flags = self.get_flags_of_team(team)
                    for idx, flag in opponent_flags:
                        flag_vecs[f"opponent_flag_{idx}_dist"] = [
                            flag.pos[0] - player_pos[0],
                            flag.pos[1] - player_pos[1]
                        ]
                else:
                    team_flags = self.get_flags_of_team(team)
                    for idx, flag in team_flags:
                        flag_vecs[f"own_flag_{idx}_dist"] = [
                            flag.pos[0] - player_pos[0],
                            flag.pos[1] - player_pos[1]
                        ]
        return flag_vecs
    
    def _get_dist_to_obstacles(self):
        """Computes the distance from each player to each obstacle"""
        dist_to_obstacles = dict()
        for player in self.players.values():
            player_pos = player.pos
            player_dists_to_obstacles = list()
            for obstacle in self.obstacles:
                dist_to_obstacle = obstacle.distance_from(player_pos, radius = self.agent_radius, heading=player.heading)
                player_dists_to_obstacles.append(dist_to_obstacle)
            dist_to_obstacles[player.id] = player_dists_to_obstacles
        self.get_state()["dist_to_obstacles"] = dist_to_obstacles
    
    def render(self):
        """Overridden method inherited from `Gym`."""
        return self._render()

    def _render(self):
        """
        Overridden method inherited from `Gym`.

        Draws all players/flags/etc on the pygame screen.
        """
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Capture the Flag")
            if self.render_mode:
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (
                        self.arena_width + 2 * self.arena_offset,
                        self.arena_height + 2 * self.arena_offset,
                    )
                )
                self.isopen = True
                self.font = pygame.font.SysFont(None, int(2*self.pixel_size*self.agent_radius))
            else:
                raise Exception(
                    "Sorry, render modes other than 'human' are not supported"
                )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state == {}:
            return None

        # arena coordinates
        if self.border_width % 2 == 0:
            top_left = (
                self.arena_offset - self.border_width / 2 - 1,
                self.arena_offset - self.border_width / 2 - 1,
            )
            top_middle = (
                self.arena_width / 2 + self.arena_offset - 1,
                self.arena_offset - 1,
            )
            top_right = (
                self.arena_width + self.arena_offset + self.border_width / 2 - 1,
                self.arena_offset - self.border_width / 2 - 1,
            )

            bottom_left = (
                self.arena_offset - self.border_width / 2 - 1,
                self.arena_height + self.arena_offset + self.border_width / 2 - 1,
            )
            bottom_middle = (
                self.arena_width / 2 + self.arena_offset - 1,
                self.arena_height + self.arena_offset - 1,
            )
            bottom_right = (
                self.arena_width + self.arena_offset + self.border_width / 2 - 1,
                self.arena_height + self.arena_offset + self.border_width / 2 - 1,
            )

        elif self.border_width % 2 != 0:
            top_left = (
                self.arena_offset - self.border_width / 2,
                self.arena_offset - self.border_width / 2,
            )
            top_middle = (self.arena_width / 2 + self.arena_offset, self.arena_offset)
            top_right = (
                self.arena_width + self.arena_offset + self.border_width / 2,
                self.arena_offset - self.border_width / 2,
            )

            bottom_left = (
                self.arena_offset - self.border_width / 2,
                self.arena_height + self.arena_offset + self.border_width / 2,
            )
            bottom_middle = (
                self.arena_width / 2 + self.arena_offset,
                self.arena_height + self.arena_offset,
            )
            bottom_right = (
                self.arena_width + self.arena_offset + self.border_width / 2,
                self.arena_height + self.arena_offset + self.border_width / 2,
            )

        # screen
        self.screen.fill((255, 255, 255))

        # arena border and scrimmage line
        draw.line(self.screen, (0, 0, 0), top_left, top_right, width=self.border_width)
        draw.line(
            self.screen, (0, 0, 0), bottom_left, bottom_right, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_left, bottom_left, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_right, bottom_right, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_middle, bottom_middle, width=self.border_width
        )
        #Draw Points Debugging
        if self.config_dict["render_field_points"]:
            for v in self.config_dict["aquaticus_field_points"]:
                draw.circle(self.screen, (128,0,128), self.world_to_screen(self.config_dict["aquaticus_field_points"][v]), 5,)

        agent_id_blit_poses = {}

        for team in Team:
            flag = self.flags[int(team)]
            teams_players = self.agents_of_team[team]
            color = "blue" if team == Team.BLUE_TEAM else "red"

            # Draw team home region
            home_center_screen = self.world_to_screen(self.flags[int(team)].home)
            draw.circle(
                    self.screen,
                    (0, 0, 0),
                    home_center_screen,
                    self.catch_radius * self.pixel_size,
                    width=round(self.pixel_size / 10),
                )

            # TODO: Add code for handling multiple flags
            for flag in self.get_flags_of_team(team):
                if not flag.taken:
                    flag_pos_screen = self.world_to_screen(flag.pos)
                    draw.circle(
                        self.screen,
                        color,
                        flag_pos_screen,
                        self.flag_radius * self.pixel_size,
                    )
                    draw.circle(
                        self.screen,
                        color,
                        flag_pos_screen,
                        (self.flag_keepout - self.agent_radius) * self.pixel_size,
                        width=round(self.pixel_size / 10),
                    )
                else:
                    # Flag is captured so draw a different shape
                    flag_pos_screen = self.world_to_screen(flag.pos)
                    draw.circle(
                        self.screen,
                        color,
                        flag_pos_screen,
                        0.55*(self.pixel_size * self.agent_radius)
                    )
            # if not self.state["flag_taken"][int(team)]:
            #     # Flag is not captured, draw normally.
            #     flag_pos_screen = self.world_to_screen(flag.pos)
            #     draw.circle(
            #         self.screen,
            #         color,
            #         flag_pos_screen,
            #         self.flag_radius * self.pixel_size,
            #     )
            #     draw.circle(
            #         self.screen,
            #         color,
            #         flag_pos_screen,
            #         (self.flag_keepout - self.agent_radius) * self.pixel_size,
            #         width=round(self.pixel_size / 10),
            #     )
            # else:
            #     # Flag is captured so draw a different shape
            #     flag_pos_screen = self.world_to_screen(flag.pos)
            #     draw.circle(
            #         self.screen,
            #         color,
            #         flag_pos_screen,
            #         0.55*(self.pixel_size * self.agent_radius)
            #     )

            # Draw obstacles:
            for obstacle in self.obstacles:
                if isinstance(obstacle, CircleObstacle):
                    draw.circle(
                        self.screen,
                        (128, 128, 128),
                        self.world_to_screen(obstacle.center_point),
                        obstacle.radius * self.pixel_size,
                        width=3
                    )
                elif isinstance(obstacle, PolygonObstacle):
                    draw.polygon(
                            self.screen,
                            (128, 128, 128),
                            [self.world_to_screen(p) for p in obstacle.anchor_points],
                            width=3,
                        )

            for player in teams_players:
                # render tagging
                player.render_tagging(self.tagging_cooldown)

                # heading
                orientation = Vector2(list(mag_heading_to_vec(1.0, player.heading)))
                ref_angle = -orientation.angle_to(self.UP)

                # transform position to pygame coordinates
                blit_pos = self.world_to_screen(player.pos)
                rotated_surface = rotozoom(player.pygame_agent, ref_angle, 1.0)
                rotated_surface_size = np.array(rotated_surface.get_size())
                rotated_blit_pos = blit_pos - 0.5*rotated_surface_size

                # blit agent onto screen
                self.screen.blit(rotated_surface, rotated_blit_pos)

                #blit agent number onto agent
                agent_id_blit_poses[player.id] = (
                    blit_pos[0] - 0.35*self.pixel_size*self.agent_radius,
                    blit_pos[1] - 0.6*self.pixel_size*self.agent_radius
                )

        # render agent ids
        if self.render_ids:
            for team in Team:
                teams_players = self.agents_of_team[team]
                font_color = "white" if team == Team.BLUE_TEAM else "black"
                for player in teams_players:
                    player_number_label = self.font.render(str(player.id), True, font_color)
                    self.screen.blit(player_number_label, agent_id_blit_poses[player.id])

        # visually indicate distances between players of both teams 
        assert len(self.agents_of_team) == 2, "If number of teams > 2, update code that draws distance indicator lines"

        for blue_player in self.agents_of_team[Team.BLUE_TEAM]:
            if not blue_player.is_tagged or (blue_player.is_tagged and blue_player.on_own_side):
                for red_player in self.agents_of_team[Team.RED_TEAM]:
                    if not red_player.is_tagged or (red_player.is_tagged and red_player.on_own_side):
                        blue_player_pos = np.asarray(blue_player.pos)
                        red_player_pos = np.asarray(red_player.pos)
                        a2a_dis = np.linalg.norm(blue_player_pos - red_player_pos)
                        if a2a_dis <= 2*self.catch_radius:
                            hsv_hue = (a2a_dis - self.catch_radius) / (2*self.catch_radius - self.catch_radius)
                            hsv_hue = 0.33 * np.clip(hsv_hue, 0, 1)
                            line_color = tuple(255 * np.asarray(colorsys.hsv_to_rgb(hsv_hue, 0.9, 0.9)))

                            draw.line(
                                self.screen, 
                                line_color,
                                self.world_to_screen(blue_player_pos),
                                self.world_to_screen(red_player_pos),
                                width=self.a2a_line_width
                            )

        if self.render_mode:
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

    def world_to_screen(self, pos):
        screen_pos = self.pixel_size * np.asarray(pos)
        screen_pos[0] += self.arena_offset
        screen_pos[1] = (
            0.5 * self.arena_height
            - (screen_pos[1] - 0.5 * self.arena_height)
            + self.arena_offset
        )

        return screen_pos

    def close(self):
        """Overridden method inherited from `Gym`."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _min(self, a, b) -> bool:
        """Convenience method for determining a minimum value. The standard `min()` takes much longer to run."""
        if a < b:
            return a
        else:
            return b