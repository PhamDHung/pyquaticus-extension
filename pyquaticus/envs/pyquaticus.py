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


import numpy as np
import copy
import random

from abc import ABC
from collections import OrderedDict
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
from pyquaticus.config import config_dict_std, ACTION_MAP
from pyquaticus.structs import Team, Player, Flag, CircleObstacle, PolygonObstacle
from pyquaticus.utils.obs_utils import ObsNormalizer
from pyquaticus.utils.utils import (
    angle180,
    closest_point_on_line,
    mag_bearing_to,
)
from typing import Dict, List, Hashable, Any

class PyQuaticusEnvBase(ParallelEnv, ABC):
    """
    ### Description.

    This class contains the base behavior for the other Pyquaticus Environment 
    implementations. The functionality of this class is shared between both the main
    Pyquaticus entry point (PyQuaticusEnv) and the PyQuaticusMoosBridge class that 
    allows deploying policies on a MOOS-IvP backend.

    The exposed functionality includes the following:
    1. converting from discrete actions to a desired speed/heading command
    2. converting from raw states in Player objects to a normalized observation space

    ### Action Space
    A discrete action space with all combinations of
    max speed, half speed; and
    45 degree heading intervals

    ### Observation Space

        Per Agent (supplied in a dictionary from agent-id to a Box):
            Opponent home flag relative bearing (clockwise degrees)
            Opponent home flag distance (meters)
            Own home flag relative bearing (clockwise degrees)
            Own home flag distance (meters)
            Wall 0 relative bearing (clockwise degrees)
            Wall 0 distance (meters)
            Wall 1 relative bearing (clockwise degrees)
            Wall 1 distance (meters)
            Wall 2 relative bearing (clockwise degrees)
            Wall 2 distance (meters)
            Wall 3 relative bearing (clockwise degrees)
            Wall 3 distance (meters)
            Scrimmage line bearing (clockwise degrees)
            Scrimmage line distance (meters)
            Own speed (meters per second)
            Own flag status (boolean)
            On side (boolean)
            Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            Is tagged (boolean)
            For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
                Bearing from you (clockwise degrees)
                Distance (meters)
                Heading of other agent relative to the vector to you (clockwise degrees)
                Speed (meters per second)
                Has flag status (boolean)
                On their side status (boolean)
                Tagging cooldown (seconds)
                Is tagged (boolean)
        Note 1 : the angles are 0 when the agent is pointed directly at the object
                 and increase in the clockwise direction
        Note 2 : the wall distances can be negative when the agent is out of bounds
        Note 3 : the boolean args Tag/Flag status are -1 false and +1 true
        Note 4: the values are normalized by default
    """
    def __init__(self, *args, num_flags:int = 1, team_size:int = 1, config_dict:Dict = config_dict_std, **kwargs):
        self.config_dict = config_dict
        self.num_flags = num_flags
        self.team_size = team_size
        self.obstacles = list()
        self.max_score = 0
        self.max_speed = 0
        self.agent_radius = 0
        self.tagging_cooldown = 0
        self.normalize = True
        # set reference variables for world boundaries
        # ll = lower left, lr = lower right
        # ul = upper left, ur = upper right
        self.world_size = np.array([0.0, 0.0])
        self.boundary_ll = np.array([0.0, 0.0], dtype=np.float32)
        self.boundary_lr = np.array([self.world_size[0], 0.0], dtype=np.float32)
        self.boundary_ul = np.array([0.0, self.world_size[1]], dtype=np.float32)
        self.boundary_ur = np.array(self.world_size, dtype=np.float32)
        self.scrimmage = 0.5*self.world_size[0] #horizontal (x) location of scrimmage line (relative to world)
        self.scrimmage_l = np.array([self.scrimmage, 0.0], dtype=np.float32)
        self.scrimmage_u = np.array([self.scrimmage, self.world_size[1]], dtype=np.float32)
        self.set_config_values(config_dict)

        # Initialize observation normalizer
        self.agent_obs_normalizer = self._register_state_elements(num_flags)

        self.players = self.create_players()
        self.action_spaces = {
            agent_id: self.get_agent_action_space() for agent_id in self.players
        }
        self.observation_spaces = {
            agent_id: self.get_agent_observation_space() for agent_id in self.players
        }

        self.flags = self.create_flags()
        self.obs_dict = OrderedDict()
        for player_id in self.players.keys():
            self.obs_dict[player_id] = OrderedDict()
        self.__state = dict()

        #Game score used to determine winner of game for MCTF competition
        #blue_captures: Represents the number of times the blue team has grabbed reds flag and brought it back to their side
        #blue_tags: The number of times the blue team successfully tagged an opponent
        #blue_grabs: The number of times the blue team grabbed the opponents flag
        #red_captures: Represents the number of times the blue team has grabbed reds flag and brought it back to their side
        #red_tags: The number of times the blue team successfully tagged an opponent
        #red_grabs: The number of times the blue team grabbed the opponents flag
        self.game_score = {'blue_captures':0, 'blue_tags':0, 'blue_grabs':0, 'red_captures':0, 'red_tags':0, 'red_grabs':0}


        # Debug assertions to make sure everything was set to the right type
        assert isinstance(self.players, dict), f"Expected players to be a dict, not {type(self.players)}"
        for player in self.players.values():
            assert isinstance(player, Player), f"Expected all players to be a Player instance, not {type(player)}"
        assert isinstance(self.flags, dict), f"Expected flags to be a dict, not {type(self.flags)}"
        for team, flags in self.flags.items():
            assert isinstance(team, Team), f"Expected all keys of flags to be a Team, not {type(team)}"
            assert isinstance(flags, list), f"Expected all teams to have a list of Flags, not {type(flags)}"
            for flag in flags:
                assert isinstance(flag, Flag), f"Expected all flags to be a Flag, not {type(flag)}"

    def create_players(self) -> Dict[Hashable, Player]:
        raise NotImplementedError

    def create_flags(self) -> Dict[Team, List[Flag]]:
        raise NotImplementedError
    
    def get_player(self, agent_id) -> Player:
        return self.players.get(agent_id, None)
    
    def get_players_of_team(self, team:Team) -> List[Player]:
        return list(filter(lambda player: player.team == team, self.players.values()))
    
    def get_flags_of_team(self, team:Team) -> List[Flag]:
        return self.flags.get(team, None)
    
    def reset(self, seed: int | None = None, options: Dict | None = None) -> tuple[Dict[str, Any], dict[str, dict]]:
        """Resets all player and flag positions"""
        if seed is not None:
            self.seed(seed=seed)
        if options is not None:
            self.set_config_values(options)

        for player in self.players.values():
            player.reset()
        for flags_of_team in self.flags.values():
            for flag in flags_of_team:
                flag.reset()

    def set_config_values(self, config_dict: Dict):
        # set reference variables for world boundaries
        # ll = lower left, lr = lower right
        # ul = upper left, ur = upper right

        self.normalize = config_dict.get("normalize", config_dict_std["normalize"])
        self.world_size = np.asarray(config_dict.get("world_size", config_dict_std["world_size"]))
        self.boundary_ll = np.array([0.0, 0.0], dtype=np.float32)
        self.boundary_lr = np.array([self.world_size[0], 0.0], dtype=np.float32)
        self.boundary_ul = np.array([0.0, self.world_size[1]], dtype=np.float32)
        self.boundary_ur = np.array(self.world_size, dtype=np.float32)
        self.scrimmage = 0.5*self.world_size[0] #horizontal (x) location of scrimmage line (relative to world)
        self.scrimmage_l = np.array([self.scrimmage, 0.0], dtype=np.float32)
        self.scrimmage_u = np.array([self.scrimmage, self.world_size[1]], dtype=np.float32)
        self.agent_radius = config_dict.get(
            "agent_radius", config_dict_std["agent_radius"]
        )
        self.tagging_cooldown = config_dict.get(
            "tagging_cooldown", config_dict_std["tagging_cooldown"]
        )

        self.max_score = config_dict.get("max_score", config_dict_std["max_score"])
        self.max_speed = config_dict.get("max_speed", config_dict_std["max_speed"])
        obstacle_params = config_dict.get("obstacles", config_dict_std["obstacles"])
        if obstacle_params is not None and isinstance(obstacle_params, dict):
                circle_obstacles = obstacle_params.get("circle", None)
                if circle_obstacles is not None and isinstance(circle_obstacles, list):
                    for param in circle_obstacles:
                        self.obstacles.append(CircleObstacle(param[0], (param[1][0], param[1][1])))
                elif circle_obstacles is not None:
                    raise TypeError(f"Expected circle obstacle parameters to be a list of tuples, not {type(circle_obstacles)}")
                poly_obstacle = obstacle_params.get("polygon", None)
                if poly_obstacle is not None and isinstance(poly_obstacle, list):
                    for param in poly_obstacle:
                        converted_param = [(p[0], p[1]) for p in param]
                        self.obstacles.append(PolygonObstacle(converted_param))
                elif poly_obstacle is not None:
                    raise TypeError(f"Expected polygon obstacle parameters to be a list of tuples, not {type(poly_obstacle)}")
        elif obstacle_params is not None:
            raise TypeError(f"Expected obstacle_params to be None or a dict, not {type(obstacle_params)}")
    
    def seed(self, seed=None):
        """
        Overridden method from Gym inheritance to set seeds in the environment.

        Args:
            seed (optional): Starting seed

        Returns:
            List of seeds used for the environment.
        """
        random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _to_speed_heading(self, action_dict):
        """
        Processes the raw discrete actions.

        Returns:
            dict from agent id -> (speed, relative heading)
            Note: we use relative heading here so that it can be used directly
                  to the heading error in the PID controller
        """
        processed_action_dict = OrderedDict()
        for player in self.players.values():
            if player.id in action_dict:
                default_action = True
                try:
                    action_dict[player.id] / 2
                except:
                    default_action = False
                if default_action:
                    speed, heading = self._discrete_action_to_speed_relheading(action_dict[player.id])
                else:
                    #Make point system the same on both blue and red side
                    if player.team == Team.BLUE_TEAM:
                        if 'P' in action_dict[player.id]:
                            action_dict[player.id] = 'S' + action_dict[player.id][1:]
                        elif 'S' in action_dict[player.id]:
                            action_dict[player.id] = 'P' + action_dict[player.id][1:]
                        if 'X' not in action_dict[player.id] and action_dict[player.id] not in ['SC', 'CC', 'PC']:
                            action_dict[player.id] += 'X'
                        elif action_dict[player.id] not in ['SC', 'CC', 'PC']:
                            action_dict[player.id] = action_dict[player.id][:-1]

                    _, heading = mag_bearing_to(player.pos, self.config_dict["aquaticus_field_points"][action_dict[player.id]], player.heading)
                    if -0.3 <= self.get_distance_between_2_points(player.pos, self.config_dict["aquaticus_field_points"][action_dict[player.id]]) <= 0.3: #
                        speed = 0.0
                    else:
                        speed = self.max_speed
            else:
                # if no action provided, stop moving
                speed, heading = 0.0, player.heading
            processed_action_dict[player.id] = np.array(
                [speed, heading], dtype=np.float32
            )
        return processed_action_dict

    def _discrete_action_to_speed_relheading(self, action):
        return ACTION_MAP[action]

    def _relheading_to_global_heading(self, player_heading, relheading):
        return angle180((player_heading + relheading) % 360)

    def _register_state_elements(self, num_obstacles):
        """Initializes the normalizer."""
        agent_obs_normalizer = ObsNormalizer(False)
        max_bearing = [180]
        max_dist = [np.linalg.norm(self.world_size) + 10]  # add a ten meter buffer
        max_dist_scrimmage = [self.scrimmage]
        min_dist = [0.0]
        max_bool, min_bool = [1.0], [0.0]
        max_speed, min_speed = [self.max_speed], [0.0]
        max_score, min_score = [self.max_score], [0.0]
        agent_obs_normalizer.register("opponent_home_bearing", max_bearing)
        agent_obs_normalizer.register("opponent_home_distance", max_dist, min_dist)
        agent_obs_normalizer.register("own_home_bearing", max_bearing)
        agent_obs_normalizer.register("own_home_distance", max_dist, min_dist)
        for flag in range(self.num_flags):
            agent_obs_normalizer.register(f"opponent_flag_{flag}_bearing", max_bearing)
            agent_obs_normalizer.register(f"opponent_flag_{flag}_distance", max_dist, min_dist)
            agent_obs_normalizer.register(f"own_flag_{flag}_bearing", max_bearing)
            agent_obs_normalizer.register(f"own_flag_{flag}_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_0_bearing", max_bearing)
        agent_obs_normalizer.register("wall_0_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_1_bearing", max_bearing)
        agent_obs_normalizer.register("wall_1_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_2_bearing", max_bearing)
        agent_obs_normalizer.register("wall_2_distance", max_dist, min_dist)
        agent_obs_normalizer.register("wall_3_bearing", max_bearing)
        agent_obs_normalizer.register("wall_3_distance", max_dist, min_dist)
        agent_obs_normalizer.register("scrimmage_line_bearing", max_bearing)
        agent_obs_normalizer.register("scrimmage_line_distance", max_dist_scrimmage, min_dist)
        agent_obs_normalizer.register("speed", max_speed, min_speed)
        agent_obs_normalizer.register("has_flag", max_bool, min_bool)
        agent_obs_normalizer.register("on_side", max_bool, min_bool)
        agent_obs_normalizer.register(
            "tagging_cooldown", [self.tagging_cooldown], [0.0]
        )
        agent_obs_normalizer.register("is_tagged", max_bool, min_bool)
        agent_obs_normalizer.register("team_score", max_score, min_score)
        agent_obs_normalizer.register("opponent_score", max_score, min_score)

        for i in range(self.team_size - 1):
            teammate_name = f"teammate_{i}"
            agent_obs_normalizer.register((teammate_name, "bearing"), max_bearing)
            agent_obs_normalizer.register(
                (teammate_name, "distance"), max_dist, min_dist
            )
            agent_obs_normalizer.register(
                (teammate_name, "relative_heading"), max_bearing
            )
            agent_obs_normalizer.register(
                (teammate_name, "speed"), max_speed, min_speed
            )
            agent_obs_normalizer.register(
                (teammate_name, "has_flag"), max_bool, min_bool
            )
            agent_obs_normalizer.register(
                (teammate_name, "on_side"), max_bool, min_bool
            )
            agent_obs_normalizer.register(
                (teammate_name, "tagging_cooldown"), [self.tagging_cooldown], [0.0]
            )
            agent_obs_normalizer.register(
                (teammate_name, "is_tagged"), max_bool, min_bool
            )

        for i in range(self.team_size):
            opponent_name = f"opponent_{i}"
            agent_obs_normalizer.register((opponent_name, "bearing"), max_bearing)
            agent_obs_normalizer.register(
                (opponent_name, "distance"), max_dist, min_dist
            )
            agent_obs_normalizer.register(
                (opponent_name, "relative_heading"), max_bearing
            )
            agent_obs_normalizer.register(
                (opponent_name, "speed"), max_speed, min_speed
            )
            agent_obs_normalizer.register(
                (opponent_name, "has_flag"), max_bool, min_bool
            )
            agent_obs_normalizer.register(
                (opponent_name, "on_side"), max_bool, min_bool
            )
            agent_obs_normalizer.register(
                (opponent_name, "tagging_cooldown"), [self.tagging_cooldown], [0.0]
            )
            agent_obs_normalizer.register(
                (opponent_name, "is_tagged"), max_bool, min_bool
            )
        
        for i in range(num_obstacles):
            agent_obs_normalizer.register(
                f"obstacle_{i}_distance", max_dist, min_dist
            )
            agent_obs_normalizer.register(
                f"obstacle_{i}_bearing", max_bearing
            )

        self._state_elements_initialized = True
        return agent_obs_normalizer

    def state_to_obs(self, agent_id, normalize=True):
        """
        Returns a local observation space. These observations are
        based entirely on the agent local coordinate frame rather
        than the world frame.
        This was originally designed so that observations can be
        easily shared between different teams and agents.
        Without this the world frame observations from the blue and
        red teams are flipped (e.g., the goal is in the opposite
        direction)
        Observation Space (per agent):
            Opponent flags relative bearing (clockwise degrees)
            Opponent flags distance (meters)
            Own flags relative bearing (clockwise degrees)
            Own flags distance (meters)
            Wall 0 relative bearing (clockwise degrees)
            Wall 0 distance (meters)
            Wall 1 relative bearing (clockwise degrees)
            Wall 1 distance (meters)
            Wall 2 relative bearing (clockwise degrees)
            Wall 2 distance (meters)
            Wall 3 relative bearing (clockwise degrees)
            Wall 3 distance (meters)
            Scrimmage line bearing (clockwise degrees)
            Scrimmage line distance (meters)
            Own speed (meters per second)
            Own flag status (boolean)
            On side (boolean)
            Tagging cooldown (seconds) time elapsed since last tag (at max when you can tag again)
            Is tagged (boolean)
            Team score (cummulative flag captures)
            Opponent score (cummulative flag captures)
            For each other agent (teammates first) [Consider sorting teammates and opponents by distance or flag status]
                Bearing from you (clockwise degrees)
                Distance (meters)
                Heading of other agent relative to the vector to you (clockwise degrees)
                Speed (meters per second)
                Has flag status (boolean)
                On their side status (boolean)
                Tagging cooldown (seconds)
                Is tagged (boolean)
        Note 1 : the angles are 0 when the agent is pointed directly at the object
                 and increase in the clockwise direction
        Note 2 : the wall distances can be negative when the agent is out of bounds
        Note 3 : the boolean args Tag/Flag status are -1 false and +1 true
        Developer Note 1: changes here should be reflected in _register_state_elements.
        Developer Note 2: check that variables used here are available to PyQuaticusMoosBridge in pyquaticus_moos_bridge.py
        """
        if not hasattr(self, '_state_elements_initialized') or not self._state_elements_initialized:
            raise RuntimeError("Have not registered state elements")

        agent = self.get_player(agent_id)
        
        own_team = agent.team
        other_team = Team.BLUE_TEAM if own_team == Team.RED_TEAM else Team.RED_TEAM
        obs = OrderedDict()
        np_pos = np.array(agent.pos, dtype=np.float32)
        # Goal flags
        for idx, flag in enumerate(self.get_flags_of_team(other_team)):
            flag_dist, flag_bearing = mag_bearing_to(np_pos, flag.pos, agent.heading)
            obs[f"opponent_flag_{idx}_distance"] = flag_dist
            obs[f"opponent_flag_{idx}_bearing"] = flag_bearing

        # Defend flags
        for idx, flag in enumerate(self.get_flags_of_team(team)):
            flag_dist, flag_bearing = mag_bearing_to(np_pos, flag.pos, agent.heading)
            obs[f"own_flag_{idx}_distance"] = flag_dist
            obs[f"own_flag_{idx}_bearing"] = flag_bearing

        # Obstacles
        for idx, obstacle in enumerate(self.obstacles):
            distance_to_obstacle, bearing_to_obstacle = obstacle.distance_from(np_pos, radius = self.agent_radius, heading=agent.heading)
            obs[f"obstacle_{idx}_distance"] = distance_to_obstacle
            obs[f"obstacle_{idx}_bearing"] = bearing_to_obstacle

        # Walls
        for i, wall in enumerate(self._walls[int(own_team)]):
            wall_closest_point = closest_point_on_line(
                wall[0], wall[1], np_pos
            )
            wall_dist, wall_bearing = mag_bearing_to(
                np_pos, wall_closest_point, agent.heading
            )
            obs[f"wall_{i}_bearing"] = wall_bearing
            obs[f"wall_{i}_distance"] = wall_dist

        # Scrimmage line
        scrimmage_line_closest_point = closest_point_on_line(
            self.scrimmage_l, self.scrimmage_u, np_pos
        )
        scrimmage_line_dist, scrimmage_line_bearing = mag_bearing_to(
            np_pos, scrimmage_line_closest_point, agent.heading
        )
        obs["scrimmage_line_bearing"] = scrimmage_line_bearing
        obs["scrimmage_line_distance"] = scrimmage_line_dist

        # Own speed
        obs["speed"] = agent.speed
        # Own flag status
        obs["has_flag"] = agent.has_flag
        # On side
        obs["on_side"] = agent.on_own_side
        # Tagging cooldown
        obs["tagging_cooldown"] = agent.tagging_cooldown
        #Is tagged
        obs["is_tagged"] = agent.is_tagged

        #Team score and Opponent score
        if agent.team == Team.BLUE_TEAM:
            obs["team_score"] = self.game_score["blue_captures"]
            obs["opponent_score"] = self.game_score["red_captures"]
        else:
            obs["team_score"] = self.game_score["red_captures"]
            obs["opponent_score"] = self.game_score["blue_captures"]

        # Relative observations to other agents
        # teammates first
        # TODO: consider sorting these by some metric
        #       in an attempt to get permutation invariance
        #       distance or maybe flag status (or some combination?)
        #       i.e. sorted by perceived relevance
        for team in [own_team, other_team]:
            dif_agents = filter(lambda a: a.id != agent.id, self.get_players_of_team(team))
            for i, dif_agent in enumerate(dif_agents):
                entry_name = f"teammate_{i}" if team == own_team else f"opponent_{i}"

                dif_np_pos = np.array(dif_agent.pos, dtype=np.float32)
                dif_agent_dist, dif_agent_bearing = mag_bearing_to(
                    np_pos, dif_np_pos, agent.heading
                )
                _, hdg_to_agent = mag_bearing_to(dif_np_pos, np_pos)
                hdg_to_agent = hdg_to_agent % 360
                # bearing relative to the bearing to you
                obs[(entry_name, "bearing")] = dif_agent_bearing
                obs[(entry_name, "distance")] = dif_agent_dist
                obs[(entry_name, "relative_heading")] = angle180(
                    (dif_agent.heading - hdg_to_agent) % 360
                )
                obs[(entry_name, "speed")] = dif_agent.speed
                obs[(entry_name, "has_flag")] = dif_agent.has_flag
                obs[(entry_name, "on_side")] = dif_agent.on_own_side
                obs[(entry_name, "tagging_cooldown")] = dif_agent.tagging_cooldown
                obs[(entry_name, "is_tagged")] = dif_agent.is_tagged

        if normalize:
            obs = self.agent_obs_normalizer.normalized(obs)

        self.obs_dict[agent.id].update(obs)
        return self.obs_dict[agent_id]

    def get_agent_observation_space(self):
        """Overridden method inherited from `Gym`."""
        if self.normalize:
            agent_obs_space = self.agent_obs_normalizer.normalized_space
        else:
            agent_obs_space = self.agent_obs_normalizer.unnormalized_space
            raise Warning(
                "Unnormalized observation space has not been thoroughly tested"
            )
        return agent_obs_space

    def get_agent_action_space(self):
        """Overridden method inherited from `Gym`."""
        return Discrete(len(ACTION_MAP))

    def _determine_team_wall_orient(self):
        """
        To ensure that the observation space is symmetric for both teams,
        we rotate the order wall observations are reported. Otherwise
        there will be differences between which wall is closest to your
        defend flag vs capture flag.

        For backwards compatability reasons, here is the order:

             _____________ 0 _____________
            |                             |
            |                             |
            |   opp                own    |
            3   flag               flag   1
            |                             |
            |                             |
            |_____________ 2 _____________|

        Note that for the other team, the walls will be rotated such that the
        first wall observation is from the wall to the right if facing away
        from your own flag.
        """

        all_walls = [
            [self.boundary_ul, self.boundary_ur],
            [self.boundary_ur, self.boundary_lr],
            [self.boundary_lr, self.boundary_ll],
            [self.boundary_ll, self.boundary_ul]
        ]

        def rotate_walls(walls, amt):
            rot_walls = copy.deepcopy(walls)
            return rot_walls[amt:] + rot_walls[:amt]

        def dist_from_wall(flag_pos, wall):
            pt = closest_point_on_line(wall[0], wall[1], flag_pos)
            dist, _ = mag_bearing_to(flag_pos, pt)
            return dist

        # short walls are at index 1 and 3
        blue_flag = self.flags[Team.BLUE_TEAM][0].home
        red_flag  = self.flags[Team.RED_TEAM][0].home
        self._walls = {}
        if dist_from_wall(blue_flag, all_walls[1]) < dist_from_wall(blue_flag, all_walls[3]):
            self._walls[int(Team.BLUE_TEAM)] = all_walls
            self._walls[int(Team.RED_TEAM)] = rotate_walls(all_walls, 2)
        else:
            assert dist_from_wall(red_flag, all_walls[1]) < dist_from_wall(red_flag, all_walls[3])
            self._walls[int(Team.RED_TEAM)] = all_walls
            self._walls[int(Team.BLUE_TEAM)] = rotate_walls(all_walls, 2)

    def state(self) -> np.ndarray:
        return np.asarray(self.obs_dict.values())
    
    def get_state(self) -> Dict:
        return self.__state
    
    def reset_state(self, new_state:Dict):
        self.__state = new_state