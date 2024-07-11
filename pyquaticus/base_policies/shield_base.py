from pyquaticus.base_policies.ctf_config import ACTIONS
import math
import numpy as np


class BaseAgentPolicy:
    def __init__(
        self,
        agent_radius,
        catch_radius,
        world_size
    ):
        self.actions = ACTIONS
        self.heading_to_action_idx = {np.sign(action[1]): i for i, action in enumerate(ACTIONS) if action[0] != 0} #exclude no-op
        self.no_op = ACTIONS.index([0, 0])

        self.agent_radius = agent_radius
        self.catch_radius = catch_radius
        self.world_size = world_size

        self.wall_safety_distance = 3*agent_radius
        self.corner_safety_distance = 5*agent_radius
        self.bearing_tol = 6 #degrees
        self.static_goal_tol = agent_radius

    def compute_action(self, obs, agent_id, team_ids, opponent_ids):
        """
        **THIS FUNCTION REQUIRES UNNORMALIZED OBSERVATIONS**.

        Compute an action for the given position. This function uses observations
        of both teams.

        Args:
            obs: Unnormalized observation from the gym

        Returns
        -------
            action: The action index describing which speed/heading combo to use (assumes
            discrete action values from `ctf_gym.envs.ctf.ACTIONS`)
        """
        pass
    
    def get_action_idx(self, goal_bearing):
        if np.abs(goal_bearing) <= self.bearing_tol:
            action_idx = self.heading_to_action_idx[0]
        else:
            action_idx = self.heading_to_action_idx[
                np.sign(goal_bearing)
            ]
        return action_idx

    def avoid_walls(self, agent_obs, action_idx):
        """
        agent_obs: observations for agent
        action_idx: desired action (potentially unsafe)
        """
        if action_idx != self.no_op:
            avoid_wall_bearings = {n: np.abs(agent_obs[f"wall_{n}_bearing"]) for n in range(4) if (
                agent_obs[f"wall_{n}_distance"] <= self.wall_safety_distance
                )
            }
            avoid_corner_bearings = {n: np.abs(agent_obs[f"wall_{n}_bearing"]) for n in range(4) if (
                agent_obs[f"wall_{n}_distance"] <= self.corner_safety_distance
                )
            }
            avoid_corner_distances = {n: agent_obs[f"wall_{n}_distance"] for n in range(4) if (
                agent_obs[f"wall_{n}_distance"] <= self.corner_safety_distance
                )
            }

            avoid_wall_idx = None
            safe_parallel_wall = False
            if len(avoid_corner_bearings) == 2:
                bearings = np.asarray(list(avoid_corner_bearings.values()))
                max_wall_bearing_idx = max(avoid_corner_bearings, key=avoid_corner_bearings.get)
                min_wall_bearing_idx = min(avoid_corner_bearings, key=avoid_corner_bearings.get)
                min_wall_dis_idx = min(avoid_corner_distances, key=avoid_corner_distances.get)

                if np.all(bearings <= 90):
                    avoid_wall_idx = min_wall_dis_idx

                elif 90 < round(np.sum(bearings), 4) < 270:
                    avoid_wall_idx = max_wall_bearing_idx

                elif np.any(np.round(bearings, 4)) == 180:
                    avoid_wall_idx = min_wall_bearing_idx

            elif len(avoid_wall_bearings) == 1:
                avoid_wall_bearing = next(iter(avoid_wall_bearings.values()))
                avoid_wall_bearing = round(avoid_wall_bearing, 4)

                if avoid_wall_bearing <= 90:
                    avoid_wall_idx = next(iter(avoid_wall_bearings))


            # generate safe action
            if avoid_wall_idx is not None:
                avoid_wall_bearing = round(agent_obs[f"wall_{avoid_wall_idx}_bearing"], 4)

                if np.sign(avoid_wall_bearing) == 0:
                    action_idx = self.heading_to_action_idx[
                            np.random.choice([-1, 1])
                        ]
                else:
                    action_idx = self.heading_to_action_idx[
                        -np.sign(avoid_wall_bearing)
                    ]

        return action_idx

    def bearing_to_vec(self, heading):
        return [np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))]

    def vec_to_heading(self, vec):
        """Converts a vector to a magnitude and heading (deg)."""
        angle = math.degrees(math.atan2(vec[1], vec[0]))
        return self.angle180(angle)

    def angle180(self, deg):
        """Rotates an angle to be between -180 and +180 degrees."""
        while deg > 180:
            deg -= 360
        while deg < -180:
            deg += 360
        return deg