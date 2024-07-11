from pyquaticus.base_policies.shield_base import BaseAgentPolicy
import numpy as np


class BaseShield(BaseAgentPolicy):
    def __init__(
        self,
        agent_radius,
        catch_radius,
        world_size
    ):
        super().__init__(
            agent_radius,
            catch_radius,
            world_size
        )

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
        agent_obs = obs[agent_id]
        team_target_id = team_ids[1]
        defender_team_ids = [team_ids[0], team_ids[2]]
        target_opp_ids = [opponent_ids[0], opponent_ids[2]]

        opp_target_dists = {}
        opp_target_idx2id = {}
        for opp_id in target_opp_ids:
            opp_idx = opponent_ids.index(opp_id)
            opp_target_idx2id[opp_idx] = opp_id
            opp_target_dists[opp_idx] = obs[team_target_id][(f"opponent_{opp_idx}", "distance")]

        #attacker shield priority
        opp_target_idx_1 = min(opp_target_dists, key=opp_target_dists.get)
        for k in opp_target_dists.keys():
            if k != opp_target_idx_1:
                opp_target_idx_2 = k

        #assign defender to attacker1
        opp1_team_dists = {}
        for team_id in defender_team_ids:
            opp1_team_dists[team_id] = obs[team_id][(f"opponent_{opp_target_idx_1}", "distance")]
        opp1_team_defender_id = min(opp1_team_dists, key=opp1_team_dists.get)

        #assign defender to attacker2
        for team_id in defender_team_ids:
            if team_id != opp1_team_defender_id:
                opp2_team_defender_id = team_id

        opp_goal_distances = []
        agent_guard_vecs = []
        #get goal vec for defender 1
        have_action_1 = False
        if obs[opp_target_idx2id[opp_target_idx_1]]["is_tagged"]:
            if obs[opp_target_idx2id[opp_target_idx_2]]["is_tagged"]:
                action_1 = self.no_op
                have_action_1 = True
            else:
                defender1_opp_target_idx = opp_target_idx_2
        else: 
            defender1_opp_target_idx = opp_target_idx_1

        if not have_action_1:
            defender1_opp_target_id = opp_target_idx2id[defender1_opp_target_idx]
            opp_flag_dis = obs[defender1_opp_target_id][(f"opponent_{team_ids.index(team_target_id)}", "distance")]
            
            # agent to opp vec
            agent_opp_bearing = obs[opp1_team_defender_id][(f"opponent_{defender1_opp_target_idx}", "bearing")]
            agent_opp_dis = obs[opp1_team_defender_id][(f"opponent_{defender1_opp_target_idx}", "distance")]
            agent_opp_vec = agent_opp_dis * np.asarray(self.bearing_to_vec(agent_opp_bearing))

            # opp to guard vec
            opp_flag_bearing = (
                agent_opp_bearing
                + self.angle180(obs[opp1_team_defender_id][(f"opponent_{defender1_opp_target_idx}", "relative_heading")] + 180)
                + obs[defender1_opp_target_id][(f"opponent_{team_ids.index(team_target_id)}", "bearing")]
            )
            opp_goal_distances.append(opp_flag_dis)
            opp_flag_vec = opp_flag_dis * np.asarray(self.bearing_to_vec(opp_flag_bearing))

            # guard vec
            guard_vec = agent_opp_vec + 0.5*opp_flag_vec
            goal_bearing = self.vec_to_heading(guard_vec)
            goal_dis = np.linalg.norm(guard_vec)

            if goal_dis <= self.static_goal_tol:
                action_1 = self.no_op
            else:
                action_1 = self.get_action_idx(goal_bearing)


        #get goal vec for defender 2
        have_action_2 = False
        if obs[opp_target_idx2id[opp_target_idx_2]]["is_tagged"]:
            if obs[opp_target_idx2id[opp_target_idx_1]]["is_tagged"]:
                action_2 = self.no_op
                have_action_2 = True
            else:
                defender2_opp_target_idx = opp_target_idx_1
        else: 
            defender2_opp_target_idx = opp_target_idx_2

        if not have_action_2:
            defender2_opp_target_id = opp_target_idx2id[defender2_opp_target_idx]
            opp_flag_dis = obs[defender2_opp_target_id][(f"opponent_{team_ids.index(team_target_id)}", "distance")]
            
            # agent to opp vec
            agent_opp_bearing = obs[opp2_team_defender_id][(f"opponent_{defender2_opp_target_idx}", "bearing")]
            agent_opp_dis = obs[opp2_team_defender_id][(f"opponent_{defender2_opp_target_idx}", "distance")]
            agent_opp_vec = agent_opp_dis * np.asarray(self.bearing_to_vec(agent_opp_bearing))

            # opp to guard vec
            opp_flag_bearing = (
                agent_opp_bearing
                + self.angle180(obs[opp2_team_defender_id][(f"opponent_{defender2_opp_target_idx}", "relative_heading")] + 180)
                + obs[defender2_opp_target_id][(f"opponent_{team_ids.index(team_target_id)}", "bearing")]
            )
            opp_goal_distances.append(opp_flag_dis)
            opp_flag_vec = opp_flag_dis * np.asarray(self.bearing_to_vec(opp_flag_bearing))

            # guard vec
            guard_vec = agent_opp_vec + 0.5*opp_flag_vec
            goal_bearing = self.vec_to_heading(guard_vec)
            goal_dis = np.linalg.norm(guard_vec)

            if goal_dis <= self.static_goal_tol:
                action_2 = self.no_op
            else:
                action_2 = self.get_action_idx(goal_bearing)

        if agent_id == opp1_team_defender_id:
            action_idx = action_1
            #avoid walls
            action_idx = self.avoid_walls(agent_obs, action_idx)
        else:
            action_idx = action_2
            #avoid walls
            action_idx = self.avoid_walls(agent_obs, action_idx)

        return action_idx