from pyquaticus.config import ACTION_MAP as PYQUATICUS_ACTIONS
from pyquaticus.config import config_dict_std


MAX_SPEED = config_dict_std["max_speed"]

########### CUSTOM ACTIONS ##########
######### DO NOT MODIFY ORDER! ######
ACTIONS = [
    [MAX_SPEED, 135],
    [MAX_SPEED, 0],
    [MAX_SPEED, -135]
]
# add a none action
ACTIONS.append([0.0, 0.0])
#####################################


# Map custom actions to pyquaticus actions
PYQUATICUS_ACTION_MAP = {i:PYQUATICUS_ACTIONS.index(a) for i, a in enumerate(ACTIONS)}