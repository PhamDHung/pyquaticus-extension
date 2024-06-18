import pathlib
import re

def get_alg_name_from_trainer(algo):
    class_str = str(type(algo.config))
    dot_indices = [ch.start() for ch in re.finditer('[.]',class_str)]
    alg_name = class_str[dot_indices[3]+1 : dot_indices[4]]

    return alg_name

def get_checkpoint_root_path(exp_name):
    '''
    exp_name: <alg>_<env_mode>
    '''
    parent_dir = pathlib.Path(__file__).parent.resolve()
    path_to_chkpt = parent_dir / 'output' / exp_name

    home_dir = pathlib.Path.home()

    return str(path_to_chkpt), str(home_dir)

def get_model_save_path(exp_name):
    parent_dir = pathlib.Path(__file__).parent.resolve()
    repo_root = (parent_dir / "../").resolve()

    model_name = exp_name + '_trained_model'
    path_to_model = repo_root / 'base_policies' / model_name

    return str(path_to_model)

def find_env_mode(string):
    string = str(string)
    backwards_string = string[::-1]
    end = find_nth(backwards_string, '/', 1)
    idx = end

    start_found = False
    while not start_found:
        idx += 1
        if backwards_string[idx] == '_':
            start_found = True

    return backwards_string[end+1:idx][::-1]

def find_nth(string, substring, n):
    start = string.find(substring)
    while start >= 0 and n > 1:
        start = string.find(substring, start+len(substring))
        n -= 1
    return start