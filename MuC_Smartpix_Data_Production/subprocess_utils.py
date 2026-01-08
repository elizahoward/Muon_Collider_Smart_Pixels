import subprocess
import multiprocessing

def run_executable(executable_path, options):
    command = [executable_path] + options
    subprocess.run(command)

def run_commands(commands):
    for command in commands:
        print(command)
        if "PixelAV" in command[0]:
            subprocess.run(command[1:], cwd=command[0])
        elif "ddsim" in command[0] or "pgun_lcio.py" in command[1] or "make_tracklists.py" in command[1]:
            subprocess.run(command, env=env)
        else:
            subprocess.run(command)

def pool_commands(commands, num_cores):
    # Create a pool of processes to run in parallel
    pool = multiprocessing.Pool(num_cores)
    
    # Launch the executable N times in parallel with different options
    # pool.starmap(run_executable, [(path_to_executable, options) for options in options_list])
    #print(commands) 
    pool.starmap(run_commands, commands)
    
    # Close the pool of processes
    pool.close()
    pool.join()

def get_env_from_setup(script_path):
    # Launch a new bash shell, source the script, and print the environment
    command = f"bash -c 'source {script_path} >/dev/null 2>&1 && env'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to source {script_path}:\n{result.stderr}")

    env = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            env[key] = value

    return env