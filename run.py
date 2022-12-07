import time
import random
import numpy as np
import pybullet as p

from gym.envs.CtrlAviary import CtrlAviary
from gym.utils.Logger import Logger
from gym.utils.utils import sync
from gym.utils.enums import DroneModel
from quad3d_ctrl import Quad3D

DURATION = 50
"""int: The duration of the simulation in seconds."""
GUI = True
"""bool: Whether to use PyBullet graphical interface."""
RECORD = False
"""bool: Whether to save a video under /files/videos. Requires ffmpeg"""

if __name__ == "__main__":

    # Create the environment
    env = CtrlAviary(num_drones=1,
                     drone_model=DroneModel.CF2P,
                     initial_xyzs=np.array([ [.0, .0, .15]]), #, [-.3, .0, 1.], [.3, .0, .15] 
                     gui=GUI,
                     record=RECORD
                     )
    PYB_CLIENT = env.getPyBulletClient()

    # Initialize the LOGGER  
    LOGGER = Logger(logging_freq_hz=env.SIM_FREQ,
                    num_drones=1,
                    )

    # Initialize the CONTROLLERS
    CTRL_0 = Quad3D(env=env)

    # Initialize the action
    action = {}
    obs = env.reset()
    state = obs["0"]["state"]
    action["0"] = CTRL_0.compute_control(current_position=state[0:3],
                                         current_velocity=state[10:13],
                                         current_rpy=state[7:10],
                                         target_position=state[0:3],
                                         target_velocity=np.zeros(3),
                                         target_acceleration=np.zeros(3)
                                         )

    # Initialize the target trajectory   
    TARGET_POSITION = np.array([[1.0*np.sin(0.002*i), 1.0*np.cos(0.001*i), 0.7+0.5*np.cos(0.001*i)] for i in range(DURATION*env.SIM_FREQ)])
    TARGET_VELOCITY = np.zeros([DURATION * env.SIM_FREQ, 3])
    TARGET_ACCELERATION = np.zeros([DURATION * env.SIM_FREQ, 3])

    # Derive the target trajectory to obtain target velocities and accelerations
    TARGET_VELOCITY[1:, :] = (TARGET_POSITION[1:, :] - TARGET_POSITION[0:-1, :]) / env.SIM_FREQ
    TARGET_ACCELERATION[1:, :] = (TARGET_VELOCITY[1:, :] - TARGET_VELOCITY[0:-1, :]) / env.SIM_FREQ

    # Run the simulation    
    START = time.time()
    for i in range(0, DURATION*env.SIM_FREQ):

        #  Secret control performance booster
        # if i/env.SIM_FREQ>3 and i%30==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [random.gauss(0, 0.3), random.gauss(0, 0.3), 3], p.getQuaternionFromEuler([random.randint(0, 360),random.randint(0, 360),random.randint(0, 360)]), physicsClientId=PYB_CLIENT)

        # Step the simulation 
        obs, _, _, _ = env.step(action)

        # Compute control for drone 0
        state = obs["0"]["state"]
        action["0"] = CTRL_0.compute_control(current_position=state[0:3],
                                             current_velocity=state[10:13],
                                             current_rpy=state[7:10],
                                             target_position=TARGET_POSITION[i, :],
                                             target_velocity=TARGET_VELOCITY[i, :],
                                             target_acceleration=TARGET_ACCELERATION[i, :]
                                             )
        # Log drone 0 
        LOGGER.log(drone=0, timestamp=i/env.SIM_FREQ, state=state)

        # Printout
        if i%env.SIM_FREQ == 0:
            env.render()

        # Sync the simulation
        if GUI:
            sync(i, START, env.TIMESTEP)

    # Close the environment 
    env.close()

    # Save the simulation results 
    LOGGER.save()

    # Plot the simulation results 
    LOGGER.plot()
