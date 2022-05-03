import numpy as np
import math
import matplotlib.pyplot as plt
import gym
from gym import spaces

import io

# An arm consists of a set of joints, each of which is at a certain angle relative to the previous
# Each joint connects between limbs of certain length
class Manipulator2DEnv(gym.Env):
    def __init__(self, num_joints, arm_lengths, max_steps=100, config=None):
        self.num_joints = num_joints
        # each joint is limited in the angles it can take, we set this from -90 to 90 degrees
        self.min_angle = np.zeros(self.num_joints) - 90
        self.max_angle = np.zeros(self.num_joints) + 90
        # start_point is the starting (x,y) coordinate of the first limb/joint
        self.start_point = [0,0]
        # arm_lengths is the length of each arm limb
        self.arm_lengths = arm_lengths
        # config is the current angle each joint is at
        # the angle is relative to the previous joint
        if config is not None: self.config = config
        else: self.config = np.zeros(self.num_joints)
        
        # workspace_config is the (x,y) coordinate of each limb end
        self.workspace_config = np.zeros((self.num_joints + 1, 2))
        self.update_workspace_config()
        self.tip_position = self.workspace_config[-1]

        # self.goal = (10,-5,2)
        self.obstacles = [np.array([7.0, 3.0, 2.0])]
        self.movable_objects = [np.array([10.0,-5.0,2.0])]
        
        self.artist = DrawArm(self)

        # GYM STUFF
        self.max_steps = max_steps
        # for each joint we can either increase or decrease the angle (by 1 degree)
        self.action_space = spaces.Discrete(2 * self.num_joints)
        # our observation is the angle for each joint 
        # (TODO: along with the position of all movable objects)
        self.observation_space = spaces.Box(
            low=-90,
            high=90,
            shape=(self.num_joints,),
            dtype='int8'
        )
        
        # Initialize the state
        self.reset()

    def to_list(self):
        return list(self.config)
    
    def reset(self):
        self.config = np.zeros(self.num_joints)
        self.update_workspace_config() # do we care about workspace config?
        self.step_count = 0
        return self.to_list()

    def step(self, action):
        self.step_count += 1
        old_config = np.copy(self.config)
        old_w_config = np.copy(self.workspace_config)

        # if action is even we subtract from angle, if odd we add
        sign = 2*(action % 2) - 1
        self.config[action // 2] += sign

        # angle needs to be between -90 and 90
        self.config[action // 2] = max(-90, self.config[action // 2])
        self.config[action // 2] = min(90, self.config[action // 2])

        # update workspace config
        self.update_workspace_config()
        
        # check obstacle collision, go back to previous location if collision
        col, _, _ = self.detect_collision(self.obstacles)
        if col:
            self.config = old_config
            self.workspace_config = old_w_config
        else:
            # update location of movable objects
            col, vec, ob_idx = self.detect_collision(self.movable_objects)
            while col: # ideally this should be one and done, no loop
                self.movable_objects[ob_idx][:2] += vec*0.1 # should remove this magic constant but ok
                col, vec, ob_idx = self.detect_collision(self.movable_objects)
        
        done = False
        if self.step_count >= self.max_steps:
            done = True

        reward = 0

        # self.artist.update_draw()
        return self.to_list(), reward, done, {}

    def update_config(self, new_config):
        self.config = new_config
        self.update_workspace_config()
        
    # Compute (x,y) coordinates of arm joints - FORWARD KINEMATICS
    # Also updates the location of arm tip
    def update_workspace_config(self):
        x = self.start_point[0]
        y = self.start_point[1]
        self.workspace_config[0] = [x,y]
        rel_angle = 0
        for index, (angle, length) in enumerate(zip(self.config, self.arm_lengths)): 
            rel_angle += angle
            x += length * math.cos(math.radians(rel_angle)) 
            y -= length * math.sin(math.radians(rel_angle))
            self.workspace_config[index+1] = [x,y]
        self.tip_position = self.workspace_config[-1]
        return self.workspace_config
                  
    # Check if a configuration is within limits
    def in_boundary(self):
        for angle in self.config:
            for min_a, max_a in zip(self.min_angle, self.max_angle):
                if angle < min_a or angle > max_a:
                    return False
        return True

    # detects whether an arm is in collision with circular obstacles
    # uses the workspace configuration
    def detect_collision(self, obstacles):
        for obidx, ob in enumerate(obstacles):
            center = ob[:2]
            radius = ob[2]
            for i in range(self.num_joints):
                start = self.workspace_config[i]
                end = self.workspace_config[i+1]

                # check distance between a line segment (the arm link) and a point (center)
                s_e = end - start
                e_c = center - end
                s_c = center - start

                # if the center is below the line
                if np.dot(s_c, s_e) < 0:
                    distance = np.linalg.norm(s_c)
                elif np.dot(e_c, s_e) > 0:
                    distance = np.linalg.norm(e_c)
                else:
                    cross = np.linalg.norm(np.cross(s_e, s_c))
                    # this should be the same as self.arm_lengths[i]
                    joint_length = np.linalg.norm(s_e) 
                    distance = cross / joint_length
                # the distance between the center and the link should be greater than radius
                if distance <= radius:
                    vec = e_c + s_c
                    return True, vec / np.linalg.norm(vec), obidx
        return False, -1, -1

    # Check whether the tip of the arm is close to the goal location
    # Notice that the goal is defined in WORKSPACE - it is an (x,y) point
    def reached_goal(self, goal):
        return np.linalg.norm(self.tip_position - np.array(goal[:2])) <= goal[2]
        
class DrawArm:
    def __init__(self, arm):
        self.arm = arm
        
        self.fig = plt.figure(figsize=(4, 4))
        self.axis = plt.axis((-25,25,-25,25))

        # draw arm
        self.line, = plt.plot(arm.workspace_config[:,0], arm.workspace_config[:,1])
        # self.line.set_animated(True)

        # add obstacles
        for ob in self.arm.obstacles:
            circle = plt.Circle(ob[:2], ob[2], color="r")
            plt.gcf().gca().add_artist(circle)
        self.mov_ob = []
        for mov_ob in self.arm.movable_objects:
            circle = plt.Circle(mov_ob[:2], mov_ob[2], color="g")
            plt.gcf().gca().add_artist(circle)
            self.mov_ob.append(circle)

        # self.goal_circle = plt.Circle(goal[:2], goal[2], color="g")
        # plt.gcf().gca().add_artist(self.goal_circle)

        self.text_collide = plt.text(-20, 20, "Collision: False")
        # self.text_goal = plt.text(-20, 15, "Reached Goal: False")

    def update_draw(self):
        # if joint is not None:
        #     self.arm.config[joint] = angle
        #     self.arm.update_workspace_config()
        
        if False: # this is now done in step...
            col, vec, ob_idx = self.arm.detect_collision([self.goal])
            while col:
                # self.goal_circle.set_radius(1)
                np_goal = np.array(self.goal[:2])
                #vec = np_goal - self.arm.tip_position
                #vec = vec / np.linalg.norm(vec)
                
                new_center = np_goal + vec*0.1
                self.goal_circle.set(center=new_center)
                self.goal = (new_center[0], new_center[1], self.goal[2])
                
                col, vec, ob_idx = self.arm.detect_collision([self.goal])
                
        self.line.set_xdata(self.arm.workspace_config[:,0])
        self.line.set_ydata(self.arm.workspace_config[:,1])
        for i, m in enumerate(self.mov_ob):
            m.set(center=self.arm.movable_objects[i][:2])
        # self.text_goal.set_text("Reached Goal: {}".format(self.arm.reached_goal(self.goal)))
        self.text_collide.set_text("Collision: {}".format(self.arm.detect_collision(self.arm.obstacles)[0]))
        plt.title("Current arm angles: {}\nTip position: {}".format(self.arm.config, np.round(self.arm.tip_position)))
        
        # plt.savefig(f"tmp_{idx}.png")
        # plt.draw()
        # print(plt.gcf())

        # io_buf = io.BytesIO()
        # self.fig.savefig(io_buf, format='raw', dpi=400)
        # io_buf.seek(0)
        # img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        #                     newshape=(int(self.fig.bbox.bounds[3]), int(self.fig.bbox.bounds[2]), -1))
        # io_buf.close()
        # return img_arr
        # return plt.plot(self.arm.workspace_config[:,0], self.arm.workspace_config[:,1],c="black")[0]
        return self.line

# def save_anim(self):
#     import matplotlib.animation as animation
#     ani = animation.ArtistAnimation(self.fig, frames, interval=50, blit=True, repeat_delay=1000)

