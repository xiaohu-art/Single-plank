# Quadrupedal Robot Track #

Project link: [https://github.com/xiaohu-art/Single-plank.git](https://github.com/xiaohu-art/Single-plank.git)

## 1. Create the single-plank bridge environment ##
The generated single-plank terrain are as below, the left one is the uniform radius terrain and the right one is the random radius terrain indicating the curvature change for different part of the cylinder. 

<img src="./pic/uniform_radius.jpeg" alt="image" style="zoom:50%;" /> <img src="./pic/random_radius.jpeg" alt="image" style="zoom:50%;" />

Based on the other functions in `terrain.py`, the single-plank terrain is generated as below. Specifically, the `curve_height` is multiplied by 10 to mitigate the effect of `vertical_scale = 0.005`. Besides, I also modified the `env_origin` to initialize the quadrupedal robot position at the boundary of the terrain. Position perturbations are not applied as the bridge is too narrow too support random position initialization.

```python
def single_plank_terrain(terrain, radius, max_height=None):

    # max_height: to pull all the points to the same height, only used for random radius
    # radius: list of radius for each x in shape of (terrain.length,)

    terrain.height_field_raw[:,:] = -200
    center = terrain.length // 2
    if isinstance(radius, int):
        min_radius = radius
        radius = np.ones(terrain.length) * radius
    elif isinstance(radius, np.ndarray):

        min_radius = np.min(radius)
    else:
        raise ValueError("radius should be either int or list of int")
        
    for x in range(terrain.length):
        for y in range(terrain.width):
            distance = abs(y-center)
            if distance <= radius[x]:
                curve_height = np.sqrt(radius[x]**2 - distance**2) * 10
                terrain.height_field_raw[x,y] = curve_height

    delta = max_height - terrain.height_field_raw[:, center]
    for x in range(terrain.length):
        for y in range(terrain.width):
            distance = abs(y-center)
            if distance <= radius[x]:
                terrain.height_field_raw[x,y] += delta[x]

    terrain.height_field_raw[:, :center-min_radius] = -200
    terrain.height_field_raw[:, center+min_radius:] = -200
```
###### PS: failure experience

*Actually I chose the invalid method at the beginning of generating terrain. Considering to use cylinder as bridge, my first thought was to import urdf file or create asset like this (I even created a urdf file):* 

<img src="./pic/cylinder.png" style="zoom:33%;" />

```python
# create cylinder asset
radius = 0.1
length = 8
density = 1000
cylinder_asset_options = gymapi.AssetOptions()
cylinder_asset_options.density = density
cylinder_asset = self.gym.create_capsule(self.sim, radius, length, cylinder_asset_options)

cylinder_ahandle = self.gym.create_actor(env_handle, cylinder_asset, start_pose, "cylinder", i, False, 0)
```

*and I realized that there are two actors, cylinder and robot, in each environment. So I modified the root states as:*

```python
actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
actor_per_env = actor_root_state // self.num_envs

self.root_states = gymtorch.wrap_tensor(actor_root_state)
self.root_states = self.root_states[::actor_per_env]
```
*And I got an uninformative error "RuntimeError: CUDA error: an illegal memory access was encountered" when indicing the `self.root_states[env_ids]`. However I couldn't find the reason so I generated the cylinder directly based on mesh.* 

## 2. Reference trajectories by MPC

Here is a visualization of catwalk gait. I directly modified the `hip_offsets` in `raibert_swing_leg_controller.py` as I didn't find the constrains equations in the original code. 

![](./pic/catwalk.gif)

Based on the setting of base x linear velocity in [0, 1], base y linear velocity and base angular velocity are both equal to 0, I collected state-action transitions at x velocity in [0.2, 0.4, 0.6, 0.8, 1.0], each lasting 5 secs.

The correspondence between observations and actions in the two different simulators is shown in the following table (height map is ignored):

| IsaacGym          | PyBullet                              | dimension | Note                                        |
| ----------------- | ------------------------------------- | --------- | ------------------------------------------- |
| base_lin_vel      | robot.GetBaseVelocity()               | 3         |                                             |
| base_ang_vel      | robot.GetBaseRollPitchYawRate()       | 3         |                                             |
| projected_gravity | [0, 0, -9.81]                         | 3         | fixed                                       |
| command           | lin_speed[0], lin_speed[1], ang_speed | 3         |                                             |
| dof_pos           | robot.GetMotorAngles()                | 12        |                                             |
| dof_vel           | robot.GetMotorVelocities()            | 12        |                                             |
| last_action       | ---                                   | 12        | initialize with zero and update iteratively |
| action            | robot.GetMotorAngles()                | 12        | target position after robot step            |

Notably, the joint angles' order are also different in the two simulators. So we need to reorder the joint angles from pybullet to IsaacGym as we create the reference datasets (details in `reference/dataset.py`).


###### PS: failure experience
*The dual graphics adapters (intel and nvidia) troubled me a lot becuase the IsaacGym GUI and the PyBullet GUI seems like they cannot be utilized meanwhile. After I visualize the demo catwalk in pybullet, the isaacgym GUI corrupted totally showing no response, at that time I forgot to visualize the environment.*
*It lead re installation for my Ubuntu OS after I tried thousand of methods and reminded me to generate reference data in IsaacGym just like the coding test mentioned.*
