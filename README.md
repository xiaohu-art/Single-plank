# Quadrupedal Robot Track #

Project link: [https://github.com/xiaohu-art/Single-plank.git](https://github.com/xiaohu-art/Single-plank.git)

## 1. Create the single-plank bridge environment ##
The generated single-plank terrain are as below, the left one is the uniform radius terrain and the right one is the random radius terrain indicating the curvature change for different part of the cylinder. 

<figure class="third">
    <center>
    <img src="./pic/uniform_radius.jpeg" alt="image" style="zoom:25%;" /><img src="./pic/random_radius.jpeg" alt="image" style="zoom:25%;" />
    </center>
</figure>



Based on the other functions in `terrain.py`, the single-plank terrain is generated as below. Specifically, the `curve_height` is multiplied by 10 to mitigate the effect of `vertical_scale = 0.005`. Besides, I also modified the `env_origin` to initialize the quadrupedal robot position at the boundary of the terrain. Position perturbations are not applied as the bridge is too narrow too support random position initialization.

```python
def single_plank_terrain(terrain, radius, max_height=None):

    # max_height: to pull all the points to the same height, only used for random radius
    # radius: list of radius for each x in shape of (terrain.length,)

    # ...
    # Partial code
    for x in range(terrain.length):
        for y in range(terrain.width):
            distance = abs(y-center)
            if distance <= radius[x]:
                curve_height = np.sqrt(radius[x]**2 - distance**2) * 10
                terrain.height_field_raw[x,y] = curve_height
    # ...
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

*I have realized that there are two actors, cylinder and robot, in each environment. So I modified the root states as:*

```python
actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
actor_per_env = actor_root_state // self.num_envs

self.root_states = gymtorch.wrap_tensor(actor_root_state)
self.root_states = self.root_states[::actor_per_env]
```
*And I got an uninformative error "RuntimeError: CUDA error: an illegal memory access was encountered" when indicing the `self.root_states[env_ids]`. However I couldn't find the reason so I generated the cylinder directly based on mesh.* 

## 2. Reference trajectories by MPC

Here is a visualization of catwalk gait. I directly modified the `hip_offsets` in `raibert_swing_leg_controller.py` as I didn't find the constrains equations in the original code. 

<center>
<figure>
    <img src="./pic/catwalk.gif" alt="image"  />
    <figcaption>Visualization of catwalk gait</figcaption>
</figure>
</center>

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

## 3. Train the IL and RL agent
I trained a discriminator network for adversarial imitation learning following the setting in [HumanMimic](https://arxiv.org/pdf/2309.14225).

The Wasserstein discriminator $D_{\theta}(\cdot)$ is a series of linear layers coming with loss:

$
argmin_{\theta} -\mathbb{E}_{x \sim \mathcal{P_r}}[tanh(\eta D_{\theta}(x))] + \mathbb{E}_{\widetilde{x} \sim \mathcal{P_g}}[tanh(\eta D_{\theta}(\widetilde{x}))] + \lambda \mathbb{E}_{\hat{x} \sim \mathcal{P_{\hat{x}}}}[(||\nabla_{\hat{x}} D_{\theta}(\hat{x})||_2 - 1)^2]
$

where $\hat{x} = \alpha x + (1-\alpha) \widetilde{x}$ are samples obtained through random interpolation between the reference samples $x$ and the generated samples $\widetilde{x}$; $\eta$ means softer constrains to unbounded values of linear outputs; $\lambda$ is the weight of the gradient penalty term. For imitation reward, it is defined as $r^{IL} = e^{D_{\theta}(\widetilde{x})}$.

The architecture and hyperparameters of the discriminator are shown below:
<table>
    <tbody>
        <tr>
            <td> Architecture </td>
            <td> [48+12, 256, 128, 1] </td>
        </tr>
        <tr>
            <td> &#951 </td>
            <td> 0.3 </td>
        </tr>
        <tr>
            <td> &#955 </td>
            <td> 10 </td>
        </tr>
        <tr>
            <td> optimizer </td>
            <td> Adam </td>
        </tr>
        <tr>
            <td> learning rate </td>
            <td> 1e-4 </td>
        </tr>
    </tbody>
</table>

Combining with the imitation reward $r = r^{IL} + r^{RL}$, the RL agent is trained with PPO algorithm. The architecture and hyperparameters of the policy network are the same as the origin code in but `num_envs = 4` for debugging. 

You can check the first training cureve in the `./logs/go2_single_plank_imitation_num_envs_4` and the design above didn't work well. It was observed that the discriminator could not distinguish the experts' transitions and the agent transitions in the early stage, allowing $r^{IL}$ so high (up to 137) to dominate the learning process, which leads to instability.

In addition, the number of samples from reference dataset is `num_iter * num_epoch * batch_size * num_mini_batch = 3000 * 5 * 24 * 4 = 1.44M`, and the length of reference dataset is 12500. The discriminator is trained equivalently to around 100 epochs, which might also be inefficient.

## 4. Scaling to multiple environments and coefficients scheduling
After simple visualizatioon of the simulator, I build docker caontainer and run the training with 4096 envs.
Meanwhile, I apply clip and a cosine coefficients scheduling in the format $r =\lambda \cdot \min\{1, r^{IL}\} + r^{RL}$.