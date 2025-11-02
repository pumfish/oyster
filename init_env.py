"""Launch Isaac Sim Simulator first."""


from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create base environment
cfg = load_cfg_from_registry("Isaac-Factory-PegInsert-Direct-v0", "env_cfg_entry_point")

# spec = gym.spec("Isaac-Factory-PegInsert-Direct-v0")
# print(spec)
# breakpoint()
# print(cfg)
# print("over")
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
# wrap environment to enforce that reset is called before step
env = gym.wrappers.OrderEnforcing(env)

obs = env.reset()
breakpoint()
done = False
while True:
    # 可选：让 Gym 环境渲染（部分 IsaacLab 环境可能不需要）
    env.render()

    # 随机动作示例，可换成你的控制策略
    action = torch.tensor(env.action_space.sample(), dtype=torch.float32, device=device)
    breakpoint()
    obs, reward, done, truncated, extras = env.step(action)


    # 如果 episode 结束，重新 reset
    if done:
        obs = env.reset()