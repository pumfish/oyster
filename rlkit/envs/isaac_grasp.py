import torch
import gymnasium as gym
# from isaaclab.app import AppLauncher

# # 启动 Isaac Sim
# app_launcher = AppLauncher(headless=True)
# simulation_app = app_launcher.app

# import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.utils import load_cfg_from_registry

from . import register_env

@register_env('isaac-grasp')
class IsaacGraspWrapper(gym.Env):
    """
    Wrapper for IsaacLab Grasp task, compatible with Gymnasium.
    """

    def __init__(
            self,
            task_name="Isaac-Grasp-Cube-Franka-DR",
            device=None,
            task={},
            n_tasks=2,
            **kwargs
        ):

        from isaaclab.app import AppLauncher

        # 启动 Isaac Sim
        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import load_cfg_from_registry


        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载环境配置
        self.cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
        self.cfg.scene.num_envs = 1  # 单环境设置, agent.py要和z合并，所以应该只支持环境数为1

        # 创建 Gymnasium 环境
        self.env = gym.make(task_name, cfg=self.cfg)
        # self.env = gym.wrappers.OrderEnforcing(self.env)

        # 保留动作和观测空间
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space["policy"]

        # Pearl 相关属性
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']

    def reset(self, seed=None, options=None):
        # 返回numpy
        obs, info = self.env.reset(seed=seed, options=options)
        # Tensor -> numpy (兼容Pearl)
        pearl_obs = obs['policy']
        pearl_obs = pearl_obs.cpu().numpy()
        return pearl_obs.squeeze()

    def step(self, action):
        # numpy -> Tersor
        # 接受Pearl的输入
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device)
            action = action.unsqueeze(0)
        obs, reward, done, truncated, info = self.env.step(action)
        # Tensor -> numpy (兼容Pearl)
        pearl_obs = obs['policy'].cpu().numpy().squeeze()
        pearl_reward = reward.cpu().numpy().squeeze()
        pearl_done = done.cpu().numpy().squeeze()
        #TODO: info字典的值还没有转换
        return (
            pearl_obs,
            pearl_reward,
            pearl_done,
            info,
        )

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
        if hasattr(self, "simulation_app"):
            self.simulation_app.close()

    def sample_tasks(self, num_tasks):
        # 返回所有任务的目标位置，按照[dict(), dict(), ...]的格式返回数据
        #TODO: 物体的初始位置作为目标不太恰当
        goal_pos = self.env.env.scene['object'].data.root_pos_w
        goal_pos = goal_pos.cpu().numpy()
        tasks = [{'goal': goal_pos} for _ in range(num_tasks)]
        return tasks

    def reset_task(self, idx):
        self.reset()

    def get_all_task_idx(self):
        return range(self.cfg.scene.num_envs)


# 测试环境类
if __name__ == "__main__":
    print("🚀 初始化 Isaac-Grasp-dir 环境...")
    env = IsaacGraspWrapper()
    breakpoint()

    # 打印空间信息
    print("✅ Action space:", env.action_space)
    print("✅ Observation space:", env.observation_space)

    # reset 环境
    obs = env.reset()
    print(f"✅ Reset成功，观测维度: {obs.shape}, 示例值: {obs[:5]}")

    # 连续执行几个 step
    for i in range(5):
        action = env.action_space.sample()
        action = torch.as_tensor(action)
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: reward={reward}, done={done}")

    print("🎯 测试完成，环境运行正常。")

    print("get all task idx:", env.get_all_task_idx())