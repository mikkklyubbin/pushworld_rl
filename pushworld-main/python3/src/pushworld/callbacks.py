from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import wandb
class StatsCallback(BaseCallback):
    def __init__(self, stats_func, eval_freq=50000, verbose=0):
        super().__init__(verbose)
        self.stats_func = stats_func
        self.eval_freq = eval_freq
        self.last_eval_step = 0
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            if self.stats_func is not None:
                self.stats_func(self.model)

class MetricsCallback(BaseCallback):
    def __init__(self, eval_freq=50000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.last_eval_step = 0
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                for key, value in self.model.logger.name_to_value.items():
                    if key in ['train/entropy_loss', 'train/policy_gradient_loss', 'train/value_loss', 'train/clip_fraction', 'train/loss', 'train/explained_variance']:
                        wandb.log({key: value})