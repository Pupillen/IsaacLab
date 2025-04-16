脚本参见[https://docs.robotsfan.com/isaaclab/source/overview/reinforcement-learning/rl_existing_scripts.html]

```sh
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Ant-v0 --num_envs 32 --checkpoint /home/ztr/IsaacLab/logs/rl_games/ant/2025-04-14_16-14-50/nn/last_ant_ep_500_rew_77.055626.pth
```
/home/ztr/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/rl_games/algos_torch/models.py
对应模型，例如这个inference模型：
```json
        {
            "name": "Python: Play Isaac-Cartpole-Direct-v0",
            "type": "debugpy",
            "request": "launch",
            "args" : ["--task", "Isaac-Cartpole-Direct-v0", "--num_envs", "2", "--checkpoint","/home/ztr/IsaacLab/logs/rl_games/cartpole_direct/2025-04-14_21-48-36/nn/last_cartpole_direct_ep_25_rew_293.4907.pth"],
            "program": "${workspaceFolder}/scripts/reinforcement_learning/rl_games/play.py",
            "console": "integratedTerminal",
            "justMyCode": false
        },
```
```py
        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
```
查看学习的日至loss
```sh
./isaaclab.sh -p -m tensorboard.main --logdir=logs
```
