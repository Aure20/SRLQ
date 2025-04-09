"""
The environment is given by 
"""
from __future__ import annotations

import random
from typing import Any, ClassVar 
from copy import deepcopy
from itertools import accumulate

#from omnisafe.common.logger import Logger
from omnisafe.common.normalizer import Normalizer
from omnisafe.common.logger import Logger
from omnisafe.common.buffer.offpolicy_buffer import OffPolicyBuffer
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import OmnisafeSpace

from exllamav2.conversion.quantize import quant
from exllamav2.conversion.optimize import optimize
from exllamav2.conversion.measure import embeddings, measure_quant
from exllamav2.conversion.qparams import QParams
from exllamav2.conversion.exl_tokenize import tokenize
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer

from gymnasium import spaces
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import os
import re
import scipy.stats as stats


@env_register 
class QuantEnv(CMDP):
    """Custom environment for GPTQ quantization with delayed rewards (episodic setting)."""

    _support_envs: ClassVar[list[str]] = ["QuantEnv-v0"]
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    metadata: ClassVar[dict[str, int]] = {}
    env_spec_log: dict[str, Any]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True 

    def __init__(
        self,
        env_id: str,
        device: torch.device,
        num_envs: int = 1,
        **kwargs: Any, #model, job, batch_size
    ) -> None:
        """Initialize the GPTQ quantization environment.""" 
        self.job = kwargs.get("job")     
        self.measurement = deepcopy(self.job['measurement'])
        self.opt_strat = None
        self.exlama_config = ExLlamaV2Config()
        self.exlama_config.model_dir = self.job['in_dir']
        self.exlama_config.prepare()
        self.exlama_config.arch_compat_overrides()

        self.model = ExLlamaV2(self.exlama_config)
        gpu_split = [0.0]*6
        gpu_split[int(os.environ['LOCAL_RANK'])] = 24
        self.model.load(lazy = True, gpu_split = gpu_split) 
        self.cfg = self.model.config
        self._device = device  
        self._num_envs = num_envs

        self.cost_limit = self.get_total_bits()*10e-9 
        self.utils = self.get_utils() 

        #Normalizer treats each feature separately, initialize with the values from the measurement file
        self.obs_normalizer = Normalizer(shape = (12,), clip = 3) 
        self.get_obs(self.job['measurement'])

        self.previous_obs = None
        self.checkpoint_dir = ''

        assert self.cfg.arch.lm.attention_bias_qkv == False, "Don't support bias for now"
        assert self.cfg.arch.lm.attention_bias_o == False, "Don't support bias for now"
        assert self.cfg.arch.lm.mlp_bias == False, "Don't support bias for now"

        self._max_episode_steps = 4
         #The actor has 4 possibilities to refine the actions
        self.mini_batch_size = self.utils["num_layers"]*self.utils["lin_per_layer"] #How many linear layers to process at the same time

        self._count = 0 #Track at which point of the episode we are

        #Observation space (already normalized and over the whole model):
        # 1. Hessian incoherence
        # 2. Weight incoherence
        # 3. Frobenius upper bound
        # 4. Distortion lower bound
        # 5. Accuracy 
        # 6. MSE
        # 7. Err 1
        # 8. Err 5
        # 9. Err 10
        # 10. Current bits (of the batch being processed)
        # 11. Layer index: At which point in the model the layer is (a value between 0 and 1)
        # 12. Layer type: What kind of layer we are working with {k,q,v,o,gate,up,down}
        #------------------------------------------------------------------#
        # 13. Group 1
        # 14. Group 2
        # 15. Bit 1
        # 16. Bit 2
        # 17. Prop 1
        # 18. Prop 2
        self._observation_space = spaces.Box(
            low=np.full((self.utils["num_layers"] * self.utils["lin_per_layer"], 12), -3).flatten(),
            high=np.full((self.utils["num_layers"] * self.utils["lin_per_layer"], 12), 3).flatten()
        )  # Observations for each linear layer (normalized, cutoff at 3)

        self.group_options = [32,64,96,128]
        self.bit_options = [2,3,4,5,6,8] #Higher probability of larger value
        self.quantiles = self.get_quantiles()
        self.vars_per_layer = 2 #How many different options can a model have

        # Action space:
        # 1. Group size:self.vars_per_layer  elements from {32, 64, 128}
        # 2. Bit size: self.vars_per_layer elements from {2, 3, 4, 5, 6, 8}
        # 3. Scale: on exllama scale is fixed at 4 bits
        # 4. Bits proportion: self.vars_per_layer elements normalized later so they sum to 1, each a multiple of 0.05
        self._action_space = spaces.Box(
            low=np.zeros(self.mini_batch_size*3*self.vars_per_layer),
            high=np.concatenate([
                np.full((self.mini_batch_size, self.vars_per_layer), len(self.group_options)-1),  # group_size
                np.full((self.mini_batch_size, self.vars_per_layer), len(self.bit_options)-1),  # bits 
                np.ones((self.mini_batch_size, self.vars_per_layer))  # bit_prop
            ], axis=1).flatten()
        )  

        self.env_spec_log = {} 
        self.buffer = OffPolicyBuffer(self._observation_space,self._action_space, size = 200, batch_size=16)  
        
    def step(
        self,
        action: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        #Implement delayed rewarding becuse the reward is only given at the end of the quantization, otherwise it's 0 (same for cost) 
        self._count += 1  
        
        cost,action = self.get_strategy(action)
        reward = self.apply_quant() 

        # Return normalized observation from the strategy, important!! Need to add all the variables
        obs = self.get_obs(self.job['strategy'])
        
        self.buffer.store(obs=self.previous_obs.to(torch.float16), act=action.to(torch.float16), reward=torch.tensor(-reward), cost=torch.tensor(cost), done=torch.tensor(False), next_obs = obs)
        #If the episode is truncated by epoch then I can reuse the last observation as the new one 
        self.previous_obs = obs
 
        info = {'final_observation': obs.to(self._device)}
        local_rank = int(os.environ["LOCAL_RANK"])

        reward = reward if cost<=self.cost_limit else reward*(cost/self.cost_limit)**4 #Reduce reward proportional to the size of the model
        print(reward,cost,self._count,int(os.environ["LOCAL_RANK"]))
        
        if self._count % 4 == 0: 
            self.save_process_state(self.buffer.data, local_rank) 
             
        return obs.to(self._device), torch.tensor(-reward, device=self._device), torch.tensor(cost, device=self._device), torch.tensor(self._count==4, device=self._device), torch.tensor(False, device=self._device), info  # No reward yet


    def apply_quant(self):
        """Process all actions and compute the final reward."""
        #Store quantization and measurement 
        ppl, self.job['strategy'] = quant(deepcopy(self.job), lambda *args, **kwargs: None, self.model)  
        return ppl
    
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment."""
        if seed is not None:
            self.set_seed(random.randint(0,1000))  
        #Optimization
        opt_params = [random.random(),0,random.random(),0,0,0,0,0]
        self.job['measurement'] = deepcopy(self.measurement) 
        optimize(self.job, lambda *args, **kwargs: None, self.model, opt_params) 
        #job will have an updated strategy element
        self.previous_obs =  self.get_obs(self.job["strategy"])
        self.opt_strat = deepcopy(self.job["strategy"])
        self.job["progress"] = "tokens_cal" 

        # Tokenizer
        tokenizer = ExLlamaV2Tokenizer(self.exlama_config)
        noise_rows = self.cfg.arch.standard_calib_noise
        tokenize(self.job, lambda *args, **kwargs: None, tokenizer, noise_rows = noise_rows)
        self.job["progress"] = "embeddings" 

        #Embeddings
        embeddings(self.job, lambda *args, **kwargs: None, self.model)
        self.job["progress"] = "quant" 
        self._count = 0 
        return self.previous_obs.to(self._device), {}

    @staticmethod
    def process_entry(entry, features):
        """Extracts and appends feature values from a given dictionary entry."""
        for key, value in entry.items():
            if "proj" in key:
                layer = key.split("_")[0]
                features["incoherences_h"].append(value[layer + "_h_incoherence"])
                features["incoherences_w"].append(value[layer + "_w_incoherence"])
                features["bounds"].append(value[layer + "_ub"])
                features["distortions"].append(value[layer + "_distortion"])
                features["err_1"].append(value[layer + "_err_1"])
                features["err_5"].append(value[layer + "_err_5"])
                features["err_10"].append(value[layer + "_err_10"])
                features["bits"].append(value[layer + "_bits"])
                features["accuracies"].append(entry["accuracy"])
                features["mse"].append(entry["mse"])

    def get_obs(self, m): 
        """Get observations from a dictionary as a torch tensor with shape (batch, 11) and return them already normalized"""    
        # Initialize a dictionary to store features
        features = {
            "accuracies": [],
            "mse": [],
            "incoherences_h": [],
            "incoherences_w": [],
            "bounds": [],
            "distortions": [],
            "err_1": [],
            "err_5": [],
            "err_10": [],
            "bits": []
        } 

        for v in m.values():
            if v is None:
                break
            if isinstance(v, list): #In the case we are in the measurement part
                for config in v:
                    self.process_entry(config, features)
            else: 
                self.process_entry(v, features)

        layer_idx = self.create_repeated_range_tensor(len(features["accuracies"]), self.utils["num_layers"])
        layer_type = self.create_repeated_range_tensor(len(features["accuracies"]), self.utils["lin_per_layer"])

        # Convert features to tensors
        feature_tensors = [torch.tensor(features[key], dtype=torch.float32) for key in features] + [
            layer_idx.float(),
            layer_type.float(),
        ]

        # Concatenate all tensors along the last dimension to get shape (batch_size, 12)
        observation_tensor = torch.stack(feature_tensors, dim=1)
        observation_tensor = observation_tensor.to(self._device)

        # Normalize the tensor
        observation_tensor = self.obs_normalizer.normalize(observation_tensor)
        return observation_tensor.flatten()

    @staticmethod   
    def create_repeated_range_tensor(length: int, repeat_count: int):
        base_range = torch.arange(repeat_count)  # Create the range tensor [0, 1, ..., repeat_count-1]
        num_repeats = length // repeat_count  # Full repeats
        remainder = length % repeat_count  # Remaining elements

        repeated_part = base_range.repeat(num_repeats)  # Repeat fully
        remainder_part = base_range[:remainder]  # Add the remaining elements

        return torch.cat((repeated_part, remainder_part))

    def get_strategy(self, action: torch.Tensor):
        """Takes the action as input and updates the job dictionary"""
        #Round the proportions to the nearest 0.05 across all batches at one
        strategy_bits = 0 
        
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self._device)
        action = action.reshape(self.mini_batch_size, self.vars_per_layer*3)
    
        props = torch.nn.functional.softmax(action[:, self.vars_per_layer*2:], dim = 1).contiguous()*20  # Shape: (mini_batch_size, vars_per_layer)  
        rounded_props = torch.zeros_like(props)
        #In theory don't need this as the strategy is simply updated and
        #The values measured during quant are not lost
        #if self.current_layer_idx == 0: #Reset strategy
        #    self.job["strategy"] = {}

        if not "strategy" in self.job:
            self.job["strategy"] = {}

        bits = action[:,self.vars_per_layer:self.vars_per_layer*2].contiguous()
        groups = action[:,:self.vars_per_layer].contiguous()
        
        #Apply the adjustement to correct the behaviour
        for key,value in self.opt_strat.items():
            mode = key.split('.')[-1]
            layer_idx = int(key.split('.')[-2])
            layer_types = ["q", "k", "v", "o"] if mode == "self_attn" else ["gate", "up", "down"]
            offset = 0 if mode == "self_attn" else 4  # Offset for MLP layers
            for i,l in enumerate(layer_types):
                proj = value[f'{l}_proj']
                idx = layer_idx*self.utils["lin_per_layer"] + i + offset  
                for param in range(self.vars_per_layer):
                    if len(proj['bits']) == 1:
                        param = 0
                    bit = proj['bits'][param] #Get the current bit
                    group = proj['group_size'][str(bit)]
                    prop_idx = int(proj['bits_prop'][param]*20)
                    bits[idx,param] = torch.bucketize(bits[idx,param], self.quantiles['bits'][self.bit_options.index(bit)])
                    groups[idx,param] = torch.bucketize(groups[idx,param], self.quantiles['groups'][self.group_options.index(group)]) 
                    rounded_props[idx,param] = torch.bucketize(props[idx,param], self.quantiles['props'][prop_idx])*0.05

        #Sort them from smallest to largest 
        bits, indices = torch.sort(bits, dim=1, descending=True)
        groups = torch.gather(groups, dim=1, index=indices)

        bits = torch.round(bits).to(torch.int8).clamp(0,len(self.bit_options)-1)
        groups = torch.round(groups).to(torch.int8).clamp(0,len(self.group_options)-1) 

        # Compute rounding errors (problem, there is some approximation errors) 
        diff = 1.0 - rounded_props.sum(dim=1, keepdim=True)  # Shape: (mini_batch_size, 1)
        #Make sure it sump up to 1
        max_diff_idx = torch.argmax(torch.abs(props - rounded_props), dim=1)
        batch_indices = torch.arange(props.shape[0])  # Indices for batch layers
        rounded_props[batch_indices, max_diff_idx] += diff.squeeze()
        rounded_props = torch.gather(rounded_props, dim=1, index=indices)    

        for layer_ in range(self.utils["num_layers"]):
            k1 = self.cfg.arch.lm_prefix + "model.layers." + str(layer_) + ".self_attn"
            k2 = self.cfg.arch.lm_prefix + "model.layers." + str(layer_) + "." + self.utils["mlp_mode"]
            p1, b1 = self.get_module_strategy(bits, groups, rounded_props, "attn", layer_)
            p2, b2 = self.get_module_strategy(bits, groups, rounded_props, "mlp", layer_)   

            for (k, p) in zip((k1, k2), (p1, p2)):
                if not k in self.job["strategy"]:
                    self.job["strategy"][k] = {}
                self.job["strategy"][k].update(p)
            
            strategy_bits += b1 + b2 

        return strategy_bits*10e-9, torch.cat([groups, bits, rounded_props], dim=1).flatten() #Return modified bits

    def get_module_strategy(self, bits, groups, props, mode, layer):
        """
        Generates a module strategy dictionary for either attention (`attn`) or MLP (`mlp`) layers.
        """
        total_bits = 0
        p = {}
        layer_types = ["q", "k", "v", "o"] if mode == "attn" else ["gate", "up", "down"] #Implement case where there is no gate
        offset = 0 if mode == "attn" else 4  # Offset for MLP layers

        for i, l in enumerate(layer_types):
            idx = layer*self.utils["lin_per_layer"] + i + offset  # Adjust index for MLP layers  

            # Extract bit size, group size, bit proportion
            b = [self.bit_options[j.item()] for j in bits[idx, :]]
            g = [self.group_options[v.item()] for v in groups[idx, :]] #Only keep last group size if duplicate bits
            pr = [round(j.item(), 2) for j in props[idx, :]] 

            
            # Merge duplicates bit, remove props of 0, also sum proportions if bits are duplicated
            assert sum([abs(p) for p in pr]) == 1, pr
            corrected_g = {} 
            corrected_pr = {} 
            for bit, group, prop in zip(b, g, pr): 
                if prop < 0.05: #If proportion is small remove it 
                    continue 
                if bit in corrected_pr:
                    corrected_pr[bit] += prop
                else: 
                    corrected_pr[bit] = prop 
                corrected_g[bit] = group
                

            # Construct final dictionary
            p[f"{l}_proj"] = q_param_dict = {
                "group_size": corrected_g,
                "bits": list(corrected_g.keys()),  # Extract unique bits
                "bits_prop": list(corrected_pr.values()),
                "scale_bits": 4
            }

            # Compute total bits for the layer
            q_param = QParams.from_dict(q_param_dict)
            layer_bits = q_param.total_bits(self.utils[f"shape_{l}"], None)  # Assume no bias for now
            total_bits += layer_bits
            p[f"{l}_proj"].update({f"{l}_bits" : layer_bits})
         
        return p, total_bits


    def get_total_bits(self): 
        #Get the index of the first q layer
        first_q_layer = 0
        while not self.model.modules[first_q_layer].key.startswith(self.cfg.arch.lm_prefix + "model.layers"):
            first_q_layer += 1

        # Combined size of hidden layers
        num_layers = self.cfg.num_hidden_layers
        num_modules = num_layers * 2
        numel = sum(m.numel() for m in self.model.modules[first_q_layer : num_modules + first_q_layer]) 
        target_bpw = self.job["bits"]
        weight_budget = int(numel * target_bpw) #Total number of bits we gave at disposal for the model
        return weight_budget

    def get_utils(self):
        utils = {}
        key = self.cfg.arch.lm_prefix + "model.layers.0"
        km = self.cfg.arch.lm.keys

        key_q = key + km["attn_q"]
        key_k = key + km["attn_k"]
        key_v = key + km["attn_v"]
        key_o = key + km["attn_o"]

        has_gate = self.cfg.arch.lm.mlp_gate
        if has_gate: mlp_key_gate = km["mlp_gate"]
        mlp_key_up = km["mlp_up"]
        mlp_key_down = km["mlp_down"]

        if not self.cfg.arch.lm.is_moe:
            if has_gate: key_g = key + mlp_key_gate
            key_u = key + mlp_key_up
            key_d = key + mlp_key_down
            utils["mlp_mode"] = "mlp"
        else:
            if has_gate: key_g = key + mlp_key_gate.replace("*", "0")
            key_u = key + mlp_key_up.replace("*", "0")
            key_d = key + mlp_key_down.replace("*", "0")
            utils["mlp_mode"] = "block_sparse_moe"

        num_experts = self.cfg.num_experts if self.cfg.num_experts is not None else 1

        # Store shapes in utils instead of numel directly
        utils["shape_q"] = self.model.modules_dict[key_q].matrix_shape()
        utils["shape_k"] = self.model.modules_dict[key_k].matrix_shape()
        utils["shape_v"] = self.model.modules_dict[key_v].matrix_shape()
        utils["shape_o"] = self.model.modules_dict[key_o].matrix_shape()
        utils["shape_gate"] = self.model.modules_dict[key_g].matrix_shape() if has_gate else None
        utils["shape_up"] = self.model.modules_dict[key_u].matrix_shape()
        utils["shape_down"] = self.model.modules_dict[key_d].matrix_shape()

        # Compute numel dynamically based on stored shapes
        numel_q = utils["shape_q"][0] * utils["shape_q"][1]
        numel_k = utils["shape_k"][0] * utils["shape_k"][1]
        numel_v = utils["shape_v"][0] * utils["shape_v"][1]
        numel_o = utils["shape_o"][0] * utils["shape_o"][1]
        numel_g = utils["shape_gate"][0] * utils["shape_gate"][1] * num_experts if has_gate else 0
        numel_u = utils["shape_up"][0] * utils["shape_up"][1] * num_experts
        numel_d = utils["shape_down"][0] * utils["shape_down"][1] * num_experts

        utils["numel_attn"] = numel_q + numel_k + numel_v + numel_o
        utils["numel_mlp"] = numel_g + numel_u + numel_d
        utils["lin_per_layer"] = 4 + (3*num_experts if has_gate else 2*num_experts)
        utils["num_layers"] = self.cfg.num_hidden_layers
        utils["has_gate"] = has_gate 
        return utils

    def get_quantiles(self, a = 0, sigma=1.5):
        #Precompute quantiles
        #Use this to adjust probabilities and push towards a solution that is more similar to the exllama model, truncated gaussian probability, increase sigma for more smoothness 
        quantiles = {'bits' : [], 'groups': [], 'props': []}
        for key in quantiles.keys():
            b = len(self.bit_options)+1 if key == 'bits' else (len(self.group_options)+1 if key == 'groups' else 21)
            for mu in range(b):
                curr_quant = []
                cdf_a = stats.norm.cdf(a, loc=mu+0.5, scale=sigma)
                cdf_b = stats.norm.cdf(b, loc=mu+0.5, scale=sigma)
                for i in range(b-1):
                    cdf_x1 = stats.norm.cdf(i, loc=mu+0.5, scale=sigma)  # P(X ≤ i)
                    cdf_x2 = stats.norm.cdf(i+1, loc=mu+0.5, scale=sigma)  # P(X ≤ i)
                    curr_quant.append((cdf_x2 - cdf_x1) / (cdf_b - cdf_a))
                curr_quant = [(b-1)*q for q in list(accumulate(curr_quant))]
                quantiles[key].append(torch.tensor(curr_quant, device = self._device))
        return quantiles

    def save_process_state(self,buffer_data, local_rank):
        """
        Gather all ranks' buffer data into each process, then save it
        to a separate file per rank using torch.save().
        """

        if self.checkpoint_dir == '' and local_rank == 0:
            # Define the directory to save checkpoints
            base_dir = '/cluster/scratch/negria/checkpoints/'
            checkpoint_prefix = 'checkpoints_'

            # List all directories in the base directory
            existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(checkpoint_prefix)]

            # Extract numbers from the directory names
            existing_numbers = [int(d.split('_')[1]) for d in existing_dirs]

            # Find the smallest number and increment it
            next_number = max(existing_numbers) + 1

            # Define the new checkpoint directory
            self.checkpoint_dir = os.path.join(base_dir, f'checkpoints_{next_number}')
            
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Get world size (number of processes)
        world_size = dist.get_world_size()

        # Determine the device for the current process
        device = torch.device(f"cuda:{local_rank}")

        # Prepare a dictionary for combined data from all ranks
        combined_buffer = {}

        # For each tensor in buffer_data, gather from all ranks
        for key, tensor in buffer_data.items():
            # Move the local tensor to this rank's GPU (NCCL requires CUDA tensors)
            tensor = tensor.to(device)
            # Preallocate a list of empty tensors to receive data from all ranks
            gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
            # All processes contribute their local tensor
            dist.all_gather(gathered_tensors, tensor)
            # Concatenate them along the first dimension
            combined_buffer[key] = torch.cat(gathered_tensors, dim=0)

        # Create a state_dict with the combined buffer data
        state_dict = {'buffer_data': combined_buffer}

        if(local_rank == 0):
            # Each rank saves its own file (all files contain the combined data)
            checkpoint_path = os.path.join(self.checkpoint_dir, f"combined_buffer.pth")
            torch.save(state_dict, checkpoint_path) 

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return self._max_episode_steps

    def spec_log(self, logger: Logger) -> None:
        """Log specific environment information."""
        pass

    def set_seed(self, seed: int) -> None:
        """Set the random seed for the environment."""
        random.seed(seed)
        np.random.seed(seed)

    def render(self) -> Any:
        """Render the environment."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def close(self) -> None:
        """Close the environment."""