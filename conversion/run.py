import os
import json
import argparse
import shutil
import sys
import numpy as np
import torch
import torch.distributed as dist
import random

import omnisafe

from exllamav2 import ExLlamaV2, ExLlamaV2Config
from exllamav2.architecture import RopeStyle
from exllamav2.conversion.measure import measure_quant
from exllamav2.conversion.quantize import quant
from exllamav2.conversion.compile import compile_model
from exllamav2.conversion.qparams import qparams_headoptions

from env import QuantEnv


# Initialize distributed training
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])  # Get GPU rank
torch.cuda.set_device(local_rank)

parser = argparse.ArgumentParser(description = "Convert model to ExLlamaV2")
parser.add_argument("-i", "--in_dir", type = str, help = "Input directory", default = "")
parser.add_argument("-o", "--out_dir", type = str, help = "Output (working) directory")
parser.add_argument("-res", "--resume", action = "store_true", help = "Resume job from specified output directory (without specifying other options)")
parser.add_argument("-nr", "--no_resume", action = "store_true", help = "Do not resume an interrupted job (deletes all files in the output directory)")
parser.add_argument("-cf", "--compile_full", type = str, help = "Output folder for compiled model with all config/tokenizer files")
parser.add_argument("-c", "--cal_dataset", type = str, help = "Calibration dataset (.parquet file)")
parser.add_argument("-b", "--bits", type = float, default = 4.125, help = "Target bits per weight")
parser.add_argument("-ss", "--shard_size", type = float, help = "Max shard size in MB (default: 8192)", default = 8192)
parser.add_argument("-rs", "--rope_scale", type = float, help = "RoPE scaling factor")
parser.add_argument("-ra", "--rope_alpha", type = float, help = "RoPE alpha value (NTK)")
parser.add_argument("-hb", "--head_bits", type = int, default = 6, help = "Target bits per weight (head layer)")
parser.add_argument("-om", "--output_measurement", type = str, help = "Only perform measurement pass, then save measurement to the specified file")
parser.add_argument("-m", "--measurement", type = str, help = "Reuse previous measurement")
parser.add_argument("-r", "--dataset_rows", type = int, default = 100, help = "Number of rows to apply from dataset")
parser.add_argument("-mr", "--measurement_rows", type = int, default = 16, help = "Number of rows to apply from dataset when measuring")
parser.add_argument("-l", "--length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-ml", "--measurement_length", type = int, default = 2048, help = "Max no. tokens per sample when measuring")
parser.add_argument("-so", "--status_output", action = "store_true", help = "Include machine-parseable status updates in console output")
parser.add_argument("-hsol", "--hidden_state_offload_layers", type = int, default = 0, help = "Number of hidden/target states to keep in VRAM. Speed-up but increases VRAM usage")
parser.add_argument("-fst", "--fast_safetensors", action = "store_true", help = "Deprecated (does nothing)")
#parser.add_argument("-strt", "--already_started", action = "store_true")

args = parser.parse_args()
torch.set_printoptions(precision = 7, sci_mode = False, linewidth = 200) 

args.out_dir = args.out_dir+f"_{local_rank}"
# Check some args

resuming = False
if args.out_dir:
    if not args.no_resume:
        if os.path.exists(os.path.join(args.out_dir, "job_new.json")):
            resuming = True
else:
    print(" ## Please specify output/working directory (-o, --out_dir)")
    sys.exit()

if not args.in_dir and not resuming:
    print(" ## Please specify input model directory (-i, --in_dir)")
    sys.exit()

if args.length > 2048 or args.measurement_length > 2048:
    print(" !! Warning: calibration rows > 2048 tokens may result in excessive VRAM use")

if not args.head_bits in qparams_headoptions:
    print(f" ## Error: {args.head_bits} is not a supported option for head layer bitrate")
    sys.exit()

if args.output_measurement is not None and args.compile_full is not None:
    print(" ## Conflicting options: --output_measurement and --compile_full")
    sys.exit()

if args.bits < 2 or args.bits > 8:
    print(f" !! Warning: target bitrate {args.bits} will likely not be attainable")


if not os.path.exists(args.out_dir):
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f" -- Created output directory: {args.out_dir}")
    except OSError as e:
        print(f" ## Error: Failed to create output directory: {args.out_dir}")
        print(f"    {str(e)}")
        sys.exit()

# Create job


def save_job():
    global job_file, job
    with open(job_file, "w", encoding = "utf8") as f:
        f.write(json.dumps(job, indent = 4))

job_file = os.path.join(args.out_dir, "job_new.json")

if args.no_resume or not os.path.exists(job_file):

    print(f" -- Beginning new job")
    if len(os.listdir(args.out_dir)) != 0:
        print(f" !! Warning: Output directory is not empty: {args.out_dir}")

        if args.no_resume:
            print(f" !! Cleaning output directory: {args.out_dir}")
            for filename in os.listdir(args.out_dir):
                file_path = os.path.join(args.out_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

output_measurement = args.output_measurement
if output_measurement is not None:
    if os.path.isdir(output_measurement):
        output_measurement = os.path.join(output_measurement, "measurement.json")

job = {"in_dir": args.in_dir,
       "out_dir": args.out_dir,
       "cal_dataset": args.cal_dataset,
       "bits": args.bits,
       "dataset_rows": args.dataset_rows,
       "measurement_rows": args.measurement_rows,
       "length": args.length,
       "measurement_length": args.measurement_length,
       "head_bits": args.head_bits,
       "shard_size": args.shard_size if args.shard_size > 0 else 1024 ** 3,  # 1 PB = unlimited,
       "compile_full": args.compile_full,
       "rope_scale": args.rope_scale,
       "rope_alpha": args.rope_alpha,
       "output_measurement": output_measurement,
       "progress": "begin"}

if args.measurement is not None:
    with open(args.measurement, "r", encoding = "utf8") as f:
        imp_measurement = json.load(f)
        job["measurement"] = imp_measurement["measurement"]
        job["last_module_idx"] = imp_measurement["last_module_idx"]
        job["reuse_measurement"] = args.measurement

# Resume existing job

if args.no_resume or not os.path.exists(job_file):
    pass

else:
    print(f" -- Resuming job")
    if args.in_dir:
        print(f" !! Note: Overriding options with settings from existing job")

    with open(job_file, "r", encoding = "utf8") as f:
        resume_job = json.load(f)

    # Override keys in existing job
    del resume_job["out_dir"]

    job.update(resume_job)
    if "invalid" in job:
        print(" ** Error: Corrupted job")
        sys.exit()

    if job["progress"] == "finished":
        print(f" !! Job is already finished. Clear the working directory, or run this script with -nr/--no_resume to clear it automatically.")
        sys.exit()

# Feedback

print(f" -- Input: {job['in_dir']}")
print(f" -- Output: {job['out_dir']}")
if job.get("cal_dataset"):
    print(f" -- Calibration dataset: {job['cal_dataset']}, {job['dataset_rows']} / {job['measurement_rows']} rows, {job['length']} tokens per sample")
else:
    print(f" -- Using default calibration dataset")
if job["output_measurement"] is None:
    print(f" -- Target bits per weight: {job['bits']} (decoder), {job['head_bits']} (head)")
    print(f" -- Max shard size: {job['shard_size']} MB")
else:
    print(f" -- Measurement will be saved to {job['output_measurement']}")
    print(f" !! Conversion script will end after measurement pass")

if job['rope_scale']: print(f" -- RoPE scale: {job['rope_scale']:.2f}")
if job['rope_alpha']: print(f" -- RoPE alpha: {job['rope_alpha']:.2f}")

# Make sure subfolders exist

if job.get("compile_full"):
    print(f" -- Full model will be compiled to: {job['compile_full']}")
    if os.path.exists(job["compile_full"]):
        if not os.path.isdir(job["compile_full"]):
            print(f" ## Error: Output path {job['compile_full']} exists but is not a directory")
            sys.exit()
        if len(os.listdir(job["compile_full"])) > 0:
            print(f" !! Warning: Output path {job['compile_full']} exists but is not empty")

out_tensor_dir = os.path.join(job["out_dir"], "out_tensor")
if not os.path.exists(out_tensor_dir):
    os.makedirs(out_tensor_dir)

# Create config

config = ExLlamaV2Config()
config.model_dir = job['in_dir']
config.prepare()
config.arch_compat_overrides() 

# Set scaling for input model

if job["rope_scale"] is not None: config.scale_pos_emb = job["rope_scale"]
if job["rope_alpha"] is not None: config.scale_alpha_value = job["rope_alpha"]

# Create model without loading weights

model = ExLlamaV2(config)
model.load(lazy = True)

# Limit context length if necessary

if model.config.arch.lm.rope_style == RopeStyle.NONE:
    max_ctx = model.config.max_seq_len
    if job["length"] > max_ctx:
        print (f" !! Warning: Reducing calibration length to model max context: {max_ctx}")
        job["length"] = max_ctx
    if job["measurement_length"] > max_ctx:
        print (f" !! Warning: Reducing measurement calibration length to model max context: {max_ctx}")
        job["measurement_length"] = max_ctx

# Overridable settings

job["status_output"] = args.status_output

# Do the things
save_job()

model.unload()
# Training configuration

steps_per_epoch = 4

custom_cfgs = {
    "seed" : random.randint(0,1000),
    "env_cfgs": {"job": job},
    "train_cfgs": {
        "total_steps": 400,
        "device": f'cuda:{str(local_rank)}',  # Assign correct GPU ID
        "parallel": 1,
        "vector_env_nums": 1
    },
    "algo_cfgs": {
        "steps_per_epoch": steps_per_epoch,
        "update_iters": 300,
        "obs_normalize": False,
        "batch_size": 256,
        "use_exploration_noise": False,
        "exploration_noise": 0.1,
        "warmup_epochs": 0,
        "start_learning_steps": 0,
        "policy_delay": 2,
        "gamma": 0.2,
        "auto_alpha":True
    },
    "lagrange_cfgs": {
        "cost_limit":  38.92314112*4, #29.19235584
        'lagrangian_multiplier_init': 8,
        "pid_kp": 0.1,
        "pid_ki" : 0.05,
        "pid_kd": 0.05,
    },
    "model_cfgs": {
        "actor": {"hidden_sizes": [1024,768],  "lr": 0.0001, "activation": "softplus"},
        "critic": {"hidden_sizes": [1024,512],  "lr": 0.0001}
    },
    "logger_cfgs": {"save_model_freq": 2, "use_wandb": True}
}

# Initialize agent and train
agent = omnisafe.Agent("SACPID", "QuantEnv-v0", custom_cfgs=custom_cfgs)
agent.learn()

# Cleanup
dist.destroy_process_group()
