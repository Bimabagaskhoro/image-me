#!/usr/bin/env python3
"""
Standalone script for image model training (SDXL or Flux)
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time

import toml
import yaml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    """Merge default config with model-specific overrides."""
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None
def get_config_for_model(lrs_config: dict, model_name: str) -> dict:
    """Get configuration overrides based on model name."""
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    """Load the appropriate LRS configuration based on model type and training type"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hash_model(model_name)
            lrs_settings = get_config_for_model(lrs_config, model_hash)

            if lrs_settings:
                for optional_key in [
                    "max_grad_norm",
                    "prior_loss_weight",
                    "max_train_epochs",
                    "train_batch_size",
                    "optimizer_args",
                    "unet_lr",
                    "text_encoder_lr",
                    "noise_offset",
                    "min_snr_gamma",
                    "seed",
                    "lr_warmup_steps",
                    "loss_type",
                    "huber_c",
                    "huber_schedule",
                ]:
                    if optional_key in lrs_settings:
                        config[optional_key] = lrs_settings[optional_key]
            else:
                print(f"Warning: No LRS configuration found for model '{model_name}'", flush=True)
        else:
            print("Warning: Could not load LRS configuration, using default values", flush=True)

        # Update config
        network_config_person = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 467,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 467,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 699,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 467,
            "dataautogpt3/TempestV0.1": 456,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 467,
            "fluently/Fluently-XL-Final": 228,
            "mann-e/Mann-E_Dreams": 456,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 228,
            "recoilme/colorfulxl": 228,
            "zenless-lab/sdxl-aam-xl-anime-mix": 456,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
            "zenless-lab/sdxl-anything-xl": 228,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
            "Corcelio/mobius": 228,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
        }

        network_config_style = {
            "stabilityai/stable-diffusion-xl-base-1.0": 235,
            "Lykon/dreamshaper-xl-1-0": 235,
            "Lykon/art-diffusion-xl-0.9": 235,
            "SG161222/RealVisXL_V4.0": 235,
            "stablediffusionapi/protovision-xl-v6.6": 235,
            "stablediffusionapi/omnium-sdxl": 235,
            "GraydientPlatformAPI/realism-engine2-xl": 235,
            "GraydientPlatformAPI/albedobase2-xl": 235,
            "KBlueLeaf/Kohaku-XL-Zeta": 235,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
            "John6666/nova-anime-xl-pony-v5-sdxl": 235,
            "cagliostrolab/animagine-xl-4.0": 235,
            "dataautogpt3/CALAMITY": 235,
            "dataautogpt3/ProteusSigma": 235,
            "dataautogpt3/ProteusV0.5": 235,
            "dataautogpt3/TempestV0.1": 228,
            "ehristoforu/Visionix-alpha": 235,
            "femboysLover/RealisticStockPhoto-fp16": 235,
            "fluently/Fluently-XL-Final": 235,
            "mann-e/Mann-E_Dreams": 235,
            "misri/leosamsHelloworldXL_helloworldXL70": 235,
            "misri/zavychromaxl_v90": 235,
            "openart-custom/DynaVisionXL": 235,
            "recoilme/colorfulxl": 235,
            "zenless-lab/sdxl-aam-xl-anime-mix": 235,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
            "zenless-lab/sdxl-anything-xl": 235,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
            "Corcelio/mobius": 235,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
            "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
        }

        config_mapping = {
            228: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": []
            },
            235: {
                "network_dim": 32,
                "network_alpha": 32,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            456: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": []
            },
            467: {
                "network_dim": 64,
                "network_alpha": 64,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
            699: {
                "network_dim": 96,
                "network_alpha": 96,
                "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
            },
        }

        config["pretrained_model_name_or_path"] = model_path
        config["train_data_dir"] = train_data_dir
        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        config["output_dir"] = output_dir

        if model_type == "sdxl":
            if is_style:
                network_config = config_mapping[network_config_style[model_name]]
            else:
                network_config = config_mapping[network_config_person[model_name]]

            config["network_dim"] = network_config["network_dim"]
            config["network_alpha"] = network_config["network_alpha"]
            config["network_args"] = network_config["network_args"]

        # Save config to file
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, config_path)
        print(f"config is {config}", flush=True)
        print(f"Created config at {config_path}", flush=True)
        return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]
    else:
        if model_type == "sdxl":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-script/{model_type}_train_network.py",
                "--config_file", config_path
            ]
        elif model_type == "flux":
            training_command = [
                "accelerate", "launch",
                "--dynamo_backend", "no",
                "--dynamo_mode", "default",
                "--mixed_precision", "bf16",
                "--num_processes", "1",
                "--num_machines", "1",
                "--num_cpu_threads_per_process", "2",
                f"/app/sd-scripts/{model_type}_train_network.py",
                "--config_file", config_path
            ]
    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 


def get_aitoolkit_config_template_path(model_type: str, train_data_dir: str) -> tuple[str, bool]:
    """Get the appropriate AI-toolkit YAML template path based on model type and training type"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "core", "config")
    
    model_type = model_type.lower()
    
    if model_type == ImageModelType.SDXL.value:
        prompts_path = os.path.join(train_data_dir, "5_lora style")
        if not os.path.exists(prompts_path):
            template_path = os.path.join(config_dir, "base_diffusion_sdxl_person_aitoolkit.yaml")
            return template_path, False
        
        prompts = []
        for file in os.listdir(prompts_path):
            if file.endswith(".txt"):
                try:
                    with open(os.path.join(prompts_path, file), "r") as f:
                        prompt = f.read().strip()
                        prompts.append(prompt)
                except Exception as e:
                    print(f"Warning: Could not read prompt file {file}: {e}", flush=True)
        
        try:
            from trainer.utils.style_detection import detect_styles_in_prompts
            styles = detect_styles_in_prompts(prompts)
            print(f"Detected styles: {styles}", flush=True)
            
            if styles:
                template_path = os.path.join(config_dir, "base_diffusion_sdxl_style_aitoolkit.yaml")
                return template_path, True
        except Exception as e:
            print(f"Warning: Style detection failed: {e}", flush=True)
        
        template_path = os.path.join(config_dir, "base_diffusion_sdxl_person_aitoolkit.yaml")
        return template_path, False
    
    elif model_type == ImageModelType.FLUX.value:
        template_path = os.path.join(config_dir, "base_diffusion_flux_aitoolkit.yaml")
        return template_path, False
    
    elif model_type == ImageModelType.Z_IMAGE.value:
        template_path = os.path.join(config_dir, "base_diffusion_zimage.yaml")
        return template_path, False
    
    elif model_type == ImageModelType.QWEN_IMAGE.value:
        template_path = os.path.join(config_dir, "base_diffusion_qwen_image.yaml")
        return template_path, False
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_network_config_for_sdxl(model_name: str, is_style: bool) -> dict:
    """Get network configuration based on SDXL model and training type"""
    
    network_config_person = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 467,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 467,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 699,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 467,
        "dataautogpt3/TempestV0.1": 456,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 467,
        "fluently/Fluently-XL-Final": 228,
        "mann-e/Mann-E_Dreams": 456,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 228,
        "recoilme/colorfulxl": 228,
        "zenless-lab/sdxl-aam-xl-anime-mix": 456,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
        "zenless-lab/sdxl-anything-xl": 228,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
        "Corcelio/mobius": 228,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
    }

    network_config_style = {k: 235 for k in network_config_person.keys()}
    network_config_style["dataautogpt3/TempestV0.1"] = 228
    config_mapping = {
        228: {
            "network_dim": 32,
            "network_alpha": 32,
            "use_conv": False
        },
        235: {
            "network_dim": 32,
            "network_alpha": 32,
            "use_conv": True,
            "conv_dim": 4,
            "conv_alpha": 4
        },
        456: {
            "network_dim": 64,
            "network_alpha": 64,
            "use_conv": False
        },
        467: {
            "network_dim": 64,
            "network_alpha": 64,
            "use_conv": True,
            "conv_dim": 4,
            "conv_alpha": 4
        },
        699: {
            "network_dim": 96,
            "network_alpha": 96,
            "use_conv": True,
            "conv_dim": 4,
            "conv_alpha": 4
        },
    }

    network_config_dict = network_config_style if is_style else network_config_person
    config_id = network_config_dict.get(model_name, 235)
    
    return config_mapping[config_id]


def create_aitoolkit_config(task_id: str, model_path: str, model_name: str, model_type: str, 
                            expected_repo_name: str, trigger_word: str = None) -> str:
    """Create ai-toolkit configuration for any model type"""
    
    train_data_dir = train_paths.get_image_training_images_dir(task_id)
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    config_template_path, is_style = get_aitoolkit_config_template_path(model_type, train_data_dir)
    print(f"Loading config template from: {config_template_path}", flush=True)
    print(f"Training type: {'Style' if is_style else 'Person'}", flush=True)
    
    try:
        with open(config_template_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading template: {e}", flush=True)
        raise RuntimeError(f"Failed to load config template from {config_template_path}")
    
    process = config["config"]["process"][0]
    process["training_folder"] = output_dir
    process["datasets"][0]["folder_path"] = train_data_dir
    process["model"]["name_or_path"] = model_path
    
    if trigger_word:
        process["trigger_word"] = trigger_word
    
    if model_type == "sdxl":
        network_config = get_network_config_for_sdxl(model_name, is_style)
        print(f"Model: {model_name}", flush=True)
        print(f"Network config: linear={network_config['network_dim']}, alpha={network_config['network_alpha']}, use_conv={network_config['use_conv']}", flush=True)
        
        process["network"]["linear"] = network_config["network_dim"]
        process["network"]["linear_alpha"] = network_config["network_alpha"]
        
        if network_config["use_conv"]:
            process["network"]["conv"] = network_config["conv_dim"]
            process["network"]["conv_alpha"] = network_config["conv_alpha"]
        else:
            process["network"].pop("conv", None)
            process["network"].pop("conv_alpha", None)
        
        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hashlib.sha256(model_name.encode('utf-8')).hexdigest()
            default_config = lrs_config.get("default", {})
            model_config = lrs_config.get("data", {}).get(model_hash, {})
            
            merged_lrs = {**default_config, **model_config}
            
            if model_config:
                print(f"Applying LRS overrides for model hash: {model_hash[:16]}...", flush=True)
                
                if "unet_lr" in merged_lrs:
                    process["train"]["lr"] = merged_lrs["unet_lr"]
                    print(f"  - LR override: {merged_lrs['unet_lr']}", flush=True)
                
                if "text_encoder_lr" in merged_lrs:
                    te_lr = merged_lrs["text_encoder_lr"]
                    if isinstance(te_lr, list):
                        te_lr = te_lr[0]
                    process["train"]["text_encoder_lr"] = te_lr
                    print(f"  - Text Encoder LR override: {te_lr}", flush=True)
                
                if "max_train_steps" in merged_lrs:
                    if "max_train_epochs" in process["train"]:
                        print(f"  - Note: Template uses epochs, keeping epoch-based training", flush=True)
                    else:
                        process["train"]["steps"] = merged_lrs["max_train_steps"]
                        print(f"  - Steps override: {merged_lrs['max_train_steps']}", flush=True)

                if "min_snr_gamma" in merged_lrs:
                    process["train"]["min_snr_gamma"] = merged_lrs["min_snr_gamma"]

                if "train_batch_size" in merged_lrs:
                    process["train"]["batch_size"] = merged_lrs["train_batch_size"]

                if "max_grad_norm" in merged_lrs:
                    process["train"]["max_grad_norm"] = merged_lrs["max_grad_norm"]

                if "max_train_epochs" in merged_lrs:
                    process["train"]["max_train_epochs"] = merged_lrs["max_train_epochs"]

                if "noise_offset" in merged_lrs:
                    process["train"]["noise_offset"] = merged_lrs["noise_offset"]

                if "lr_warmup_steps" in merged_lrs:
                    process["train"]["lr_warmup_steps"] = merged_lrs["lr_warmup_steps"]
                
                if "prior_loss_weight" in merged_lrs:
                    process["train"]["prior_loss_weight"] = merged_lrs["prior_loss_weight"]
                    print(f"  - Prior Loss Weight override: {merged_lrs['prior_loss_weight']}", flush=True)
                
                if "seed" in merged_lrs:
                    process["train"]["seed"] = merged_lrs["seed"]
                    print(f"  - Seed override: {merged_lrs['seed']}", flush=True)
                
                if "optimizer_args" in merged_lrs:
                    optimizer_args_dict = {}
                    for arg in merged_lrs["optimizer_args"]:
                        if "=" in arg:
                            key, value = arg.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if value.lower() == "true":
                                optimizer_args_dict[key] = True
                            elif value.lower() == "false":
                                optimizer_args_dict[key] = False
                            elif value.startswith("(") and value.endswith(")"):
                                import ast
                                try:
                                    optimizer_args_dict[key] = ast.literal_eval(value)
                                except:
                                    optimizer_args_dict[key] = value
                            else:
                                try:
                                    if "e" in value.lower() or "." in value:
                                        optimizer_args_dict[key] = float(value)
                                    else:
                                        optimizer_args_dict[key] = int(value)
                                except ValueError:
                                    optimizer_args_dict[key] = value
                    
                    if "optimizer_args" not in process["train"]:
                        process["train"]["optimizer_args"] = {}
                    process["train"]["optimizer_args"].update(optimizer_args_dict)
                    print(f"  - Optimizer Args override: {optimizer_args_dict}", flush=True)

        process["model"]["is_xl"] = True
        
    elif model_type == "flux":
        print(f"Flux model: {model_name}", flush=True)
        
        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hashlib.sha256(model_name.encode('utf-8')).hexdigest()
            default_config = lrs_config.get("default", {})
            model_config = lrs_config.get("data", {}).get(model_hash, {})
            
            merged_lrs = {**default_config, **model_config}
            
            if model_config:
                print(f"Applying LRS overrides for Flux model", flush=True)
                
                if "unet_lr" in merged_lrs:
                    process["train"]["lr"] = merged_lrs["unet_lr"]
                    print(f"  - LR override: {merged_lrs['unet_lr']}", flush=True)
                
                if "text_encoder_lr" in merged_lrs:
                    te_lr = merged_lrs["text_encoder_lr"]
                    if isinstance(te_lr, list):
                        te_lr = te_lr[0]
                    process["train"]["text_encoder_lr"] = te_lr
                    print(f"  - Text Encoder LR override: {te_lr}", flush=True)
                
                if "max_train_steps" in merged_lrs:
                    if "max_train_epochs" in process["train"]:
                        print(f"  - Note: Template uses epochs, keeping epoch-based training", flush=True)
                    else:
                        process["train"]["steps"] = merged_lrs["max_train_steps"]
                        print(f"  - Steps override: {merged_lrs['max_train_steps']}", flush=True)
                
                if "min_snr_gamma" in merged_lrs:
                    process["train"]["min_snr_gamma"] = merged_lrs["min_snr_gamma"]
                    print(f"  - Min SNR Gamma override: {merged_lrs['min_snr_gamma']}", flush=True)

                if "train_batch_size" in merged_lrs:
                    process["train"]["batch_size"] = merged_lrs["train_batch_size"]
                    print(f"  - Batch Size override: {merged_lrs['train_batch_size']}", flush=True)

                if "max_grad_norm" in merged_lrs:
                    process["train"]["max_grad_norm"] = merged_lrs["max_grad_norm"]
                    print(f"  - Max Grad Norm override: {merged_lrs['max_grad_norm']}", flush=True)

                if "max_train_epochs" in merged_lrs:
                    process["train"]["max_train_epochs"] = merged_lrs["max_train_epochs"]
                    print(f"  - Max Train Epochs override: {merged_lrs['max_train_epochs']}", flush=True)

                if "noise_offset" in merged_lrs:
                    process["train"]["noise_offset"] = merged_lrs["noise_offset"]
                    print(f"  - Noise Offset override: {merged_lrs['noise_offset']}", flush=True)

                if "lr_warmup_steps" in merged_lrs:
                    process["train"]["lr_warmup_steps"] = merged_lrs["lr_warmup_steps"]
                    print(f"  - LR Warmup Steps override: {merged_lrs['lr_warmup_steps']}", flush=True)
                
                if "prior_loss_weight" in merged_lrs:
                    process["train"]["prior_loss_weight"] = merged_lrs["prior_loss_weight"]
                    print(f"  - Prior Loss Weight override: {merged_lrs['prior_loss_weight']}", flush=True)
                
                if "seed" in merged_lrs:
                    process["train"]["seed"] = merged_lrs["seed"]
                    print(f"  - Seed override: {merged_lrs['seed']}", flush=True)
                
                if "optimizer_args" in merged_lrs:
                    # Parse optimizer_args from array format to dict
                    optimizer_args_dict = {}
                    for arg in merged_lrs["optimizer_args"]:
                        if "=" in arg:
                            key, value = arg.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Parse value types
                            if value.lower() == "true":
                                optimizer_args_dict[key] = True
                            elif value.lower() == "false":
                                optimizer_args_dict[key] = False
                            elif value.startswith("(") and value.endswith(")"):
                                # Parse tuple like (0.9, 0.999)
                                import ast
                                try:
                                    optimizer_args_dict[key] = ast.literal_eval(value)
                                except:
                                    optimizer_args_dict[key] = value
                            else:
                                try:
                                    # Try parse as number
                                    if "e" in value.lower() or "." in value:
                                        optimizer_args_dict[key] = float(value)
                                    else:
                                        optimizer_args_dict[key] = int(value)
                                except ValueError:
                                    optimizer_args_dict[key] = value
                    
                    if "optimizer_args" not in process["train"]:
                        process["train"]["optimizer_args"] = {}
                    process["train"]["optimizer_args"].update(optimizer_args_dict)
                    print(f"  - Optimizer Args override: {optimizer_args_dict}", flush=True)
        
        process["model"]["is_flux"] = True
        if "quantize" not in process["model"]:
            process["model"]["quantize"] = True
        
    elif model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]:
        print(f"Using template config for {model_type}", flush=True)
        pass
    
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
    save_config(config, config_path)
    
    print(f"Created ai-toolkit config at {config_path}", flush=True)
    print(f"Config content:\n{yaml.dump(config, default_flow_style=False)}", flush=True)
    
    return config_path


def flatten_aitoolkit_output(output_dir: str, config_name: str = "last"):
    """
    Flatten AI-toolkit output structure by moving files from nested subdirectory
    
    AI-toolkit creates: /output_dir/{config_name}/files
    We need: /output_dir/files
    
    Args:
        output_dir: Base output directory
        config_name: Config name used in training (default: "last")
    """
    import shutil
    
    nested_dir = os.path.join(output_dir, config_name)
    
    if not os.path.exists(nested_dir):
        print(f"No nested directory found: {nested_dir}", flush=True)
        return
    
    print(f"\n{'='*80}", flush=True)
    print("Flattening directory structure...", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Moving files from: {nested_dir}", flush=True)
    print(f"To: {output_dir}", flush=True)
    
    moved_files = 0
    
    try:
        for item in os.listdir(nested_dir):
            src_path = os.path.join(nested_dir, item)
            dest_path = os.path.join(output_dir, item)
            
            if os.path.exists(dest_path):
                print(f"⚠️  Skipped (already exists): {item}", flush=True)
                continue
            
            shutil.move(src_path, dest_path)
            print(f"✅ Moved: {item}", flush=True)
            moved_files += 1
        
        if not os.listdir(nested_dir):
            os.rmdir(nested_dir)
            print(f"✅ Removed empty directory: {config_name}/", flush=True)
        
        print(f"\nMoved {moved_files} files successfully.", flush=True)
        print(f"{'='*80}\n", flush=True)
        
    except Exception as e:
        print(f"❌ Error flattening directory: {e}", flush=True)
        raise


async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "qwen-image", "z-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    # Prepare dataset
    print("Preparing dataset...", flush=True)
    training_images_repeat = (cst.DIFFUSION_SDXL_REPEATS 
                              if args.model_type == ImageModelType.SDXL.value 
                              else cst.DIFFUSION_FLUX_REPEATS)
    
    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=training_images_repeat,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    if args.model_type in (ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value,): 
        config_path = create_aitoolkit_config(
            args.task_id,
            model_path,
            args.model,
            args.model_type,
            args.expected_repo_name,
            args.trigger_word
        )
    
        output_dir = train_paths.get_checkpoints_output_path(args.task_id, args.expected_repo_name)
    
        run_training(model_type=args.model_type,config_path=config_path,output_dir=output_dir)
    else:
        # Create config file
        config_path = create_config(
            args.task_id,
            model_path,
            args.model,
            args.model_type,
            args.expected_repo_name,
            args.trigger_word
        )

        # Run training
        run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
