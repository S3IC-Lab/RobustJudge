import argparse
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.data.registry import DatasetRegistry
from model.model_class import load_model_instance
from modules.eval import EvaluatorRegistry, evaluator_list
from modules.attack import AttackExecutor
from modules.defense import DefenseExecutor
from judge.pipeline import Pipe
# from modules.attack.tg_opt import TgAttacker
from modules.data import lan_flores_dict
def pick_evaluator(target_instance, args):
    if args.task in evaluator_list:
        return EvaluatorRegistry.get_evaluator(evaluator_list[args.task], target_instance=target_instance, args=args)
    else:
        raise ValueError(f"There is no suitable benchmark_evaluator for {args.task} task.")

def main(args):

    dataset_specific_params = {"num_objects": args.num_items}

    # dataset_specific_params = {"num_objects": args.dataset_objects}
    source_lan_code = lan_flores_dict(args.source)
    target_lan_code = lan_flores_dict(args.target)
    if args.dataset == "flores200":
        print(f"source_lan is: {source_lan_code}, tartget_lan is: {target_lan_code}")
        dataset_specific_params.update({
            "source_lang": source_lan_code,
            "target_lang": target_lan_code
        })
    
    dataset = DatasetRegistry.get_dataset(name=args.dataset,
                                          data_dir=args.data_dir,
                                          **dataset_specific_params)
    
    dataset.load_data()
    dataset.process_data()
    
    #=================== LOAD MODEL ===================#
    target_instance = load_model_instance(
        args,
        args.target_model,
        args.target_model_id,
        args.target_model_temperature,
        args.max_new_token
        )
    attack_instance = load_model_instance(
        args,
        args.attack_model,
        args.attack_model_id,
        args.attack_model_temperature,
        args.max_new_token
        )
    
    bench_evaluator = pick_evaluator(target_instance, args)
    
    #=================== PIPELINE ===================#
    attacker = AttackExecutor(bench_evaluator, attack_instance, target_instance, args).create()
    defender = DefenseExecutor(attack_instance, target_instance, args).create()
    from loguru import logger
    if defender.check_pre_result():
        defender.run()
        logger.info(f"Defense result has already existed in {defender.file_path}")
        return
    pipe = Pipe(attacker, defender, dataset, args)
    pipe.run()
    
    defender.run()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-items",
        type=int,
        default=2,
        help="The number of items to process.",
    )
    # =================== MODEL CONFIG ===================
    parser.add_argument(
        "--target-model",
        type=str,
        default="/home/model/Meta-Llama-3-8B-Instruct", # vicuna-7b-v1.5 # Qwen-7B-Chat # Meta-Llama-3-8B-Instruct
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--target-model-id", type=str, default="Meta-Llama-3-8B-Instruct", help="A custom name for the model."
    )
    parser.add_argument(
        "--target-model-temperature",type=float, default=0.001
    )
    parser.add_argument(
        "--attack-model",type=str,default="/home/model/openchat-3.5-0106",help="attack model path"
        # /home/model/openchat-3.5-0106
        # /home/model/Meta-Llama-3-8B-Instruct
    )
    parser.add_argument(
        "--attack-model-id", type=str, default="openchat-3.5-0106"
        # Meta-Llama-3-8B-Instruct
        # openchat-3.5-0106
    )
    parser.add_argument(
        "--attack-model-temperature",type=float,default=0.3
    )
    
    # =================== MODEL CONFIG ===================
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. ",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    # =================== DATA CONFIG ===================
    parser.add_argument(
        "--cache",type=str,default="en2zh.json"
    )
    parser.add_argument(
        "--task", type=str,default="translation",help="The task to perform."
    )
    parser.add_argument(
        "--judge",type=str,default="score",help="The judge type to use."
    )
    parser.add_argument(
        "--judge-prompt",type=str,default="normal",help="The judge prompt to use."
    )
    parser.add_argument(
        "--output-file",type=str,default="result/en2zh1",help="The output file name."
    )
    parser.add_argument(
        "--attack",type=str,default="naive_po",
        help="The attack method to use."
    )
    parser.add_argument(
        "--attack-custom-injections",type=str,default="custom injections",
        help="The custom injections to attack."
    )
    
    parser.add_argument(
        "--defense",type=str,default="none",help="The defense method to use."
    )
    
    parser.add_argument(
        "--dataset",type=str,default="flores200",help="Name of the dataset to process."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/dataset/flores200/dev",
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--dataset-objects",type = int, default=10, help = "numbers of dataset"
    )

    # =================== Translation task config ===================
    parser.add_argument(
        "--source", type=str, default="Chinese"
    )
    parser.add_argument(
        "--target", type=str, default="English"
    )
    args = parser.parse_args()
      
    main(args)

