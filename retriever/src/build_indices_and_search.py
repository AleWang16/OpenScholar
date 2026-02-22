import os

import logging
from omegaconf.omegaconf import OmegaConf, open_dict

from src.hydra_runner import hydra_runner
from src.index import build_index
from src.search import search_topk, post_hoc_merge_topk_multi_domain

@hydra_runner(config_path="../ric/conf", config_name="default")
def main(cfg)-> None:
    print("Start buildin index...")
    build_index(cfg)
    print("Done building index.  Searching top k...")
    search_topk(cfg)
    print("Done searching top k.")
    
if __name__ == "__main__": 
    main()