import torch
from args.args import args
from Con_Rep import concept_main
from train_code import train_main
from train_exp import train_main as train_exp_main



if __name__ == '__main__':
    torch.cuda.empty_cache()
    # train_main(args)
    # concept_main()
    train_exp_main(args)