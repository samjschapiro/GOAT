import torch
from model import *
import torch.optim as optim
from train_model import *
from util import *
from ot_util import ot_ablation
from da_algo import *
from ot_util import generate_domains
from dataset import *
import copy
import argparse
import random
import os
import torch.backends.cudnn as cudnn
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_source_model(args, trainset, testset, n_class, mode, encoder=None, epochs=50, verbose=True, opt_name='sgd'):

    print("Start training source model")
    model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # TODO: Add metrics here
    for epoch in range(1, epochs+1):
        train(epoch, trainloader, model, base_opt=args.base_opt, opt_name=opt_name, verbose=verbose)
        if epoch % 5 == 0:
            test(testloader, model, verbose=verbose)
    return model


def run_goat(source_model, all_sets, opt_name, epochs=10):
    st_acc_all, st_rep_shift_all, st_sharpnesses_all, rep_norm_all = self_train(args, source_model, all_sets, epochs=epochs, opt_name=opt_name, base_opt=args.base_opt)
    return st_acc_all, st_rep_shift_all, np.mean(st_sharpnesses_all), np.mean(rep_norm_all)


def run_portraits_experiment(intermediate_domains, opt_name):

    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, _, _, trg_val_x, trg_val_y, trg_test_x, trg_test_y) = make_portraits_data(1000, 1000, 14000, 2000, 1000, 1000)
    tr_x, tr_y = np.concatenate([src_tr_x, src_val_x]), np.concatenate([src_tr_y, src_val_y])
    ts_x, ts_y = np.concatenate([trg_val_x, trg_test_x]), np.concatenate([trg_val_y, trg_test_y])

    encoder = ENCODER().to(device)
    transforms = ToTensor()

    src_trainset = EncodeDataset(tr_x, tr_y.astype(int), transforms)
    tgt_trainset = EncodeDataset(ts_x, ts_y.astype(int), transforms)
    source_model = get_source_model(args, src_trainset, src_trainset, 2, mode="portraits", encoder=encoder, epochs=args.source_epochs, opt_name=opt_name)

    def get_domains(n_domains):
        domain_set = []
        n2idx = {0:[], 1:[3], 2:[2,4], 3:[1,3,5], 4:[0,2,4,6], 7:[0,1,2,3,4,5,6]}
        domain_idx = n2idx[n_domains]
        for i in domain_idx:
            start, end = i*2000, (i+1)*2000
            domain_set.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), transforms))
        return domain_set

    all_sets = get_domains(intermediate_domains)
    all_sets.append(tgt_trainset)
    
    st_acc_all, st_rep_shift_all, st_sharp_all, rep_norm_all = run_goat(source_model, all_sets, epochs=args.intermediate_epochs, opt_name=opt_name)

    with open(f"logs/portraits_opt:{args.optname}_num_int_dom:{args.intermediate_domains}.txt", "a") as f:
        f.write(f"{args.seed},{intermediate_domains},{round(st_acc_all, 2)},{round(np.mean(st_rep_shift_all), 2)}, {round(st_sharp_all, 2)}, {round(rep_norm_all, 2)}\n")


def run_mnist_experiment(target, intermediate_domains, opt_name):

    src_trainset, tgt_trainset = get_single_rotate(False, 0), get_single_rotate(False, target)

    encoder = ENCODER().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, n_class=10, mode="mnist", encoder=encoder, epochs=args.source_epochs, opt_name=opt_name)

    all_sets = []
    for i in range(1, intermediate_domains+1):
        all_sets.append(get_single_rotate(False, i*target//(intermediate_domains+1)))
        print(i*target//(intermediate_domains+1))
    all_sets.append(tgt_trainset)

    st_acc_all, st_rep_shift_all, st_sharp_all, rep_norm_all = run_goat(source_model, all_sets, epochs=args.intermediate_epochs, opt_name=opt_name)

    with open(f"logs/mnist_{target}_opt:{args.optname}_num_int_dom:{args.intermediate_domains}.txt", "a") as f:
        f.write(f"{args.seed},{intermediate_domains},{round(st_acc_all, 2)},{round(np.mean(st_rep_shift_all), 2)}, {round(st_sharp_all, 2)}, {round(rep_norm_all, 2)}\n")


def run_covtype_experiment(intermediate_domains, opt_name):
    data = make_cov_data(40000, 10000, 400000, 50000, 25000, 20000)
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = data
    
    src_trainset = EncodeDataset(torch.from_numpy(src_val_x).float(), src_val_y.astype(int))
    tgt_trainset = EncodeDataset(torch.from_numpy(trg_test_x).float(), torch.tensor(trg_test_y.astype(int)))

    encoder = MLP_Encoder().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, 2, mode="covtype", encoder=encoder, epochs=args.source_epochs, opt_name=opt_name)

    def get_domains(n_domains):
        domain_set = []
        n2idx = {0:[], 1:[6], 2:[3,7], 3:[2,5,8], 4:[2,4,6,8], 5:[1,3,5,7,9], 10: range(10), 200: range(200)}
        domain_idx = n2idx[n_domains]
        for i in domain_idx:
            start, end = i*40000, i*40000 + 2000
            domain_set.append(EncodeDataset(torch.from_numpy(inter_x[start:end]).float(), inter_y[start:end].astype(int)))
        return domain_set
    
    all_sets = get_domains(intermediate_domains)
    all_sets.append(tgt_trainset)

    st_acc_all, st_rep_shift_all, st_sharp_all, rep_norm_all = run_goat(source_model, all_sets, epochs=args.intermediate_epochs, opt_name=opt_name)

    with open(f"logs/covtype_opt:{args.optname}_num_int_dom:{args.intermediate_domains}.txt", "a") as f:
        f.write(f"{args.seed},{intermediate_domains},{round(st_acc_all, 2)},{round(np.mean(st_rep_shift_all), 2)}, {round(st_sharp_all, 2)}, {round(rep_norm_all, 2)}\n")


def run_color_mnist_experiment(intermediate_domains, opt_name):
    shift = 1
    total_domains = 20

    src_tr_x, src_tr_y, src_val_x, src_val_y, dir_inter_x, dir_inter_y, dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y = ColorShiftMNIST(shift=shift)
    inter_x, inter_y = transform_inter_data(dir_inter_x, dir_inter_y, 0, shift, interval=len(dir_inter_x)//total_domains, n_domains=total_domains)

    src_x, src_y = np.concatenate([src_tr_x, src_val_x]), np.concatenate([src_tr_y, src_val_y])
    tgt_x, tgt_y = np.concatenate([trg_val_x, trg_test_x]), np.concatenate([trg_val_y, trg_test_y])
    src_trainset, tgt_trainset = EncodeDataset(src_x, src_y.astype(int), ToTensor()), EncodeDataset(trg_val_x, trg_val_y.astype(int), ToTensor())

    encoder = ENCODER().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, 10, "mnist", encoder=encoder, epochs=args.source_epochs, opt_name=opt_name)

    def get_domains(n_domains):
        domain_set = []
        domain_idx = []
        if n_domains == total_domains:
            domain_idx = range(n_domains)
        else:
            for i in range(1, n_domains+1):
                domain_idx.append(total_domains // (n_domains+1) * i)
        interval = 42000 // total_domains
        for i in domain_idx:
            start, end = i*interval, (i+1)*interval
            domain_set.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), ToTensor()))
        return domain_set

    all_sets = get_domains(intermediate_domains)
    all_sets.append(tgt_trainset)

    st_acc_all, st_rep_shift_all, st_sharp_all, rep_norm_all = run_goat(source_model, all_sets, epochs=args.intermediate_epochs, opt_name=opt_name)
        
    with open(f"logs/color_opt:{args.optname}_num_int_dom:{args.intermediate_domains}.txt", "a") as f:
        if os.stat(f"logs/color_opt:{args.optname}_num_int_dom:{args.intermediate_domains}.txt").st_size == 0:
            f.write("seed, intermediate_domains, accuracy, rep_shift, sharpness, rep_norm\n")
        f.write(f"{args.seed},{intermediate_domains},{round(st_acc_all, 2)},{round(np.mean(st_rep_shift_all), 2)}, {round(st_sharp_all, 2)}, {round(rep_norm_all, 2)}\n")

def main(args):
    print(args)
    for i in range(args.number_indep_runs):
        if args.dataset == "mnist":
            run_mnist_experiment(args.rotation_angle, args.intermediate_domains, args.optname)
        else:
            eval(f"run_{args.dataset}_experiment({args.intermediate_domains}, '{args.optname}')")
        args.seed +=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAM-GDA-experiments")
    parser.add_argument("--dataset", choices=["mnist", "portraits", "covtype", "color_mnist"])
    parser.add_argument("--optname", choices=["sgd", "adam", "sam", "ssam"], default="sam")
    parser.add_argument("--base-opt", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--intermediate-domains", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--number-indep-runs", default=1, type=int)
    parser.add_argument("--rotation-angle", default=45, type=int)
    parser.add_argument("--source-epochs", default=100, type=int)
    parser.add_argument("--intermediate-epochs", default=25, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=4, type=int)
    args = parser.parse_args()

    main(args)