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
import torch.backends.cudnn as cudnn
import time

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def get_source_model(args, trainset, testset, n_class, mode, encoder=None, epochs=50, verbose=True, sharpness_aware=True):

    print("Start training source model")
    model = Classifier(encoder, MLP(mode=mode, n_class=n_class, hidden=1024)).to(device)

    if sharpness_aware == True:
        optimizer = optim.SGD
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for epoch in range(1, epochs+1):
        train(epoch, trainloader, model, base_optimizer=optimizer, verbose=verbose, sharpness_aware=sharpness_aware)
        if epoch % 5 == 0:
            test(testloader, model, verbose=verbose)

    return model


def run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=10, sharpness_aware=True):

    # get the performance of direct adaptation from the source to target, st involves self-training on target
    direct_acc, st_acc, st_rep_shift, st_sharpnesses, rep_norm = self_train(args, model_copy, [tgt_trainset], epochs=epochs, sharpness_aware=sharpness_aware)
    # get the performance of GST from the source to target, st involves self-training on target
    direct_acc_all, st_acc_all, st_rep_shift_all, st_sharpnesses_all, rep_norm_all = self_train(args, source_model, all_sets, epochs=epochs, sharpness_aware=sharpness_aware)

    # encode the source and target domains
    # e_src_trainset, e_tgt_trainset = get_encoded_dataset(source_model.encoder, src_trainset), get_encoded_dataset(source_model.encoder, tgt_trainset)

    # # encode the intermediate ground-truth domains
    # intersets = all_sets[:-1]
    # encoded_intersets = [e_src_trainset]
    # for i in intersets:
    #     encoded_intersets.append(get_encoded_dataset(source_model.encoder, i))
    # encoded_intersets.append(e_tgt_trainset)

    # generate intermediate domains
    generated_acc = 0
    # if generated_domains > 0:
    #     all_domains = []
    #     for i in range(len(encoded_intersets)-1):
    #         all_domains += generate_domains(generated_domains, encoded_intersets[i], encoded_intersets[i+1])

    #     _, generated_acc = self_train(args, source_model.mlp, all_domains, epochs=epochs, sharpness_aware=sharpness_aware)
    
    return direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc, st_rep_shift, st_rep_shift_all, np.mean(st_sharpnesses), np.mean(st_sharpnesses_all), np.mean(rep_norm), np.mean(rep_norm_all)

def run_portraits_experiment(gt_domains, generated_domains, sharpness_aware=True):
    t = time.time()

    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = make_portraits_data(1000, 1000, 14000, 2000, 1000, 1000)
    tr_x, tr_y = np.concatenate([src_tr_x, src_val_x]), np.concatenate([src_tr_y, src_val_y])
    ts_x, ts_y = np.concatenate([trg_val_x, trg_test_x]), np.concatenate([trg_val_y, trg_test_y])

    encoder = ENCODER().to(device)
    transforms = ToTensor()

    src_trainset = EncodeDataset(tr_x, tr_y.astype(int), transforms)
    tgt_trainset = EncodeDataset(ts_x, ts_y.astype(int), transforms)
    source_model = get_source_model(args, src_trainset, src_trainset, 2, mode="portraits", encoder=encoder, epochs=100, sharpness_aware=sharpness_aware)
    model_copy = copy.deepcopy(source_model)

    def get_domains(n_domains):
        domain_set = []
        n2idx = {0:[], 1:[3], 2:[2,4], 3:[1,3,5], 4:[0,2,4,6], 7:[0,1,2,3,4,5,6]}
        domain_idx = n2idx[n_domains]
        for i in domain_idx:
            start, end = i*2000, (i+1)*2000
            domain_set.append(EncodeDataset(inter_x[start:end], inter_y[start:end].astype(int), transforms))
        return domain_set

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)
    
    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc, st_rep_shift, st_rep_shift_all, st_sharp, st_sharp_all, rep_norm, rep_norm_all = run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=25, sharpness_aware=sharpness_aware)

    elapsed = round(time.time() - t, 2)
    with open(f"logs/portraits_exp_time.txt", "a") as f:
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)},{round(0 if len(st_rep_shift) == 0 else np.mean(st_rep_shift), 2)},{round(np.mean(st_rep_shift_all), 2)},{round(st_sharp, 2)}, {round(st_sharp_all, 2)},{round(rep_norm, 2)}, {round(rep_norm_all, 2)}\n")


def run_mnist_experiment(target, gt_domains, generated_domains, sharpness_aware=True):

    t = time.time()

    src_trainset, tgt_trainset = get_single_rotate(False, 0), get_single_rotate(False, target)

    encoder = ENCODER().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, 10, "mnist", encoder=encoder, epochs=100, sharpness_aware=sharpness_aware)
    model_copy = copy.deepcopy(source_model)

    all_sets = []
    for i in range(1, gt_domains+1):
        all_sets.append(get_single_rotate(False, i*target//(gt_domains+1)))
        print(i*target//(gt_domains+1))
    all_sets.append(tgt_trainset)

    # TODO: Make sure to add rep shift to other datasets
    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc, st_rep_shift, st_rep_shift_all, st_sharp, st_sharp_all, rep_norm, rep_norm_all = run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=25, sharpness_aware=sharpness_aware)

    elapsed = round(time.time() - t, 2)
    # print(elapsed)
    with open(f"logs/mnist_{target}_{gt_domains}_layer.txt", "a") as f:
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)},{round(0 if len(st_rep_shift) == 0 else np.mean(st_rep_shift), 2)},{round(np.mean(st_rep_shift_all), 2)},{round(st_sharp, 2)}, {round(st_sharp_all, 2)},{round(rep_norm, 2)}, {round(rep_norm_all, 2)}\n")


def run_covtype_experiment(gt_domains, generated_domains, sharpness_aware=True):
    data = make_cov_data(40000, 10000, 400000, 50000, 25000, 20000)
    (src_tr_x, src_tr_y, src_val_x, src_val_y, inter_x, inter_y, dir_inter_x, dir_inter_y,
        trg_val_x, trg_val_y, trg_test_x, trg_test_y) = data
    
    src_trainset = EncodeDataset(torch.from_numpy(src_val_x).float(), src_val_y.astype(int))
    tgt_trainset = EncodeDataset(torch.from_numpy(trg_test_x).float(), torch.tensor(trg_test_y.astype(int)))

    encoder = MLP_Encoder().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, 2, mode="covtype", encoder=encoder, epochs=100, sharpness_aware=sharpness_aware)
    model_copy = copy.deepcopy(source_model)

    def get_domains(n_domains):
        domain_set = []
        n2idx = {0:[], 1:[6], 2:[3,7], 3:[2,5,8], 4:[2,4,6,8], 5:[1,3,5,7,9], 10: range(10), 200: range(200)}
        domain_idx = n2idx[n_domains]
        # domain_idx = range(n_domains)
        for i in domain_idx:
            # start, end = i*2000, (i+1)*2000
            # start, end = i*10000, (i+1)*10000
            start, end = i*40000, i*40000 + 2000
            domain_set.append(EncodeDataset(torch.from_numpy(inter_x[start:end]).float(), inter_y[start:end].astype(int)))
        return domain_set
    
    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc, st_rep_shift, st_rep_shift_all, st_sharp, st_sharp_all, rep_norm, rep_norm_all = run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=25, sharpness_aware=sharpness_aware)

    with open(f"logs/covtype_exp_{args.log_file}.txt", "a") as f:
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)},{round(0 if len(st_rep_shift) == 0 else np.mean(st_rep_shift), 2)},{round(np.mean(st_rep_shift_all), 2)},{round(st_sharp, 2)}, {round(st_sharp_all, 2)},{round(rep_norm, 2)}, {round(rep_norm_all, 2)}\n")


def run_color_mnist_experiment(gt_domains, generated_domains, sharpness_aware=True):
    shift = 1
    total_domains = 20

    src_tr_x, src_tr_y, src_val_x, src_val_y, dir_inter_x, dir_inter_y, dir_inter_x, dir_inter_y, trg_val_x, trg_val_y, trg_test_x, trg_test_y = ColorShiftMNIST(shift=shift)
    inter_x, inter_y = transform_inter_data(dir_inter_x, dir_inter_y, 0, shift, interval=len(dir_inter_x)//total_domains, n_domains=total_domains)

    src_x, src_y = np.concatenate([src_tr_x, src_val_x]), np.concatenate([src_tr_y, src_val_y])
    tgt_x, tgt_y = np.concatenate([trg_val_x, trg_test_x]), np.concatenate([trg_val_y, trg_test_y])
    src_trainset, tgt_trainset = EncodeDataset(src_x, src_y.astype(int), ToTensor()), EncodeDataset(trg_val_x, trg_val_y.astype(int), ToTensor())

    encoder = ENCODER().to(device)
    source_model = get_source_model(args, src_trainset, src_trainset, 10, "mnist", encoder=encoder, epochs=100, sharpness_aware=sharpness_aware)
    model_copy = copy.deepcopy(source_model)

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

    all_sets = get_domains(gt_domains)
    all_sets.append(tgt_trainset)

    direct_acc, st_acc, direct_acc_all, st_acc_all, generated_acc, st_rep_shift, st_rep_shift_all, st_sharp, st_sharp_all, rep_norm, rep_norm_all = run_goat(model_copy, source_model, src_trainset, tgt_trainset, all_sets, generated_domains, epochs=25, sharpness_aware=sharpness_aware)
        
    with open(f"logs/color{args.log_file}.txt", "a") as f:
        f.write(f"seed{args.seed}with{gt_domains}gt{generated_domains}generated,{round(direct_acc, 2)},{round(st_acc, 2)},{round(direct_acc_all, 2)},{round(st_acc_all, 2)},{round(generated_acc, 2)},{round(0 if len(st_rep_shift) == 0 else np.mean(st_rep_shift), 2)},{round(np.mean(st_rep_shift_all), 2)},{round(st_sharp, 2)}, {round(st_sharp_all, 2)},{round(rep_norm, 2)}, {round(rep_norm_all, 2)}\n")


def main(args):
    
    print(args)
    
    if args.dataset == "mnist":
        if args.mnist_mode == "normal":
            run_mnist_experiment(args.rotation_angle, args.gt_domains, args.generated_domains, args.sam)
    else:
        eval(f"run_{args.dataset}_experiment({args.gt_domains}, {args.generated_domains}, {args.sam})")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="GOAT experiments")
    parser.add_argument("--dataset", choices=["mnist", "portraits", "covtype", "color_mnist"])
    parser.add_argument("--gt-domains", default=0, type=int)
    parser.add_argument("--generated-domains", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--mnist-mode", default="normal", choices=["normal"])
    parser.add_argument("--rotation-angle", default=45, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--log-file", default="")
    parser.add_argument("--sam", default=True)
    args = parser.parse_args()

    main(args)