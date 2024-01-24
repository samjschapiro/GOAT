import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from train_model import *
from util import *
from dataset import *
from ot_util import *
from model import *
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_labels(dataloader, model, confidence_q=0.1):
    logits = []
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            if len(x) == 2:
                data, labels = x
            elif len(x) == 3:
                data, labels, weight = x
                weight = weight.to(device)
            data = data.to(device)
            labels = labels.to(device)
            logits.append(model(data))
    

    logits = torch.cat(logits)
    confidence = torch.max(logits, dim=1)[0] - torch.min(logits, dim=1)[0]
    alpha = torch.quantile(confidence, confidence_q)
    indices = torch.where(confidence >= alpha)[0].to("cpu")
    labels = torch.argmax(logits, axis=1) #[indices]
    return labels.cpu().detach().type(torch.int64), list(indices.detach().numpy())

def self_train(args, source_model, datasets, epochs=10, sharpness_aware=True, supervised=False):
    steps = len(datasets)
    teacher = source_model
    targetset = datasets[-1]
        
    targetloader = DataLoader(targetset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("------------Direct adapt performance----------")
    direct_acc = test(targetloader, teacher)

    # TODO: Need to store representation shift here
    representation_weights = []
    representation_biases = []
    sharpnesses = []

    # start self-training on intermediate domains
    for i in range(steps):
        print(f"--------Training on the {i}th domain--------")
        trainset = datasets[i]
        ogloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                
        test(targetloader, teacher)
        train_labs, train_idx = get_labels(ogloader, teacher)

        if torch.is_tensor(trainset.data):
            data = trainset.data.cpu().detach().numpy()
        else:
            data = trainset.data
        trainset  = EncodeDataset(data, train_labs, trainset.transform)
        if supervised == False:
            # filter out the least 10% confident data
            filter_trainset = Subset(trainset, train_idx)
            print("Trainset size: " + str(len(filter_trainset)))
            trainloader = DataLoader(filter_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        else:
            trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        # initialize and train student model
        student = copy.deepcopy(teacher)
        if sharpness_aware == True:
            optimizer = optim.SGD 
        else:
            optimizer = optim.Adam(student.parameters(), lr=args.lr, weight_decay=1e-4)

        final_sharpness = 0
        for i in range(1, epochs+1):
            final_sharpness = train(i, trainloader, student, base_optimizer=optimizer, sharpness_aware=sharpness_aware) 
            if i % 5 == 0:
                test(targetloader, student)

        sharpnesses.append(final_sharpness.cpu().detach().numpy())

        print("------------Performance on the current domain----------")

        test(trainloader, student)

        # test on the target domain
        print("------------Performance on the target domain----------")
        st_acc = test(targetloader, student)

        if i == 0:
            rep_weight = None
            rep_bias = None
            for name, param in teacher.named_parameters():
                if name == 'mlp.mlp.7.weight':
                    rep_weight = param 
                if name == 'mlp.mlp.7.bias':
                    rep_bias = param
            representation_weights.append(rep_weight)
            representation_biases.append(rep_bias)

        teacher = copy.deepcopy(student)
        rep_weight = None
        rep_bias = None
        for name, param in teacher.named_parameters():
            if name == 'mlp.mlp.7.weight':
                rep_weight = torch.flatten(param).cpu().detach().numpy()
            if name == 'mlp.mlp.7.bias':
                rep_bias = torch.flatten(param).cpu().detach().numpy()
        representation_weights.append(rep_weight)
        representation_biases.append(rep_bias)
    
    representation_shifts = []
    representation_norms = []
    representation_norms.append((np.linalg.norm(representation_weights[0], 2) + np.linalg.norm(representation_biases[0]))**2)
    for idx in range(len(representation_weights)-1):
        weight_diff = np.linalg.norm(representation_weights[idx] - representation_weights[idx+1], 2)
        bias_diff = np.linalg.norm(representation_biases[idx] - representation_biases[idx+1], 2)
        representation_shifts.append(weight_diff + bias_diff)
        representation_norms.append((np.linalg.norm(representation_weights[idx+1], 2) + np.linalg.norm(representation_biases[idx+1]))**2)

    return direct_acc, st_acc, representation_shifts, sharpnesses, np.mean(representation_norms)

