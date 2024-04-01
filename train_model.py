import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hessian_eigenthings import compute_hessian_eigenthings
from sam import SAM
from ssam import SSAM
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_modal_val_accuracy(model, valloader):
    model.eval()
    correct = 0.
    total = 0.

    with torch.no_grad():
        for x in valloader:
            if len(x) == 3:
                images, labels, _ = x
            else:
                images, labels = x
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    return 100 * correct / total


def train(epoch, train_loader, model, base_opt, opt_name, lr_scheduler=None, verbose=True):
    def enable_bn(model):
        if isinstance(model, nn.BatchNorm1d):
            model.backup_momentum = model.momentum
            model.momentum = 0
        
    def disable_bn(model):
        if isinstance(model, nn.BatchNorm1d):
            model.momentum = model.backup_momentum

    if base_opt == 'sgd':
        base_optimizer = optim.SGD
    elif base_opt == 'adam':
        base_optimizer = optim.Adam(model.parameters())

    if opt_name == 'sam':
        optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3, weight_decay=1e-4, momentum=0.9, adaptive=bool(opt_name == 'asam'))
    elif opt_name == 'ssam':
        optimizer = SSAM(model.parameters(), base_optimizer, lr=1e-3, weight_decay=1e-4, momentum=0.9, adaptive=bool(opt_name == 'asam'))


    model.train()
    train_loss = 0
    ssam_loss = 0
    sam_loss = 0
    for _, x in enumerate(train_loader):
        if len(x) == 2:
            data, labels = x
        elif len(x) == 3:
            data, labels, weight = x
            weight = weight.to(device)

        data = data.to(device)
        labels = labels.to(device)
        if opt_name == 'ssam':
            inputs_prep, inputs_2, inputs_3 = copy.deepcopy(data), copy.deepcopy(data), copy.deepcopy(data)
            targets_2, targets_3 = copy.deepcopy(labels), copy.deepcopy(labels)
            copy_of_net = copy.deepcopy(model)
            copy_of_optimizer = SAM(copy_of_net.parameters(), optim.SGD, rho=0.05, lr=1e-3, momentum=0.9, weight_decay=5e-4)
            outputs_1 = copy_of_net(inputs_2)
            outputs_2 = copy_of_net(inputs_3)

        if opt_name == 'sam' or opt_name == 'ssam' or opt_name == 'asam':
            enable_bn(model)
            if opt_name == 'ssam':
                if len(x) == 2:
                    loss_f = torch.mean(model(inputs_prep.cuda()))    
                elif len(x) == 3:
                    loss_f = torch.mean(weight*model(inputs_prep.cuda())) 
                loss_f.backward()
                optimizer.prep(zero_grad=True)
          
               
            output = model(data)
            if len(x) == 2:
                loss = F.cross_entropy(output, labels)
                loss.backward()
            elif len(x) == 3:
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss = criterion(output, labels)
                (loss * weight).mean().backward()

            if opt_name == 'ssam':
                # Stuff for logging SAM loss
                if len(x) == 2:
                    loss_1 = F.cross_entropy(outputs_1, targets_2)
                    loss_1.backward()
                elif len(x) == 3:
                    criterion1 = nn.CrossEntropyLoss(reduction='none')
                    loss_1 = criterion1(outputs_1, targets_2)
                    (loss_1*weight).mean().backward()
                copy_of_optimizer.first_step(zero_grad=True)
                if len(x) == 2:
                    sam_loss += F.cross_entropy(outputs_2, targets_3).item()
                elif len(x) == 3:
                    criterion1 = nn.CrossEntropyLoss(reduction='none')
                    loss_2 = criterion1(outputs_2, targets_3)
                    sam_loss += (loss_2*weight).mean().item()
                # SSAM stuff
                optimizer.first_step(zero_grad=True, n_iter=5)
            else:
                optimizer.first_step(zero_grad=True)
            
            disable_bn(model)
            if len(x) == 2:
                scnd_loss = F.cross_entropy(model(data), labels)
            elif len(x) == 3:
                criterion = nn.CrossEntropyLoss(reduction='none')
                scnd_loss = (criterion(model(data), labels) * weight).mean()
            ssam_loss += scnd_loss.item()
            scnd_loss.backward()
            optimizer.second_step(zero_grad=True)
            solution_loss = loss
        else:
            base_optimizer.zero_grad()

            output = model(data)
            if len(x) == 2:
                loss = F.cross_entropy(output, labels)
            elif len(x) == 3:
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss = criterion(output, labels)
                loss = (loss * weight).mean()
            solution_loss = loss
            loss.backward()
            train_loss += loss.item()
            base_optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    # SHARPNESS
    if opt_name in ['ssam', 'asam', 'sam']:
        final_lr = optimizer.param_groups[0]['lr']
        enable_bn(model)
        model_copy = copy.deepcopy(model).to(device)
        optimizer = SAM(model_copy.parameters(), base_optimizer, lr=final_lr)
        output = model_copy(data)
        if len(x) == 2:
            loss = F.cross_entropy(output, labels).backward()
        elif len(x) == 3:
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(output, labels)
            (loss * weight).mean().backward()
        optimizer.first_step(zero_grad=True)
        output = model_copy(data)
        if len(x) == 2:
            perturbed_loss = F.cross_entropy(output, labels)
            perturbed_loss.backward()
        elif len(x) == 3:
            criterion = nn.CrossEntropyLoss(reduction='none')
            perturbed_loss = criterion(output, labels)
            (perturbed_loss * weight).mean().backward()
    else:
        final_lr = base_optimizer.param_groups[0]['lr']
        model_copy = copy.deepcopy(model).to(device)
        base_optimizer_ = optim.SGD
        optimizer = SAM(model_copy.parameters(), base_optimizer_, lr=final_lr)
        output = model_copy(data)
        if len(x) == 2:
            loss = F.cross_entropy(output, labels).backward()
        elif len(x) == 3:
            criterion = nn.CrossEntropyLoss(reduction='none')
            loss = criterion(output, labels)
            (loss * weight).mean().backward()
        optimizer.first_step(zero_grad=True)
        output = model_copy(data)
        if len(x) == 2:
            perturbed_loss = F.cross_entropy(output, labels)
            perturbed_loss.backward()
        elif len(x) == 3:
            criterion = nn.CrossEntropyLoss(reduction='none')
            perturbed_loss = criterion(output, labels)
            (perturbed_loss * weight).mean().backward()

    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))

    if opt_name == 'ssam':
        return perturbed_loss - solution_loss, ssam_loss, sam_loss
    else:
        return perturbed_loss - solution_loss

def test(val_loader, model, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0.
    total = 0.

    with torch.no_grad():
        for x in val_loader:
            if len(x) == 2:
                data, labels = x
            elif len(x) == 3:
                data, labels, weight = x
                weight = weight.to(device)
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            if len(x) == 2:
                criterion = nn.CrossEntropyLoss()
                test_loss += criterion(output, labels).item()
            elif len(x) == 3:
                criterion = nn.CrossEntropyLoss(reduction='none')
                loss = criterion(output, labels)
                test_loss += (loss * weight).mean().item()

            predicted = output.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    test_loss /= len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    val_accuracy = val_accuracy.item()
    if verbose:
        print('====> Test loss: {:.8f}'.format(test_loss))
        print('====> Test Accuracy %.4f' % (val_accuracy))

    return val_accuracy


