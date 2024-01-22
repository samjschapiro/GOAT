import torch
import torch.nn as nn
import torch.nn.functional as F
from sam import SAM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def calculate_modal_val_accuracy(model, valloader):
    model.eval()
    correct = 0.
    total = 0.

    with torch.no_grad():
        for x in valloader:
            if len(x) == 3:
                images, labels, weight = x
            else:
                images, labels = x

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    return 100 * correct / total


def train(epoch, train_loader, model, base_optimizer, lr_scheduler=None, vae=False, verbose=True, sharpness_aware=True):
    # def closure():
    #     if vae:
    #         recon_batch, mu, log_var = model(data)
    #         loss = loss_function(recon_batch, data, mu, log_var)
    #     else:
    #         output = model(data)
    #         if len(x) == 2:
    #             loss = F.cross_entropy(output.clone(), labels)
    #         elif len(x) == 3:
    #             criterion = nn.CrossEntropyLoss(reduction='none')
    #             loss = criterion(output.clone(), labels)
    #             loss = (loss * weight).mean()
    #     loss.backward()
    #     return loss
    
    def enable_bn(model):
        if isinstance(model, nn.BatchNorm1d):
            model.backup_momentum = model.momentum
            model.momentum = 0
        
    def disable_bn(model):
        if isinstance(model, nn.BatchNorm1d):
            model.momentum = model.backup_momentum

    if sharpness_aware == True:
        optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3)

    model.train()
    train_loss = 0
    final_sharpness = 0
    for _, x in enumerate(train_loader):
        if len(x) == 2:
            data, labels = x
        elif len(x) == 3:
            data, labels, weight = x
            weight = weight.to(device)

        data = data.to(device)
        labels = labels.to(device)


        # if vae:
        #     recon_batch, mu, log_var = model(data)
        #     loss = loss_function(recon_batch, data, mu, log_var)
        # else:
        #     output = model(data)
        #     if len(x) == 2:
        #         loss = F.cross_entropy(output.clone(), labels)
        #     elif len(x) == 3:
        #         criterion = nn.CrossEntropyLoss(reduction='none')
        #         loss = criterion(output.clone(), labels)
        #         loss = (loss * weight).mean()

        if sharpness_aware == True:
            # loss.backward()
            # optimizer.step(closure)
            # optimizer.zero_grad()
            enable_bn(model)
            if vae:
                recon_batch, mu, log_var = model(data)
                loss = loss_function(recon_batch, data, mu, log_var)
            else:
                output = model(data)
                if len(x) == 2:
                    loss = F.cross_entropy(output, labels).backward()
                elif len(x) == 3:
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    loss = criterion(output, labels)
                    loss = (loss * weight).mean().backward()
            
            final_sharpness = optimizer.first_step(zero_grad=True)
                
            disable_bn(model)
            if vae:
                loss_function(model(data)[0], data, model(data)[1], model(data)[2])
            else:
                if len(x) == 2:
                    F.cross_entropy(model(data), labels).backward()
                elif len(x) == 3:
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    (criterion(model(data), labels)  * weight).mean().backward()
            optimizer.second_step(zero_grad=True)

        else:
            base_optimizer.zero_grad()
            if vae:
                recon_batch, mu, log_var = model(data)
                loss = loss_function(recon_batch, data, mu, log_var)
            else:
                output = model(data)
                if len(x) == 2:
                    loss = F.cross_entropy(output, labels)
                elif len(x) == 3:
                    criterion = nn.CrossEntropyLoss(reduction='none')
                    loss = criterion(output, labels)
                    loss = (loss * weight).mean()
            loss.backward()
            train_loss += loss.item()
            base_optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    if verbose:
        print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / len(train_loader.dataset)))

    return final_sharpness

def test(val_loader, model, vae=False, verbose=True):
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

            if vae:
                recon, mu, log_var = model(data)
                test_loss += loss_function(recon, data, mu, log_var).item()
            else:
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
        if not vae:
            print('====> Test Accuracy %.4f' % (val_accuracy))

    return val_accuracy


