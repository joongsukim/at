import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from tqdm import tqdm
import numpy as np
import copy
from utils import Logger, save_checkpoint
import torchattacks 
from sklearn.metrics import accuracy_score 

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--arch', type=str, default="resnet18", help="decide which network to use, choose from smallcnn, resnet18, WRN")
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', default=0.01, type=float)

parser.add_argument('--loss_fn', type=str, default="cent", help="loss function")
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')

parser.add_argument('--resume',type=bool, default=False, help='whether to resume training')
parser.add_argument('--out-dir',type=str, default='./logs',help='dir of output')

args = parser.parse_args()

# Training settings
args.out_dir = os.path.join(args.out_dir)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

args.num_classes = 10
weight_decay = 3.5e-3 if args.arch == 'resnet18' else 7e-4
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


class EMA(object):
    def __init__(self, model, alpha=0.999, buffer_ema=True):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha
        self.buffer_ema = buffer_ema
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self, model):
        #decay = min(self.alpha, (self.step + 1) / (self.step + 10))
        decay = self.alpha 
        state = model.state_dict()
        for name in self.param_keys:
            self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
        for name in self.buffer_keys:
            if self.buffer_ema:
                self.shadow[name].copy_(decay * self.shadow[name] + (1 - decay) * state[name])
            else:
                self.shadow[name].copy_(state[name])
        self.step += 1

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

if args.arch == 'resnet18':
    adjust_learning_rate = lambda epoch: np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr, args.lr, args.lr/10, args.lr/100])[0]
elif args.arch == 'WRN':
    args.lr = 0.1
    adjust_learning_rate = lambda epoch: np.interp([epoch], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr, args.lr, args.lr/10, args.lr/20])[0]


def train(epoch, model, teacher_model, optimizer, device, descrip_str):
    teacher_model.model.eval()

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)

        pgd_atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
        x_adv = pgd_atk(inputs,target)
        x_adv = x_adv.to(device)
        model.train()
        lr = adjust_learning_rate(epoch)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()
        
        nat_logit = teacher_model.model(inputs)

        logit = model(x_adv) 
        loss = nn.CrossEntropyLoss()(logit, target)
        loss.backward()
        optimizer.step()

        teacher_model.update_params(model)
        teacher_model.apply_shadow()

        # losses.update(loss.item())
        
def test(model, teacher_model, device):
    model.eval()
    teacher_model.model.eval()

    total_loss = 0.0
    total_num = 0.0
    y_true = []
    y_pred = []
    y_pred_adv = []
    y_logits = []
    y_pred_ema = []
    y_pred_adv_ema = []
    y_logits_ema = []
    for batch_idx, (inputs, target) in enumerate(test_loader):
        pgd_atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20)
        pgd_atk_ema = torchattacks.PGD(teacher_model.model, eps=8/255, alpha=2/255, steps=20)
        inputs, target = inputs.to(device), target.to(device)

        x_adv = pgd_atk(inputs,target)
        x_adv = x_adv.to(device)
        x_adv_ema = pgd_atk(inputs,target)
        x_adv_ema = x_adv_ema.to(device)

        num_batch = target.shape[0]
        total_num += num_batch

        logits = model(inputs)
        logits_adv = model(x_adv)

        logits_ema = teacher_model.model(inputs)
        logits_adv_ema = teacher_model.model(x_adv_ema)


        loss = loss = nn.CrossEntropyLoss()(logits, target)
        loss_ema = nn.CrossEntropyLoss()(logits_ema, target)

        y_true.extend(target.cpu().tolist())
        y_pred_adv.extend(torch.max(logits_adv, dim=-1)[1].cpu().tolist())
        y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
        y_logits.append(logits.cpu().detach().numpy())

        y_pred_adv_ema.extend(torch.max(logits_adv_ema, dim=-1)[1].cpu().tolist())
        y_pred_ema.extend(torch.max(logits_ema, dim=-1)[1].cpu().tolist())
        y_logits_ema.append(logits_ema.cpu().detach().numpy())
        total_loss += loss.item() * num_batch
        total_loss += loss_ema.item() * num_batch 

        top1 = accuracy_score(y_true, y_pred) * 100
        top1_adv = accuracy_score(y_true, y_pred_adv) * 100
        top1_ema = accuracy_score(y_true, y_pred_ema) * 100 
        top1_adv_ema = accuracy_score(y_true, y_pred_adv_ema) * 100 

    return top1, top1_adv, top1_ema, top1_adv_ema 

def main():
    best_acc_clean = 0
    best_acc_adv = best_ema_acc_adv = 0
    start_epoch = 1

    if args.arch == "smallcnn":
        model = SmallCNN()
    if args.arch == "resnet18":
        model = ResNet18(num_classes=args.num_classes)
    if args.arch == "WRN":
        model = Wide_ResNet_Madry(depth=32, num_classes=args.num_classes, widen_factor=10, dropRate=0.0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.nn.DataParallel(model)
    teacher_model = EMA(model)
    # model = model.to(device)

    if not args.resume:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
        
        logger_test = Logger(os.path.join(args.out_dir, 'log_results.txt'), title='reweight')
        logger_test.set_names(['Epoch', 'Natural', 'PGD20', 'ema_Natural', 'ema_PGD20'])

        for epoch in range(start_epoch, args.epochs+1):
            
            descrip_str = 'Training epoch:{}/{}'.format(epoch, args.epochs)

            train(epoch, model, teacher_model, optimizer, device, descrip_str)
            nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc = test(model, teacher_model, device=device)
            print("model natural accuracy:", nat_acc,"pgd accuracy", pgd20_acc, "ema_model natural accuracy", ema_nat_acc, "ema pgd accuracy", ema_pgd20_acc) 
            logger_test.append([epoch, nat_acc, pgd20_acc, ema_nat_acc, ema_pgd20_acc])
            
            if pgd20_acc > best_acc_adv:
                print('==> Updating the best model..')
                best_acc_adv = pgd20_acc
                torch.save(model.state_dict(), os.path.join(args.out_dir, 'bestpoint.pth.tar'))

            if ema_pgd20_acc > best_ema_acc_adv:
                print('==> Updating the teacher model..')
                best_ema_acc_adv = ema_pgd20_acc
                torch.save(teacher_model.model.state_dict(), os.path.join(args.out_dir, 'ema_bestpoint.pth.tar'))

            # # Save the last checkpoint
            # torch.save(model.state_dict(), os.path.join(args.out_dir, 'lastpoint.pth.tar'))

    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'bestpoint.pth.tar')))
    teacher_model.model.load_state_dict(torch.load(os.path.join(args.out_dir, 'ema_bestpoint.pth.tar')))
    #res_list = attack(model, Attackers, device)
    #res_list1 = attack(teacher_model.model, Attackers, device)

    #logger_test.set_names(['Epoch', 'clean', 'PGD20', 'PGD100', 'MIM', 'CW', 'APGD_ce', 'APGD_dlr', 'APGD_t', 'FAB_t', 'Square', 'AA'])
    #logger_test.append([1000000, res_list[0], res_list[1], res_list[2], res_list[3], res_list[4], res_list[5], res_list[6], res_list[7], res_list[8], res_list[9], res_list[10]])
    #logger_test.append([1000001, res_list1[0], res_list1[1], res_list1[2], res_list1[3], res_list1[4], res_list1[5], res_list1[6], res_list1[7], res_list1[8], res_list1[9], res_list1[10]])

    logger_test.close()


if __name__ == '__main__':
    main()