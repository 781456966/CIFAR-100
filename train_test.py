import pandas as pd

from model import *

for k in [3,0,1,2]:

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    if k == 0:
        use_cutmix = False
        use_mixup = False
        use_cutout = False
        name = 'baseline'
    if k == 1:
        use_cutmix = True
        use_mixup = False
        use_cutout = False
        name = 'cutmix'
    if k == 2:
        use_cutmix = False
        use_mixup = True
        use_cutout = False
        name = 'mixup'
    if k == 3:
        use_cutmix = False
        use_mixup = False
        use_cutout = True
        name = 'cutout'

    for epoch in range(1,101):
        train_arr = train(epoch,use_cutmix=use_cutmix,use_mixup=use_mixup,use_cutout=use_cutout)
        test_arr = test(epoch)

        train_loss_list.append(train_arr[0])
        train_acc_list.append(train_arr[1])
        test_loss_list.append(test_arr[0])
        test_acc_list.append(test_arr[1])

    df = pd.DataFrame()
    df['train_loss'] = train_loss_list
    df['test_loss'] = test_loss_list
    df['train_acc'] = train_acc_list
    df['test_acc'] = test_acc_list
    df.to_excel(f'{name}_loss_acc.xlsx')



