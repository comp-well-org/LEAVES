import torch
import torch.nn as nn
import configs
from tqdm import tqdm
from models.SimCLR import SimCLRObjective, ContrastiveLoss
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from scipy.special import softmax

def trainSimCLR(model, trainloader, testloader, device):
    optimizer_view = torch.optim.SGD(model.view.parameters(), lr=configs.LR, momentum=0.9)
    optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=configs.LR)
    
    contrastiveLoss = ContrastiveLoss(configs.batchsize).to(device)
    tb_writer = SummaryWriter(log_dir = configs.save_path, comment='init_run')
    for e in range(configs.epochs):
        model.train()
        epoch_loss_encoder, epoch_loss_view = 0, 0
        for batch in tqdm(trainloader):
            x1, x2, label = batch
            x1_emb, x2_emb = model(x1.to(device, dtype=torch.float), 
                                   x2.to(device, dtype=torch.float))
            # simclr_view = SimCLRObjective(x1_emb.squeeze(-1), x2_emb.squeeze(-1), t=0.07)
            # simclr_encoder = SimCLRObjective(x1_emb.squeeze(-1), x2_emb.squeeze(-1), t=0.07)
            encoder_loss = contrastiveLoss(x1_emb, x2_emb)
            view_maker_loss = - encoder_loss.clone()
            
            optimizer_encoder.zero_grad()
            optimizer_view.zero_grad()
            
            for param in model.encoder.parameters():
                param.requires_grad = False
            view_maker_loss.backward(retain_graph=True)
            for param in model.encoder.parameters():
                param.requires_grad = True
            
            for param in model.view.parameters():
                param.requires_grad = False
            encoder_loss.backward()
            for param in model.view.parameters():
                param.requires_grad = True
            
            optimizer_view.step()
            optimizer_encoder.step()
            
            epoch_loss_encoder += encoder_loss.item() / len(trainloader)
            epoch_loss_view += view_maker_loss.item() / len(trainloader)
            
        tb_writer.add_scalar('Loss/Encoder', epoch_loss_encoder, e)
        tb_writer.add_scalar('Loss/View', epoch_loss_view, e)
        print('Training Epoch {} - Encoder Loss : {}, View Loss : {}'.format(e, 
                                                                             epoch_loss_encoder, 
                                                                             epoch_loss_view))
        if e % 10 == 9:
            save_path = configs.save_path + 'checkpoint_{}.pth'.format(e + 1)
            # print(confusion_matrix(y_test, pred_y))
            torch.save(model.state_dict(), save_path)
            
            
def trainLinearEvalution(model, trainloader, testloader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    lr_scheduler.step();   
    criterion = nn.BCEWithLogitsLoss()
    best_acc = -0.1
    save_path = configs.save_path + 'best-model.pth'
    for epoch in range(configs.epochs):
        print("Epoch {}/{}".format(epoch, configs.epochs-1));
        print('+' * 80)

        train_losses = []
        train_true_labels = []
        train_pred_labels = []

        model.train()
        for x, labels in trainloader:
            # move data to GPU
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device)

            # reset optimizer.
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()


            # obtain necessary information for displaying.
            train_losses.append(loss.item())
            train_pred_labels.append(logits.detach().cpu())
            train_true_labels.append(labels.detach().cpu())
        lr_scheduler.step()
        all_pred = np.vstack(train_pred_labels)
        all_true = np.vstack(train_true_labels)
        # convert from one-hot coding to binary label.
        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        #all_pred_binary = logits_2_binary(all_pred)
        #all_true_binary = all_true
        # output training information after each epoch.
        print("                         Training:")
        print("Loss: %.4f" %(np.mean(np.array(train_losses))))
        #F1 = f1_score(all_true_binary, all_pred_binary)
        #print("F1 score: %.4f" %(F1))
        ACC = accuracy_score(all_true_binary, all_pred_binary)
        print("Accuracy: %.4f " %(ACC))
        print(confusion_matrix(all_true_binary, all_pred_binary))
        test_losses = []
        test_pred_labels = []
        test_true_labels = []
        model.eval()
        for x, labels in testloader:
            x = x.to(device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(logits,labels)

            test_losses.append(loss.item())
            test_true_labels.append(labels.detach().cpu())
            test_pred_labels.append(logits.detach().cpu())

        all_pred = np.vstack(test_pred_labels)
        all_true = np.vstack(test_true_labels)

        all_pred_binary = np.argmax(all_pred, axis=1)
        all_true_binary = np.argmax(all_true, axis=1)
        #all_pred_binary = logits_2_binary(all_pred)
        #np.save('./GSR.npy', all_pred_binary)
        #all_true_binary = all_true
        print("                         Testing:")
        print("Loss: %.4f" %(np.mean(np.array(test_losses))))
        #print("F1 score: %.4f" %(f1_score(all_true_binary, all_pred_binary)))
        print("Accuracy: %.4f " %(accuracy_score(all_true_binary, all_pred_binary)))
        print("f1(macro): %.4f " %(f1_score(all_true_binary, all_pred_binary, average='macro')))
        print("sen: %.4f " %(f1_score(all_true_binary, all_pred_binary, average='macro')))
        AUC = roc_auc_score(all_true, softmax(all_pred,axis=1))
        print("AUC:", AUC)
        print(confusion_matrix(all_true_binary, all_pred_binary))
        
        if f1_score(all_true_binary, all_pred_binary, average='macro') > best_acc:
            best_acc = f1_score(all_true_binary, all_pred_binary, average='macro');
            print("Save new best model")
            torch.save(model.state_dict(), save_path)

     
    # torch.save(model.state_dict(), os.path.join(args.model_weights_dir, 'model.std'))
 
    print("#"*50)
     # Test the model
    model.load_state_dict(torch.load(save_path))

    test_losses = []
    test_pred_labels = []
    test_true_labels = []
    model.eval()
    for x, labels in testloader:
        x = x.to(device, dtype=torch.float32)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(x)

        loss = criterion(logits,labels)

        test_losses.append(loss.item())
        test_true_labels.append(labels.detach().cpu())
        test_pred_labels.append(logits.detach().cpu())

    all_pred = np.vstack(test_pred_labels)
    all_true = np.vstack(test_true_labels)

    all_pred_binary = np.argmax(all_pred, axis=1)
    all_true_binary = np.argmax(all_true, axis=1)
    #all_pred_binary = logits_2_binary(all_pred)
    #np.save('./GSR.npy', all_pred_binary)
    #all_true_binary = all_true
    print("                         Testing:")
    print("Loss: %.4f" %(np.mean(np.array(test_losses))))
    #print("F1 score: %.4f" %(f1_score(all_true_binary, all_pred_binary)))
    print("Accuracy: %.4f " %(accuracy_score(all_true_binary, all_pred_binary)))
    print(confusion_matrix(all_true_binary, all_pred_binary))