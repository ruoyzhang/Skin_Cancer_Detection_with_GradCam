def test_model(model_dir, data_dir, label_dir, batch_size, num_workers = 1):
    model = vgg_preloaded(7, cuda=False)
    model.load_state_dict(torch.load(modelpath))

    dataset = MelaData(data_dir = data_dir, label_csv = label_dir)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')

    model.eval()
    predictions = [] #Store predictions in here

    running_loss = 0.0
    running_corrects = 0
    count = 0 

    for inputs,classes in data_loader:
        outputs = model(inputs)                
        loss = loss_fn(outputs,classes) 
        _,preds = torch.max(outputs.data, 1)
        running_loss += loss        
        running_corrects += preds.eq(classes.view_as(preds)).sum()
        predictions += list(preds)
        count +=1

        
    print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / len(dataset), running_corrects.data.item() / len(dataset)))
    return {'loss': running_loss / len(dataset), 'acc': running_corrects.data.item() / len(dataset), 'predictions': predictions}
