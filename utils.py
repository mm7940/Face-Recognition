import torch
from tqdm.notebook import tqdm_notebook
import matplotlib.pyplot as plt

def plot(H, args):
    fig, axis = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(15)

    axis[0].plot([x+1 for x in range(args['epochs'])], H['train_loss'], 'g', label = 'Train Loss')
    axis[0].plot([x+1 for x in range(args['epochs'])], H['val_loss'], 'r', label = 'Val Loss')
    axis[0].legend()
    axis[0].set_xlabel('Epochs')
    axis[0].set_ylabel('Loss')
    axis[0].set_title('Loss vs Epochs')

    axis[1].plot([x+1 for x in range(args['epochs'])], H['train_accuracy'], 'g', label = 'Train Accuracy')
    axis[1].plot([x+1 for x in range(args['epochs'])], H['val_accuracy'], 'r', label = 'Val Accuracy')
    axis[1].legend()
    axis[1].set_xlabel('Epochs')
    axis[1].set_ylabel('Accuracy')
    axis[1].set_title('Accuracy vs Epochs')
    
    name = args['arct_name']
    plt.savefig(f'./{name}/plot.png')
    plt.show()

def train(arct, train_ds, val_ds, args):
    optimizer = torch.optim.SGD(
        params = arct.parameters(),
        lr = args['learning_rate'],
        momentum = args['momentum'],
        weight_decay = args['weight_decay']
    )
    lossFunction = torch.nn.CrossEntropyLoss(reduction='sum')
    H = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }

    trainLoader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=args['batch_size'])
    valLoader = torch.utils.data.DataLoader(val_ds, shuffle=True, batch_size=args['batch_size'])

    def lossValue(inputs, labels):
        inputs = inputs.to(args['device'])
        labels = labels.to(args['device'])
        outputs = arct(inputs.float())

        matched = torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1))
        loss = lossFunction(outputs, labels)
        return loss,matched

    for epoch in tqdm_notebook(range(args['epochs'])):

        total_train_loss = 0
        total_val_loss = 0
        total_val_matched = 0
        total_train_matched = 0

        for inputs, labels in trainLoader:
            loss, matched = lossValue(inputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_matched += matched.item()

        with torch.no_grad():
            arct.eval()
            for inputs, labels in valLoader:
                loss, matched = lossValue(inputs, labels)
                total_val_loss += loss.item()
                total_val_matched += matched.item()

        avg_train_loss = total_train_loss/len(train_ds)
        avg_val_loss = total_val_loss/len(val_ds)
        train_accuracy = total_train_matched/len(train_ds)
        val_accuracy = total_val_matched/len(val_ds)

        H['train_loss'].append(avg_train_loss)
        H['val_loss'].append(avg_val_loss)
        H['train_accuracy'].append(train_accuracy)
        H['val_accuracy'].append(val_accuracy)
        
        if args['output']:
            print('EPOCH: {}/{}'.format(epoch+1, args['epochs']))
            print('Train loss: {:.4f}, Val loss: {:.4f},\nTrain Accuracy: {:.4f}, Val Accuracy: {:.4}'.format(avg_train_loss, avg_val_loss, train_accuracy, val_accuracy))

    if args['save']:
        name = args['arct_name']
        torch.save(arct, f'./{name}/model.pt')

    return H