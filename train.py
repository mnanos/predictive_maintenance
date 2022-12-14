import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
#from git import Repo
import os
import copy
import torchaudio
import pickle
from torchinfo import summary
import torch.nn as nn
import utils
import model
import matplotlib.pyplot as plot


tqdm.monitor_interval = 0
os.environ['CUDA_VISIBLE_DEVICES'] ='0'   #0 device is cuda, 10 device is CPU


def train(args, unmix, device, train_sampler, optimizer,critireon):
    losses = utils.AverageMeter()
    unmix.train()                                                              #  .to(device) works for cuda
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for X, Y in pbar:
        pbar.set_description("Training batch")
        X, Y = X.to(device), Y.to(device).float()
        optimizer.zero_grad()


        Y_hat = unmix(X)                      #.to(device) works for cuda                


        loss = critireon(Y_hat, Y.long(),).to(device)

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(0))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))
    return losses.avg


def valid(args, unmix, device, valid_sampler,critireon):
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for X, Y in valid_sampler:
            X, Y = X.to(device), Y.to(device).float()

            Y_hat = unmix(X)

            loss = critireon(Y_hat, Y.long())

            losses.update(loss.item(), Y.size(0))
        return losses.avg

def get_statistics(args, encoder, dataset):
    encoder = copy.deepcopy(encoder).to("cpu")
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    # if isinstance(dataset_scaler, data.SourceFolderDataset):
    #     dataset_scaler.random_chunks = False
    # else:
    #     dataset_scaler.random_chunks = False
    #     dataset_scaler.seq_duration = None

    dataset_scaler.random_chunks = False
    dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))

    return scaler.mean_, std



def main():
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")



    # Dataset paramaters
    parser.add_argument("--root", type=str, help="root path of dataset")

    parser.add_argument(
        "--output",
        type=str,
        default="open-unmix",
        help="provide output path base folder name",
    )

    parser.add_argument("--nb_classes", type=int, default=1)    


    parser.add_argument("--model", type=str, help="Name or path of pretrained model to fine-tune")
    parser.add_argument("--checkpoint", type=str, help="Path of checkpoint to resume training")
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`",
    )

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument(
        "--patience",
        type=int,
        default=300,
        help="maximum number of train epochs (default: 140)",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )


    parser.add_argument(
        "--nb-workers", type=int, default=0, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    # parser.add_argument(
    #     "--saved-statistics", action="store_true", default=False, help="flag to indicate if the statistics are calculated"
    # )

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    # repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # repo = Repo(repo_dir)
    # commit = repo.head.commit.hexsha[:7]

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("the model will be running on", device, "device")

    # train_dataset, valid_dataset, args = data.load_datasets(parser, args)

    train_dataset = torch.load(args.root+'/Spec_seg_pair_list_train.pt')
    valid_dataset = torch.load(args.root+'/Spec_seg_pair_list_valid.pt')



    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs)
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, **dataloader_kwargs)


    spec_shape = train_dataset[0][0].shape
    
    fc_dim = spec_shape[1]*spec_shape[2]
    utils.fc_dim = fc_dim

    if args.model:
        # fine tune model
        print(f"Fine-tuning model from {args.model}")
        unmix = utils.load_target_models(
            "model", model_str_or_path=args.model, device=device, pretrained=True
        )["model"]
        unmix = unmix.to(device)
    else:
        unmix = model.Net(fc_dim=fc_dim,nb_classes=args.nb_classes).to(device).float()


    # critireon = nn.BCEWithLogitsLoss()
    #compute the weights for the cross Entropy Loss
    import sklearn
    y = list( map( lambda seg : seg[1] , train_dataset ) )
    class_weights= sklearn.utils.class_weight.compute_class_weight('balanced',classes=np.unique(y),y=y)
    ## Transfer to GPU
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
    critireon = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    optimizer = torch.optim.Adam(unmix.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    # if a checkpoint is specified: resume training
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, "model" + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, "model" + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # else start optimizer from scratch
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0


    #[print(sample[0].shape) for sample in train_dataset]
    with torch.no_grad():
        it = iter(train_sampler)
        X = next(it)[0].to(device)   #works for windows depends on torch version
      #X = next(it)[0].to(device) --> works for linux
        summary(unmix, X.shape)
     

    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        train_loss = train(args, unmix , device, train_sampler, optimizer,critireon)
        valid_loss = valid(args, unmix, device, valid_sampler,critireon)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target="model",
        )

        # save params
        with open(args.root+'/Dataset_Params_log.json', 'r') as openfile:
            # Reading from json file
            Dataset_params = json.load(openfile)        

        args_json = vars(args)

        args_json["Dataset_params"] = Dataset_params

        args_json["fc_dim"] = fc_dim

        params = {
            "epochs_trained": epoch,
            # "args": vars(args),
            "args": args_json,
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
            #"commit": commit,
        }

        with open(Path(target_path, "model" + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=False))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break

    #PLOTTING--------------------------------
    #fig, axs = plot.subplots(nrows=1, ncols=1, figsize=(10, 7))

    plot.plot(train_losses,'-bx',label='train')
    plot.plot(valid_losses,'-rx',label='valid')
    plot.xlabel('epoch')
    plot.ylabel('loss')
    plot.title('Train Loss vs No of epochs')
    plot.legend()
    
    plot.show()        
    
    # axs[1,0].set_title('Validation Losses')
    # axs[1,0].plot(valid_losses)
    # axs[1,0].set_xlabel('Epoch')
    # axs[1,0].set_ylabel('Loss')
    
    # axs[0,1].set_title('Train time History')
    # axs[0,1].plot(train_times)
    # #plot.specgram(X_spec_full,Fs=Dataset_params["Fs"])
    # axs[0,1].set_xlabel('Epoch')
    # axs[0,1].set_ylabel('Time')
    
    
    # axs[1,1].set_title('Number Bad epochs')
    # axs[1,1].plot(es.num_bad_epochs)
    # axs[1,1].set_xlabel('Epoch')
    # axs[1,1].set_ylabel('number')

    #plot.tight_layout()
    
    
    
if __name__ == "__main__":
    main()
