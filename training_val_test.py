import numpy as np
from data import TUTDatamodule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch
from optparse import OptionParser
from sklearn import metrics
from model import Wavegram_AttentionMap


def transform_string_list(option, opt, value, parser):
    list_of_int =  [int(i) for i in value.split(",")]
    setattr(parser.values, option.dest, list_of_int)


def train(configs):
    # definition of the logger
    tags = ["MobileNetWavegram"]

    # Machine type if we want to train and test on a single machine
    if "valve" in configs.path_train:
        tags.append("Valve")
    elif "pump" in configs.path_train:
        tags.append("Pump")
    elif "slider" in configs.path_train:
        tags.append("Slider")
    elif "ToyCar" in configs.path_train:
        tags.append("ToyCar")
    elif "ToyConveyor" in configs.path_train:
        tags.append("ToyConveyor")
    elif "fan" in configs.path_train:
        tags.append("Fan")
    else:
        tags.append("All")
    wandb_logger = WandbLogger(project="Unsupervised Audio Anomaly Detection", config = configs,  name=configs.name, tags = tags)

    checkpoint_callback = ModelCheckpoint(
        dirpath = "best_models/", 
        filename = "MobileNetWavegram-Mixup-{epoch:02d}-{val/loss_class:.4f}",  # <--- note epoch suffix here
        save_last = True, 
        every_n_epochs = 20,
        save_top_k = -1,
        auto_insert_metric_name=False
    ) 

    # definition of the trainer 
    trainer = Trainer(accelerator="gpu", devices = 1 , max_epochs = configs.epochs, logger=wandb_logger, callbacks = [checkpoint_callback])

    # definition of the datamodule
    datamodule = TUTDatamodule(path_train = configs.path_train, path_test = configs.path_test, sample_rate = configs.sr, duration = configs.duration,
                                percentage_val = configs.percentage, batch_size = configs.batch_size)
    
    # definition of the model
    model = Wavegram_AttentionMap(lr = configs.lr, h = 128)

    wandb_logger.watch(model, log_graph=False)
    trainer.fit(model, datamodule)
    return model, trainer, datamodule

def val(trained_model:Wavegram_AttentionMap, trainer:Trainer, datamodule:TUTDatamodule):
    # inference on the validation set
    trainer.validate(trained_model, datamodule)    
    return trained_model, trainer, datamodule

def test(trained_model:Wavegram_AttentionMap, trainer:Trainer, datamodule:TUTDatamodule):
    # inference for tests
    # make list of files insider "best_models" folder
    list_checkpoints = datamodule.scan_all_dir("best_models/")
    best_performance = 0
    best_checkpoint = None
    for i, checkpoint in enumerate(list_checkpoints):
        print("{}) {}".format(i, checkpoint))
        trained_model = Wavegram_AttentionMap.load_from_checkpoint(checkpoint,  lr = 0.001, h = 128) # dummy lr
        trained_model.errors_list = []
        trained_model.anomaly_errors = []
        trained_model.clean_errors = []
        trainer.test(trained_model, datamodule)
            
        all_errors = torch.cat(trained_model.errors_list).flatten().cpu().numpy()
        all_labels = torch.cat(trained_model.labels).flatten().cpu().numpy()
        metadata = np.array(torch.cat(trained_model.classes).tolist())
        performance = metrics.roc_auc_score(all_labels, all_errors)
        if performance > best_performance:
            best_performance = performance
            best_checkpoint = checkpoint
            print("Best : {} %".format(best_performance))

    # here we select the best model
    trained_model = Wavegram_AttentionMap.load_from_checkpoint(best_checkpoint,  lr = 0.001, h = 128) # dummy lr
    trained_model.errors_list = []
    trained_model.anomaly_errors = []
    trained_model.clean_errors = []
    trainer.test(trained_model, datamodule)

    all_errors = torch.cat(trained_model.errors_list).flatten().cpu().numpy()
    all_labels = torch.cat(trained_model.labels).flatten().cpu().numpy()
    metadata = np.array(torch.cat(trained_model.classes).tolist())

    np.savez("metadata", metadata)
    np.savez("allerrors", all_errors)
    np.savez("all_labels", all_labels)
    wandb.log({"test/AUROC": metrics.roc_auc_score(all_labels, all_errors)})
    wandb.log({"test/pAUROC": metrics.roc_auc_score(all_labels, all_errors, max_fpr = 0.1)})

    print("Global AUC : {}".format(metrics.roc_auc_score(all_labels, all_errors)))
    print("Global pAUC: {}".format(metrics.roc_auc_score(all_labels, all_errors, max_fpr = 0.1)))

    # fan class
    mask = metadata < 7
    selected_label = all_labels[mask]
    selected_errors = all_errors[mask]

    print("Fan AUC : {}".format(metrics.roc_auc_score(selected_label, selected_errors)))
    print("Fan pAUC: {}".format(metrics.roc_auc_score(selected_label, selected_errors, max_fpr = 0.1)))

    # pump class
    mask = np.logical_and(metadata >= 7, metadata < 13)
    selected_label = all_labels[mask]
    selected_errors = all_errors[mask]

    print("Pump AUC : {}".format(metrics.roc_auc_score(selected_label, selected_errors)))
    print("Pump pAUC: {}".format(metrics.roc_auc_score(selected_label, selected_errors, max_fpr = 0.1)))


    # slider class
    mask = np.logical_and(metadata >= 13, metadata < 20)
    selected_label = all_labels[mask]
    selected_errors = all_errors[mask]

    print("Slider AUC : {}".format(metrics.roc_auc_score(selected_label, selected_errors)))
    print("Slider pAUC: {}".format(metrics.roc_auc_score(selected_label, selected_errors, max_fpr = 0.1)))

    # ToyCar class
    mask = np.logical_and(metadata >= 20, metadata < 27)
    selected_label = all_labels[mask]
    selected_errors = all_errors[mask]

    print("ToyCar AUC : {}".format(metrics.roc_auc_score(selected_label, selected_errors)))
    print("ToyCar pAUC: {}".format(metrics.roc_auc_score(selected_label, selected_errors, max_fpr = 0.1)))


    # ToyConveyor class
    mask = np.logical_and(metadata >= 27, metadata < 34)
    selected_label = all_labels[mask]
    selected_errors = all_errors[mask]

    print("ToyConveyor AUC : {}".format(metrics.roc_auc_score(selected_label, selected_errors)))
    print("ToyConveyor pAUC: {}".format(metrics.roc_auc_score(selected_label, selected_errors, max_fpr = 0.1)))  

    # valve class
    mask = metadata >= 34
    selected_label = all_labels[mask]
    selected_errors = all_errors[mask]

    print("Valve AUC : {}".format(metrics.roc_auc_score(selected_label, selected_errors)))
    print("Valve pAUC: {}".format(metrics.roc_auc_score(selected_label, selected_errors, max_fpr = 0.1)))   

    clean_errors = np.array(torch.cat(trained_model.clean_errors).tolist())
    anomalous_errors = np.array(torch.cat(trained_model.anomaly_errors).tolist())
    np.savez("cleanerrors", clean_errors)
    np.savez("anomalouserrors", anomalous_errors)
    wandb.finish()

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 28
    # plt.rcParams['text.usetex'] = True <--- If you have latex installed in your machine, you can use it on matplotlib
    errors_clean = np.load("cleanerrors.npz")['arr_0']
    errors_anomalous = np.load("anomalouserrors.npz")['arr_0']
    print(errors_clean.shape)
    print(errors_anomalous.shape)
    plt.figure(figsize = (8, 12))
    plt.hist(errors_clean, bins = 200, alpha = 0.4, label = "Clear", color = "g")
    plt.hist(errors_anomalous, bins = 200, alpha = 0.4, label = "Anomalous", color = "r")
    plt.grid()
    plt.legend()
    plt.title("Histograms of errors")
    plt.xlabel("Error")
    plt.ylabel("Occurrences")
    plt.show()
    

    


if __name__ == "__main__":
    parser = OptionParser()
    # dataset parameters
    parser.add_option("--pathtrain", dest="path_train",
                    help="path containing training files", default = "TUT Anomaly detection/train")
    parser.add_option("--pathtest", dest="path_test",
                    help="path containing training files", default = "TUT Anomaly detection/test")
    parser.add_option("--percentage", dest = "percentage",
                      help = "percentage of validation samples", default = 0.05, type = float)
    parser.add_option("--sr", dest = "sr",
                      help = "target sample rate", default = 16000, type = int)
    parser.add_option("--duration", dest = "duration",
                      help = "duration, in seconds, of audios", default = 10, type = float)      
  
    
    # training parameters 
    parser.add_option("--lr", dest = "lr",
                      help = "learning rate", default = 0.0001, type = float)
    parser.add_option("--epochs", dest = "epochs",
                      help = "number of epochs", default = 300, type = int)
    parser.add_option("--name", dest = "name",
                      help = "name of the run on wandb", default = "MobileFaceNet + separable wavegram + attention module + mixup 0.2 Noisy-ArcMix s 40 m 0.7")
    parser.add_option("--batch_size", dest = "batch_size",
                      help = "batch size for dataloader", default = 64, type = int)


    options, remainder = parser.parse_args()
    print(options)
    trained_model, trainer, datamodule = train(options)
    trained_model, trainer, datamodule = val(trained_model, trainer, datamodule)
    test(trained_model, trainer, datamodule)


