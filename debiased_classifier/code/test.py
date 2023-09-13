import torch
from tqdm import tqdm
from skimage import io
import os
import argparse
import h5py
from torch.autograd import Variable

from dataset import create_dataset
from model import Network
from utilities import utils

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
use_cuda = torch.backends.mps.is_available()
device = torch.device("mps" if use_cuda else "cpu")

def test(config_file):
    """
    Loads the config file with data_path,csv details and model path
    Calculate the accuracy, f1score and confusion matrix on test set
    """
    config = utils.parse_configuration(config_file)
    saved_model = config["model_path"]
    [test_loader, class_names] = create_dataset(config, mode="test")

    model = Network(len(class_names))
    model.load_state_dict(torch.load(saved_model))
    model.to(device)

    utils.evaluator(model, test_loader, class_names, device)

def feature_extraction(config_file):  
    """
    Takes the config file with data,csv and model details
    Extracts the features to a hdf5 file
    """
    config = utils.parse_configuration(config_file)
    saved_model = config["model_path"]
    config["batch_size"]=1
    [test_loader, class_names] = create_dataset(config, mode="test")

    model = Network(len(class_names))
    model.load_state_dict(torch.load(saved_model))
    model.to(device)
    

    def get_features(name):
      def hook(model, input, output):
          features[name] = output.detach()
      return hook



    #register hook at the pool layer for retrieval purpose
    model.pool.register_forward_hook(get_features('pool_features'))
  
    model.eval()
    features = {}
    predictions = []
    feats =[]


    for batch in tqdm(test_loader):
        preds = model(batch[0].to(device))   
        _,pred_label=torch.max(preds,1)  
        predictions.append(pred_label.detach().cpu().numpy())
       
        feats.append(features['pool_features'].view(-1).cpu().numpy())
   

    #saves the image index,features, and final output from network to hdf5
    filename=config['feature_savepath']
    with h5py.File(filename, "w") as f:
        f.create_dataset("final_output", data=predictions)
        f.create_dataset("pool_features", data=feats)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform model Testing..")
    parser.add_argument("configfile", help="path to the configfile")
    args = parser.parse_args()
    # test(args.configfile)
    feature_extraction(args.configfile)
