import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from main_model import CSDI_Stock
from dataset_stock import get_dataloader
from utils import train, evaluate
from tqdm import tqdm
from view import viewresult

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

missratio = config["model"]["test_missing_ratio"]
timelength = config["more"]["timestep"]
stimcount = config["more"]["stimcount"]
drift = config["more"]["drift"]
sigma = config["more"]["sigma"]
idea = config["more"]["idea"]
log = False
if (config["more"]["log"]=="y"):
    log=True

print(missratio, timelength, stimcount, drift, sigma, idea)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/stk" + "_" + current_time + "/"
#you may prefer better changing the profile name

traindata, train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=missratio,
    timelength=timelength,
    stimcount=stimcount,
    drift=drift,
    sigma=sigma,
    idea=idea,
    log=logs
)

os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

model = CSDI_Stock(config, args.device).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

outputbatchcount=2

traindata=traindata.squeeze()
torch.save(traindata, foldername+'traindata.pt')
nsample=args.nsample
with torch.no_grad(): #this part is copied from the "evaluate" function in utils.py
    model.eval()
    mse_total = 0
    mae_total = 0
    evalpoints_total = 0
    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []
    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            output = model.evaluate(test_batch, nsample)
            samples, c_target, eval_points, observed_points, observed_time = output
            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            samples = samples.squeeze()
            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
            c_target = c_target.squeeze()
            eval_points = eval_points.permute(0, 2, 1)
            eval_points = eval_points.squeeze()
            observed_points = observed_points.permute(0, 2, 1)
            samples_median, samples_median_i = samples.median(dim=1)
            if (batch_no<=outputbatchcount):
                torch.save(samples, foldername+'samples'+str(batch_no)+'.pt')
                torch.save(c_target, foldername+'c_target'+str(batch_no)+'.pt')
            else:
                torch.save(eval_points, foldername+'eval_points.pt')
                torch.save(observed_points, foldername+'observed_points.pt')
                torch.save(observed_time, foldername+'observed_time.pt')
                break

viewresult(foldername)


