
import sys, os, glob
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import nglview as nv
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore")

import modeller
from modeller import *
from modeller.scripts import complete_pdb

import biobox as bb

from tqdm import tqdm

import molearn
from molearn.models.foldingnet import AutoEncoder
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def dope_score(fname):
    env = Environ()
    env.libs.topology.read(file='$(LIB)/top_heav.lib')
    env.libs.parameters.read(file='$(LIB)/par.lib')
    mdl = complete_pdb(env, fname)
    atmsel = Selection(mdl.chains[0])
    score = atmsel.assess_dope()
    return score

def main():
    
    dope_dataset = []

    data = molearn.PDBData()
    data.import_pdb(f'alignToOneHelix.pdb')
    data.atomselect(atoms=['CA', 'C', 'CB', 'N', 'O'])
    data.prepare_dataset()

    input_set = data.dataset
    stdval = data.std
    meanval = data.mean
    mol = data.mol
    atom_names = data.get_atominfo()

    for i in tqdm(range(input_set.shape[0])):

        # calculate DOPE score of input dataset
        crd_ref = input_set[i].permute(1,0).unsqueeze(0).data.cpu().numpy()*stdval + meanval
        mol.coordinates = deepcopy(crd_ref)
        mol.write_pdb("tmp.pdb")
        s = dope_score("tmp.pdb")
        dope_dataset.append(s)

    #print(dope_dataset)
    dfInputSet = pd.DataFrame(dope_dataset)
    dfInputSet.to_csv('InputDOPEOneHelix.csv', index=False)
    
main()