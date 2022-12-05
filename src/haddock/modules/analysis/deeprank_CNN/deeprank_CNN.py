from mpi4py import MPI
import sys
import os
import re
import glob
from time import time

import deeprank
from deeprank.generate import *
from deeprank.learn import NeuralNet
from model_280619 import cnn_class

import torch
import h5py
import torch.nn.functional as F
import numpy as np
from pssmgen import PSSM

#used to rank docking poses

class CNN_score():
    def __init__(self, pdb_source, chain1, chain2, output_dir):
        self.pdb_source = pdb_source
        self.chain1 = chain1
        self.chain2 = chain2
        self.output_dir = output_dir

    def generate_pssm(self):
        gen = PSSM(work_dir=self.pdb_source)
        gen.configure(blast_exe='/trinity/login/xxu/software/ncbi-blast-2.13.0+/bin/psiblast',
            database='/trinity/login/xxu/data/DBs/nr',
            num_threads=4, evalue=0.0001, comp_based_stats='T',
            max_target_seqs=2000, num_iterations=3, outfmt=7,
            save_each_pssm=True, save_pssm_after_last_round=True)
        gen.get_fasta(pdb_dir='', chain=(self.chain1,self.chain2), out_dir='fasta')
        gen.get_pssm(fasta_dir='fasta', out_dir='pssm_raw', run=True)
        gen.map_pssm(pssm_dir='pssm_raw', pdb_dir='', out_dir='pssm', chain=(self.chain1,self.chain2))
        gen.get_mapped_pdb(pdbpssm_dir='pssm', pdb_dir='', pdbnonmatch_dir='pdb_nonmatch')

    def create_database(self):
        comm = MPI.COMM_WORLD
        self.hdf_dir = os.path.join(self.output_dir, 'output.hdf5')
        database = DataGenerator(pdb_source= self.pdb_source, #path to the models  
                         pssm_source=os.path.join(self.pdb_source, 'pssm'), #path to the pssm data
                         data_augmentation = None,
                         chain1=self.chain1, chain2=self.chain2,
                         compute_features = ['deeprank.features.AtomicFeature', 'deeprank.features.FullPSSM','deeprank.features.PSSM_IC'
                         , 'deeprank.features.BSA', 'deeprank.features.ResidueDensity'],
                         hdf5=self.hdf_dir,mpi_comm=comm)
        database.create_database(prog_bar=True)

        grid_info = {
            'number_of_points' : [30,30,30],
            'resolution' : [1.,1.,1.],
            'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
            }
        database.map_features(grid_info,try_sparse=True, time=False, prog_bar=True)

    def test_CNN(self):
        model_data = 'best_train_model.pt'
        database = self.hdf_dir
        model = NeuralNet(database, 
                         cnn_class, 
                         task='class', 
                         pretrained_model=model_data, 
                         save_hitrate=False,
                         outdir=self.output_dir)
        model.test(hdf5='prediction.hdf5')

    def analysis_result(self):
        path = os.path.join(self.output_dir, 'prediction.hdf5')
        f = h5py.File(path)
        out_class = f['epoch_0000']['test']['outputs'][()]
        out = F.softmax(torch.FloatTensor(out_class),dim=1).data.numpy()[:, 1]
        probility = F.softmax(torch.FloatTensor(out_class), dim=1).data.numpy()
        preds = pred.astype(int)
        mols = f['epoch_0000']['test']['mol']
        result_dict = {}
        for i in range(len(mols)):
            result_dict[mols[i][-1].decode()] = (preds[i], out[i])
        return result_dict
'''
#test the class
pdb_source = '/trinity/login/xxu/scripts/test_class'
outdir = '/trinity/login/xxu/scripts/test_class'
CNN = CNN_score(pdb_source=pdb_source, chain1='C', chain2='D', output_dir=outdir)
CNN.generate_pssm()
CNN.create_database()
CNN.test_CNN()
print(CNN.analysis_result())
'''

