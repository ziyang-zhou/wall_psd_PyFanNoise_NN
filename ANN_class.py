import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import numpy as np
import glob
import json
import AE_ANN_module # import the saved ANN and AE class

class ANN_WPS_loader:
    def __init__(self,project_folder):
        self.project_folder = project_folder
    def load_ANN(self):
        torch.autograd.set_detect_anomaly(True)
        if torch.cuda.is_available():
            print('Using GPUs',flush=True)
        else:
            print('Using CPUs',flush=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #---------------------------------------Load directory name---------------------------------------
        self.case_folder = 'WPSst3_AEWPS_WPSlayers_3_WPSneurons_50_Lwps_0_Normwps_BatchNorm_AE_LS_3_NbLayers_3_L_0_Norm_BatchNorm_Activ_ReLU_Sched_ReduceLROnPlateau/'
        self.read_database_folder = self.project_folder + '01-DATABASES/database_SCONE-DNS_interpolated_NEW/'
        self.testcase='H08'
        self.train_case_out='C32-H08'

        self.readpath = self.project_folder + '02-TRAININGS/'+self.case_folder+'/training_{:}_out/'.format(
            self.train_case_out)
        self.savepath = self.project_folder + '04-WPSPREDICTED/'
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        self.readpathstructure = self.readpath+'ANNstructure/'
        self.readpathstate = self.readpath+'ANNstate/'

        #%%
        #---------------------------------------MAIN code---------------------------------------
        #Load the flow conditions and the boundary layer profile
        self.flowconditions = pd.read_csv(self.read_database_folder +'{:s}_flowconditions.csv'.format(self.testcase), header='infer', delimiter=' ')
        self.coordinates = pd.read_csv(self.read_database_folder+'{:s}_coordinates.csv'.format(self.testcase), header='infer', delimiter=' ')

        # Load the previously trained ANN state
        with open(self.readpathstructure+'ANNstructure_AE.json','r') as f: 
            self.ANNstructure_dict_AE = json.load(f)
        with open(self.readpathstructure+'ANNstructure_WPS.json','r') as f:
            self.ANNstructure_dict_WPS = json.load(f)
        self.checkpoint_WPS = torch.load(self.readpathstate+'saved_model_WPS.pth',
                                    map_location=torch.device('cpu'))
        self.checkpoint_AE = torch.load(self.readpathstate+'saved_model_AE.pth',
                                    map_location=torch.device('cpu'))

        #Load the mean and std of the test feature and label
        self.mean_training_features = pd.read_hdf(self.readpathstructure+'mean_training_features.h5',
                                                key='df', mode='r')
        self.std_training_features = pd.read_hdf(self.readpathstructure+'std_training_features.h5',
                                                key='df', mode='r')
        self.mean_training_labels = pd.read_hdf(self.readpathstructure+'mean_training_labels.h5',
                                            key='df', mode='r')
        self.std_training_labels = pd.read_hdf(self.readpathstructure+'std_training_labels.h5',
                                            key='df', mode='r')

        #Load the neural network AE class with the parameters read from ANNstructure_dict_AE
        self.model_AE = AE_ANN_module.neural_network_AE(input_size=self.ANNstructure_dict_AE['input_size'],
                                        output_size=self.ANNstructure_dict_AE['nb_latent_spaces'],
                                        nb_layers=self.ANNstructure_dict_AE['nb_layers'],
                                        norm_type=self.ANNstructure_dict_AE['Norm_AE'],
                                        activation_type=self.ANNstructure_dict_AE['Activation_AE'])
        self.model_AE = self.model_AE.to(self.device)
        self.model_AE.load_state_dict(self.checkpoint_AE['model_state_dict'])
        self.model_AE.eval() # This is referring to the neural_network_WPS class. It is being set to evaluation mode.

        #Load the neural network WPS class with the parameters read from ANNstructure_dict_WPS
        self.model_WPS = AE_ANN_module.neural_network_WPS(input_size=self.ANNstructure_dict_WPS['input_size'], 
                                        output_size=self.ANNstructure_dict_WPS['output_size'], 
                                        nb_layers=self.ANNstructure_dict_WPS['nb_layers'], 
                                        nb_neurons_layer=self.ANNstructure_dict_WPS['nb_neurons_layers'],
                                        layers_and_neurons=self.ANNstructure_dict_WPS['layers_and_neurons'],
                                        norm_type=self.ANNstructure_dict_WPS['Norm_WPS'],
                                        activation_type=self.ANNstructure_dict_WPS['Activation_WPS'])
        self.model_WPS = self.model_WPS.to(self.device)
        self.model_WPS.load_state_dict(self.checkpoint_WPS['model_state_dict'])
        self.model_WPS.eval() # This is referring to the neural_network_WPS class. It is being set to evaluation mode.

        self.means = [self.mean_training_features,self.mean_training_labels]
        self.stds = [self.std_training_features,self.std_training_labels]
