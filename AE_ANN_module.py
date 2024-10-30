import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

class neural_network_WPS(nn.Module):
    def __init__(self, input_size, output_size, 
                 nb_layers=0, nb_neurons_layer=0,
                 layers_and_neurons=[],
                 norm_type='ReLU', activation_type='ReLU'):
        
        super(neural_network_WPS, self).__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        if nb_layers != 0:
            self.nb_layers = int(nb_layers)
        else:
            self.nb_layers = 10 #default value
        self.nb_neurons_layer = int(nb_neurons_layer)
        if layers_and_neurons:
            self.layers_and_neurons = layers_and_neurons
        else:
            self.layers_and_neurons = []
        self.norm_type = norm_type
        self.activation_type = activation_type
        layers = np.arange(1, self.nb_layers + 1)

        if self.nb_neurons_layer != 0:
            neurons_WPS = np.array(self.nb_neurons_layer* \
                                   np.ones(self.nb_layers + 1),
                                   dtype=int)
        elif self.layers_and_neurons:
            neurons_WPS = []
            neurons_WPS.append(self.input_size)
            for layerneuron in self.layers_and_neurons:
                neurons_WPS.append(layerneuron)
            neurons_WPS.append(self.output_size)
            neurons_WPS = np.array(neurons_WPS)
        else:
            neurons_WPS = np.linspace(self.input_size, 
                                      self.output_size, 
                                      self.nb_layers + 1,
                                      dtype=int)
        
        neurons_WPS[0] = self.input_size
        neurons_WPS[-1] = self.output_size

        self.WPS_layers = nn.ModuleList()
        self.WPS_norm_layers = nn.ModuleList()
        self.WPS_activations = nn.ModuleList()
        # Create WPS layers and append them to the list                                           
        for i1, j1 in enumerate(layers):
            
            #norm layers
            if self.norm_type.lower() == 'BatchNorm'.lower():
                WPS_norm_layer = nn.BatchNorm1d(neurons_WPS[i1])
            elif self.norm_type.lower() == 'LayerNorm'.lower():
                WPS_norm_layer = nn.LayerNorm(neurons_WPS[i1])
            else:
                WPS_norm_layer = nn.Identity()
            self.WPS_norm_layers.append(WPS_norm_layer)

            #layers
            WPS_layer = nn.Linear(in_features=int(neurons_WPS[i1]), 
                                  out_features=int(neurons_WPS[i1 + 1]))
            self.WPS_layers.append(WPS_layer)
            
            #activation of layer
            if self.activation_type.lower() == 'ReLU'.lower():
                WPS_activation = nn.ReLU()
            elif self.activation_type.lower() == 'sigmoid'.lower():
                WPS_activation = nn.Sigmoid()
            elif self.activation_type.lower() == 'tanh'.lower():
                WPS_activation = nn.Tanh()
            else:
                WPS_norm_layer = nn.Identity()
            self.WPS_activations.append(WPS_activation)
            

    def forward(self, x):
        for l1, (WPS_norm_layer, WPS_layer, WPS_activation) in enumerate(
            zip(self.WPS_norm_layers,
                self.WPS_layers,
                self.WPS_activations)):
            x = WPS_norm_layer(x)
            x = WPS_layer(x)
            if l1 < len(self.WPS_layers)-1:
                x = WPS_activation(x)
        return x

class neural_network_AE(nn.Module):
    def __init__(self, input_size=300, output_size=4, 
                 nb_layers=3, norm_type='LayerNorm',
                 activation_type='ReLU'):
        super(neural_network_AE, self).__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.nb_layers = int(nb_layers)
        self.norm_type = norm_type
        self.activation_type = activation_type
        layers = np.arange(1, self.nb_layers + 1)
        neurons_encoder = np.flip(np.linspace(self.output_size, 
                                              self.input_size, 
                                              self.nb_layers + 1, 
                                              dtype=int))
        neurons_decoder = np.linspace(self.output_size, 
                                      self.input_size, 
                                      self.nb_layers + 1,
                                      dtype=int)

        # Create lists to store encoder and decoder layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_norm_layers = nn.ModuleList()
        self.encoder_activations = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.decoder_norm_layers = nn.ModuleList()
        self.decoder_activations = nn.ModuleList()

        # Create encoder layers and append them to the list                                           
        for i1, j1 in enumerate(layers):
            
            #norm layers
            if self.norm_type.lower() == 'BatchNorm'.lower():
                norm_layer = nn.BatchNorm1d(neurons_encoder[i1])
            elif self.norm_type.lower() == 'LayerNorm'.lower():
                norm_layer = nn.LayerNorm(neurons_encoder[i1])
            else:
                norm_layer = nn.Identity()
            self.encoder_norm_layers.append(norm_layer)

            #layers
            encoder_layer = nn.Linear(in_features=neurons_encoder[i1], 
                                      out_features=neurons_encoder[i1 + 1])
            self.encoder_layers.append(encoder_layer)

            #activation of layer
            if self.activation_type.lower() == 'ReLU'.lower():
                encoder_activation = nn.ReLU()
            elif self.activation_type.lower() == 'sigmoid'.lower():
                encoder_activation = nn.Sigmoid()
            elif self.activation_type.lower() == 'tanh'.lower():
                encoder_activation = nn.Tanh()
            else:
                encoder_activation = nn.Identity()
            self.encoder_activations.append(encoder_activation)
            

        # Create decoder layers and append them to the list
        for i1, j1 in enumerate(reversed(layers)):

            #norm layers
            if self.norm_type.lower() == 'BatchNorm'.lower():
                norm_layer = nn.BatchNorm1d(neurons_decoder[i1])
            elif self.norm_type.lower() == 'LayerNorm'.lower():
                norm_layer = nn.LayerNorm(neurons_decoder[i1])
            else:
                norm_layer = nn.Identity()
            self.decoder_norm_layers.append(norm_layer)

            #layers
            decoder_layer = nn.Linear(in_features=neurons_decoder[i1], 
                                      out_features=neurons_decoder[i1 + 1])
            self.decoder_layers.append(decoder_layer)

            #activation of layer
            if self.activation_type.lower() == 'ReLU'.lower():
                decoder_activation = nn.ReLU()
            elif self.activation_type.lower() == 'sigmoid'.lower():
                decoder_activation = nn.Sigmoid()
            elif self.activation_type.lower() == 'tanh'.lower():
                decoder_activation = nn.Tanh()
            else:
                decoder_activation = nn.Identity()
            self.decoder_activations.append(decoder_activation)

    def forward(self, x):
        # Encoder
        encoded = x
        for l1, (encoder_norm_layer,encoder_layer,encoder_activation) in enumerate(
            zip(self.encoder_norm_layers,
                self.encoder_layers,
                self.encoder_activations)):
            encoded = encoder_norm_layer(encoded)
            encoded = encoder_layer(encoded)
            if l1 == len(self.encoder_layers)-1:  #not applying normalization to output
                raw_LatentAE = encoded.clone()
            encoded = encoder_activation(encoded)
        # Decoder
        decoded = encoded
        for l1, (decoder_norm_layer,decoder_layer,decoder_activation) in enumerate(
            zip(self.decoder_norm_layers,
                self.decoder_layers,
                self.decoder_activations)):
            decoded = decoder_norm_layer(decoded)
            decoded = decoder_layer(decoded)
            if l1 < len(self.decoder_layers)-1: #now applying normalization to output
                decoded = decoder_activation(decoded)

        return decoded, raw_LatentAE, encoded

def WPS_predictor_single(St,Ut,h,Mach,Re,alpha,Uinf,x_over_c,net_WPS,net_AE,means,stds):
    """
    Output the WPS spectrum given the Strouhal number (St) and tangential velocity profile (h,Ut) as input

    Parameters:
    - St (np array): Desired Strouhal number interval for PSD prediction
    - Ut (np array): Tangential velocity
    - Uinf (float): Freestream (simulation inlet) velocity
    - x_over_c (float) : streamwise location
    - Mach (float): Mach number
    - Re (float): Reynolds number
    - alpha (float): angle of attack
    - net_WPS (class): Trained wall PSD predictor ANN 
    - net_AE (class): Trained Autoencoder

    Returns:
    - Predicted_WPS (np array): Predicted wall PSD
    """
    # Read the mean and standard deviations of the features and labels for normalisation
    mean_features = means[0]
    mean_labels = means[-1]
    std_features = stds[0]
    std_labels = stds[-1]
    # Construct the input id map
    idx_features = {'id': 0, 'St_interp': 1, 'Mach': 256, 'Reynolds': 257, 'alpha': 258, 'Uinf': 259, 'x/c': 260, 'h': 261, 'Ut': 411}
    idx_h = idx_features['h']
    idx_Ut = idx_features['Ut']
    # Generate the test features dataframe for input to AE
    test_features = pd.DataFrame({'id': ['H08_001'],
                        'St_interp': [np.array(St)],
                        'Mach': [np.array(Mach)],
                        'Reynolds': [np.array(Re)],
                        'alpha': [np.array(alpha)],
                        'Uinf': [np.array(Uinf)],
                        'x/c':[np.array([x_over_c])],
                        'h':[np.array(h)],
                        'Ut':[np.array(Ut)]})
    
    test_features = normalize_validation(test_features,mean_features,std_features) #normalise the test features using mean and std dev
    data_array = torch.tensor(df_to_array(test_features),dtype=torch.float32) # Convert to numpy array
    # Prepare the input dataset for the Encoder
    data_AE = data_array[:,idx_h:idx_Ut+(idx_Ut-idx_h)]
    # Use Encoder to obtain the latent space param
    _ , latent_AE, _ = net_AE(data_AE) #Obtain the latent space parameters
    print('latent_AE',latent_AE)
    # Prepare the input dataset for the WPS predictor
    idlen = 1 #removing id element in features
    data_WPS = torch.zeros((data_array.shape[0],
                            data_array.shape[-1]-idlen-2*(idx_Ut-idx_h)+latent_AE.shape[-1]))
    data_WPS[:,:idx_h-idlen] = data_array[:,idlen:idx_h].clone().detach() # Assign the flow parameters from data to data_WPS
    data_WPS[:,idx_h-idlen:idx_h-idlen+latent_AE.shape[-1]] = latent_AE.clone().detach() # Assign the latent space param to data_WPS

    # Obtaining the cuda parameters
    data_WPS = data_WPS.to(device='cpu')
    
    # Forward pass through the WPS predictor ANN
    predicted_WPS = net_WPS(data_WPS)
    predicted_WPS_dim =  predicted_WPS.cpu().detach().numpy()*std_labels['log10phipp_interp'].values + mean_labels['log10phipp_interp'].values
    predicted_WPS_dim = np.squeeze(predicted_WPS_dim) #reduce output to 1d array
    return predicted_WPS_dim

def normalize_validation(dfdata,mean,std):
    #For each key in the dataframe, normalize the data associated to the key
    norm_data = dfdata.copy()
    if len(dfdata.shape) > 1:
        for key in norm_data.keys():
            # print(key,flush=True)
            if key != 'id':
                if std[key].values > 1e-7:
                    norm_data[key] = ((np.array(norm_data[key].values))-mean[key].values)/std[key].values
    else:
        if std[dfdata.name].values > 1e-7:
            for key in norm_data.keys():
                norm_data[key] = ((np.array(norm_data[key]))-mean[dfdata.name].values)/std[dfdata.name].values
    return norm_data

def normalize_training(dfdata):
    if len(dfdata.shape) > 1: #features
        dictmean = {}
        dictstd = {}
        for key in dfdata.keys():
            # print(key,flush=True)
            if key != 'id':
                aux = np.array(dfdata[key].values.tolist())
                dictmean[key] = np.mean(aux)
                dictstd[key] = np.std(aux)
        dfmean = pd.DataFrame(dictmean,index=[0])
        dfstd = pd.DataFrame(dictstd,index=[0])
            
        norm_data = dfdata.copy()
        for key in dfmean.keys():
            # print(key,flush=True)
            if dfstd[key].values > 1e-7:
                norm_data[key] = ((np.array(norm_data[key].values))-dfmean[key].values)/dfstd[key].values
    else: #labels
        dictmean = {}
        dictstd = {}
        aux = np.array(dfdata.values.tolist())
        aux = aux.flatten()
        dictmean[dfdata.name] = np.mean(aux)
        dictstd[dfdata.name] = np.std(aux)
        dfmean = pd.DataFrame(dictmean,index=[0])
        dfstd = pd.DataFrame(dictstd,index=[0])
        norm_data = dfdata.copy()
        if dfstd[dfdata.name].values > 1e-7:
            for key in norm_data.keys():
                norm_data[key] = ((np.array(norm_data[key]))-dfmean[dfdata.name].values)/dfstd[dfdata.name].values
    return norm_data, dfmean, dfstd

def df_to_array(dfdata):
    if len(dfdata.shape)>1:
        data_list = []
        for i in range(dfdata.shape[0]):
            linedata = []
            for key in dfdata.keys():
                if key == 'id':
                    #convert case Cxx_xxx int xx.xxx (number and decimal)
                    linedata.append(eval(dfdata[key].iloc[i][1:].replace('_','.').replace('J','1')))
                else:
                    linedata += dfdata[key].iloc[i].tolist()
            try:
                linedata = np.concatenate(linedata).ravel().tolist()
            except:
                pass
            data_list.append(linedata)
        
    else:
        data_list = []
        for i in range(dfdata.shape[0]):
            if dfdata.iloc[i].dtype == float:
                data_list.append(dfdata.iloc[i])
            else:
                data_list += dfdata.iloc[i].tolist()

    arraydata = np.array(data_list)
    return arraydata
