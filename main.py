import torch
import os
import pickle
from model import init_generator

if __name__ == '__main__':
    try: 
        _dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _dir = os.getcwd()
    PATH_TO_MODEL = os.path.join(_dir, 'model_dict.pkl')
    PATH_TO_DATA = os.path.join(_dir, 'fake_data.pkl')  
    generator_regular, generator_cris = init_generator(pretrain=False)
    print("Generator loaded. Generate fake data.")
    with torch.no_grad():
        condition = torch.ones([200, 1])
        if condition[0] == 1:
            fake_data = generator_cris(batch_size=200, n_lags=20, device='cpu')
        else:
            fake_data = generator_regular(batch_size=200, n_lags=20, device='cpu')
    with open(PATH_TO_DATA, 'wb') as file: 
        pickle.dump(fake_data, file) 