from .aircraft_dataset import AircraftDataset
from .bird_dataset import BirdDataset
from .car_dataset import CarDataset
from .dog_dataset import DogDataset
from .inat2017_dataset import INat2017
from .nabirds_dataset import NABirds

def get_trainval_datasets(tag, data_path, resize):
    if tag == 'aircraft':
        return AircraftDataset(phase='train', data_path=data_path, resize=resize), AircraftDataset(phase='val', data_path=data_path, resize=resize)
    elif tag == 'bird':
        return BirdDataset(phase='train', data_path=data_path, resize=resize), BirdDataset(phase='val', data_path=data_path, resize=resize)
    elif tag == 'car':
        return CarDataset(phase='train', data_path=data_path, resize=resize), CarDataset(phase='val', data_path=data_path, resize=resize)
    elif tag == 'dog':
        return DogDataset(phase='train', data_path=data_path, resize=resize), DogDataset(phase='val', data_path=data_path, resize=resize)
    elif tag == 'inat':
        return INat2017(root=data_path, split='train', resize=resize), INat2017(root=data_path, split='val', resize=resize)
    elif tag == 'nabirds':
        return NABirds(root=data_path, phase='train', resize=resize), NABirds(root=data_path, phase='val', resize=resize)
    else:
        raise ValueError('Unsupported Tag {}'.format(tag))