# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# Data.py

import os
import numpy as np
import util
import copy
import gc
import collections
import cv2
from keras.preprocessing.image import ImageDataGenerator
from random import randint, seed
import ipdb
import random

class Dataset():

    def __init__(self, config):
        """ Set Dataset parameters.

        Args:
        config: dict, session configuration parameters

        """

        self.config = config
        self.path = self.config['dataset']['path']
        self.file_paths = {'train': [], 'val': []}
        self.soundfields = {'train': [], 'val': []}
        self.batch_size = self.config['training']['batch_size']
        self.xSamples = self.config['dataset']['xSamples']
        self.ySamples = self.config['dataset']['ySamples']
        self.freq = util.get_frequencies()
        self.num_freq = len(self.freq)
        self.factor = self.config['dataset']['factor']

    def load_dataset(self):
        """ Load Dataset. """

        print('\nLoading Simulated Sound Field Dataset...')
        
        full_dataset_path = '/nas/home/fronchini/complex-sound-field/dataset'
        filenames = [filename for filename in os.listdir(full_dataset_path) if filename.endswith('.mat')]
        #train_set, val_set = train_test_split(filenames, test_size=0.25, random_state=42)
        percentage = 25

        # Calculate the number of elements to select
        num_elements_to_select = int(len(filenames) * (percentage / 100.0))

        # Randomly select elements from the original list
        val_set = random.sample(filenames, num_elements_to_select)

        # Create another list with the remaining elements
        train_set = [item for item in filenames if item not in val_set]
        
        for set in ['train', 'val']:
            if set == 'train':
                #current_directory = os.path.join(full_dataset_path, 'simulated_soundfields', set)
                soundfields, file_paths = self.load_directory(full_dataset_path, train_set)
            else: 
                soundfields, file_paths = self.load_directory(full_dataset_path, val_set)

            #soundfields, file_paths = self.load_directory(current_directory)
            
            self.file_paths[set] = file_paths
            self.soundfields[set] = soundfields

        return self

    def load_directory(self, directory_path, filenames):
        """ Load directory.

	    Args:
	    directory_path: string

	    Returns: list, list

	    """
        #filenames = [filename for filename in os.listdir(directory_path) if filename.endswith('.mat')]
        #directory_path

        file_paths = []
        soundfields = []

        for filename in filenames:

            filepath = os.path.join(directory_path, filename)
            file_paths.append(filepath)
            soundfield = util.load_soundfield(filepath, self.freq)
            soundfields.append(soundfield)
        return soundfields, file_paths


    def get_random_batch_generator(self, set):
        """ Generates batches of set data.

	    Args:
	    set: string

	    Returns: generator

	    """


        if set not in ['train', 'val']:
            raise ValueError("Argument SET must be either 'train' or 'val'")

        # Create datagen
        datagen = DataGenerator(ImageDataGenerator, self.config)


        if set == 'train':
            return datagen.flow(x=np.asarray(self.soundfields['train']), batch_size=self.batch_size)
        else:
            return datagen.flow(x=np.asarray(self.soundfields['val']), batch_size=self.batch_size)

class DataGenerator(ImageDataGenerator):

    def __init__(self, ImageDataGenerator, config):
        """ Set DataGenerator parameters.

        Args:
        ImageDataGenerator: ImageDataGenerator object from Keras
        config: dict

        """
        self.config = config
        ImageDataGenerator.__init__(self, ImageDataGenerator)
        self.xSamples = self.config['dataset']['xSamples']
        self.ySamples = self.config['dataset']['ySamples']
        self.num_freq = len(util.get_frequencies())
        self.factor = self.config['dataset']['factor']
        self.mask_generator = MaskGenerator(int(self.xSamples/self.factor), int(self.ySamples/self.factor), self.num_freq, rand_seed=7)

    def flow(self, x, *args, **kwargs):
        """ Generates batches of data.

	    Args:
	    x: np.ndarray

	    Yield: [np.ndarray, np.ndarray], np.ndarray

	    """
        while True:


            # Get soundfield samples
            sf_gt = next(super().flow(x, *args, **kwargs))
            initial_sf = copy.deepcopy(sf_gt)

            # Get mask samples
            mask = np.stack([self.mask_generator.sample() for _ in range(sf_gt.shape[0])], axis=0)

            # preprocessing
            irregular_sf, mask = util.preprocessing(self.factor, initial_sf, mask)

            # Scale ground truth sound field
            sf_gt = util.scale(sf_gt)

            gc.collect()
            yield [irregular_sf, mask], sf_gt

class MaskGenerator():

    def __init__(self, height, width, channels=40, num_mics=None, rand_seed=None):
        """ Set MaskGenerator parameters.

        Args:
        height: int
        width: int
        channels: int (default: 40)
        rand_seed: int (default: None)

        """

        self.height = height
        self.width = width
        self.channels = channels
        self.num_mics = num_mics

        # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)
            np.random.seed(rand_seed)

    def _generate_mask(self, num_mics):
        """Generates a random irregular mask.

        Returns: np.ndarray

        """   
        self.num_mics = num_mics
        
        if self.num_mics:
            mask_slice = np.zeros((self.height*self.width), np.uint8)

            num_holes = self.height*self.width - self.num_mics

            index_holes = np.random.choice(int(self.height*self.width), size=num_holes, replace=False)

            mask_slice[index_holes] = 1

            mask_slice.resize(self.height, self.width, 1)


        else:
            mask_slice = np.zeros((self.height, self.width, 1), np.uint8)

            size = 1

            # Draw random lines
            for _ in range(randint(1, 20)):
                x1, x2 = randint(1, self.width), randint(1, self.width)
                y1, y2 = randint(1, self.height), randint(1, self.height)
                thickness = 1
                cv2.line(mask_slice,(x1,y1),(x2,y2),(1,1,1),thickness)

            # Draw random circles
            for _ in range(randint(1, 20)):
                x1, y1 = randint(1, self.width), randint(1, self.height)
                radius = randint(1, size)
                cv2.circle(mask_slice,(x1,y1),radius,(1,1,1), -1)

            # Draw random ellipses
            for _ in range(randint(1, 20)):
                x1, y1 = randint(1, self.width), randint(1, self.height)
                s1, s2 = randint(1, self.width), randint(1, self.height)
                a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
                thickness = 1
                cv2.ellipse(mask_slice, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)

        mask_slice = 1-mask_slice
        mask = np.repeat(mask_slice, self.channels, axis=2)
        
        # fig = plt.figure(figsize=(30, 10))
        # plt.subplot(131)
        # plt.imshow(mask[1], aspect='auto')
        # plt.tight_layout()
        # plt.save("/nas/home/fronchini/sound-field-neural-network/mask.png")
        
        return mask

    def sample(self, random_seed=None):
        """Retrieve a random mask

        Returns: np.ndarray

        """

        if random_seed:
            seed(random_seed)
        else:
            # list with numbe of microphones we want to test
            num_mics_list = [5, 15, 35, 55]
            # pick one random value to generate masks
            num_mics = random.choice(num_mics_list)
            #print(f"Mask generated using n_mics: {num_mics}")
            
            return self._generate_mask(num_mics)

