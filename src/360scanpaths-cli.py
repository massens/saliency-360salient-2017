


# (c) Copyright 2017 Marc Assens. All Rights Reserved.

import click
import pathnet


__author__ = "Marc Assens"
__version__ = "0.1"

@click.command()
@click.option('--img_path', prompt='Path with folder Images containing the images')
@click.option('--ids', prompt='Ids of the images without spaces and separed by commas')
@click.option('--out_path', prompt='Path where the results will be saved')
def predict(img_path, ids, out_path):
    """ Predicts multiple images and saves them in .mat format
    on an output path
    
    \b
    Param:
        img_path :  path where there is a folder named 'Images'
                    that contains the images with format P%d.jpg 
        ids      :  list with image ids
        out_path :  path where the .mat files will be saved

    \b
    i.e.:
        img_path =  '/root/sharedfolder/360Salient/'
        ids      =  [29, 31]
        out_path =  '/root/sharedfolder/360Salient/results/'

    \b
    example command:
        ./predict-scanpath-360 /root/sharedfolder/360Salient/ 29,31 /root/sharedfolder/360Salient/results/
    """

    id_list = map(int, ids.split(','))

    print('\n\n###########################')
    print('Starting program')
    pathnet.predict_and_save(img_path, id_list, out_path)

if __name__ == '__main__':
    predict()
