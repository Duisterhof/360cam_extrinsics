
import os
import sys

# The path of the current Python script.
_CURRENT_PATH       = os.path.dirname(os.path.realpath(__file__))
_TOP_PATH           = os.path.join(_CURRENT_PATH, '..', '..')
_DATA_PIPELINE_PATH = os.path.join(_TOP_PATH, 'data_collection', 'ord_data_pipeline_rework', 'src')
sys.path.insert( 0, _DATA_PIPELINE_PATH )
sys.path.insert( 0, _TOP_PATH )
for i, p in enumerate(sys.path):
    print(f'{i}: {p}')
    
import argparse

# Local package.
from data_collection.preprocessing import Fisheye_Preprocessor_v2 as Fisheye_Preprocessor

def get_preprocessing_args():
    parser = argparse.ArgumentParser(description='preprocessing pipeline')
    
    parser.add_argument('--manifest-path', type=str, default='X:\\ProcessedData', 
                        help='Path to the manifest file. ')

    parser.add_argument('--frame-graph-path', type=str, 
                        help='The path to the frame graph JSON file.')
                        
    parser.add_argument('--metadata-path', type=str, 
                        help='The path to the metadata JSON file.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_preprocessing_args()
    preprocessor = Fisheye_Preprocessor(args)
    #print([method for method in dir(preprocessor) if callable(getattr(preprocessor, method))])
    preprocessor.preprocess_requested_imgs()
    