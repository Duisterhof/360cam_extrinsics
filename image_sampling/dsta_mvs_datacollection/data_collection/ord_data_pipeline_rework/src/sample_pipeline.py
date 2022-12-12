from mapping.ExpoController import ExpoController
from sampling.roadmap_path_sample import roadmap_path_sample
from collection.collect_images import collect_data_files
from settings import get_args

if __name__ == '__main__':
    args = get_args()
    
    if args.mapping:
        controller = ExpoController(args)
        controller.mapping()
    
    # Generate a graph by sampling the nodes end edges 
    roadmap_path_sample(args)

    # read data from AirSim    
    if args.data_collection:
        collect_data_files(args)
