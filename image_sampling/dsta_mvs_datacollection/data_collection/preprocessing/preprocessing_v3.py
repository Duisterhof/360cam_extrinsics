
import copy
import cv2
import datetime
import glob
import json
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import os
import pandas as pd
import shutil
from scipy.spatial.transform import Rotation as R
import time

# For FTensor.
import torch

# Local package.
from ..image_sampler import SAMPLERS
from ..image_sampler import make_object as make_sampler

from ..mvs_utils.camera_models import CAMERA_MODELS
from ..mvs_utils.camera_models import make_object as make_camera_model
from ..mvs_utils.ftensor import FTensor
from ..mvs_utils.frame_io import parse_orientation
from ..mvs_utils.image_io import ( read_image, read_compressed_float, write_compressed_float )
from ..mvs_utils.metadata_reader import MetadataReader
from ..mvs_utils.shape_struct import ShapeStruct

from ..multiproc.process_pool import ( ReplicatedArgument, PoolWithLogger)
from ..multiproc.shared_memory_image import ( 
    shm_size_from_img_shape, 
    pass_forward, c4_uint8_as_float, float_as_c4_uint8, 
    SharedMemoryImage )
from ..multiproc.utils import ( compose_name_from_process_name, job_print_info )

# Dummy image reader.
def read_forward_filename(fn):
    return fn

IMAGE_READERS = {
    'rgb': read_image,
    'compressed_float': read_compressed_float,
    'copy_filename': read_forward_filename, 
}

def write_image_as_copy(out_fn, img):
    shutil.copyfile(img, out_fn)

IMAGE_WRITERS = {
    'rgb': cv2.imwrite,
    'compressed_float': write_compressed_float,
    'copy_filename': write_image_as_copy,
}

# === Multiprocessing functions. ===

def image_io_job_initializer(logger_name, log_queue, shm_name, image_shape, reader_writer_type, flag_float):
    # Use global variables to transfer variables to the job process.
    global P_JOB_LOGGER
    
    # The logger.
    P_JOB_LOGGER = PoolWithLogger.job_prepare_logger(logger_name, log_queue)
    
    # print(P_JOB_LOGGER.handlers)

    channel_depth = 4 if flag_float else 1
    processor_in  = float_as_c4_uint8 if flag_float else pass_forward
    processor_out = c4_uint8_as_float if flag_float else pass_forward

    # The shared memory.
    global P_JOB_SHM_IMG
    P_JOB_SHM_IMG = SharedMemoryImage( shm_name, image_shape, channel_depth, processor_in, processor_out )
    P_JOB_SHM_IMG.initialize()

    # The reader/writer.
    global IMAGE_READERS, IMAGE_WRITERS
    global P_JOB_IMAGE_READER, P_JOB_IMAGE_WRITER
    P_JOB_IMAGE_READER = IMAGE_READERS[reader_writer_type]
    P_JOB_IMAGE_WRITER = IMAGE_WRITERS[reader_writer_type]

def read_image_job(idx, img_root_dir, num_str, suffix):
    global P_JOB_LOGGER, P_JOB_SHM_IMG, P_JOB_IMAGE_READER
    
    proc_name = multiprocessing.current_process().name
    proc_name = compose_name_from_process_name('P', proc_name)

    # job_print_info(P_JOB_LOGGER, proc_name, f'Read {fn}. ')

    # Read the image.
    fn = os.path.join(img_root_dir, f'{num_str}_{suffix}.png')
    img = P_JOB_IMAGE_READER(fn)

    # # Get the image from the shared memory.
    # buffer = P_JOB_SHM_IMG[idx]

    # # Copy the image to the shared memory.
    # buffer[:, :, ...] = img

    # Write the image to the shared memory.
    P_JOB_SHM_IMG[idx] = img

    return 0

def write_image_job(idx, out_dir, num_str, suffix):
    '''
    idx: The index of the job.
    write_index: The index of the output filename.
    '''
    global P_JOB_LOGGER, P_JOB_SHM_IMG, P_JOB_IMAGE_WRITER
    
    proc_name = multiprocessing.current_process().name
    proc_name = compose_name_from_process_name('P', proc_name)

    # Compose the output filename.
    out_fn = os.path.join( out_dir, f'{num_str}_{suffix}.png' )
    # job_print_info(P_JOB_LOGGER, proc_name, f'Write {out_fn}. ')

    # Get the image from the shared memory.
    sampled = P_JOB_SHM_IMG[idx]
    P_JOB_IMAGE_WRITER( out_fn, sampled )

    return 0

class Fisheye_Preprocessor(object):

    def __init__(self, args):
        print("Reading Manifest File...")
        with open(args.manifest_path) as manifest_file:
            self.manifest = json.load(manifest_file)
 
        self.total_frames = 0
        self.debug_n = args.debug_n # Use non-postive to disable.

        # Parse the camera models.
        self.camera_models = None
        self.parse_manifest_for_camera_models()

        # Parse the samplers.
        self.sampler_dicts = None
        self.parse_manifest_for_samplers()

        # Parse the input environments. Create output directories. Copy some configuration files.
        self.parse_manifest_for_envs_and_trajs(args)

        self.target_batch_size = args.batch_size
        self.gpu_mem_size_byte = args.gpu_mem_mb * 1024 * 1024
        self.batch_size = self.target_batch_size
        self.image_io_job_num = args.image_io_job_num
        self.shm_in = None
        self.shm_img_in = None
        self.shm_out = None
        self.shm_img_out = None

    @staticmethod
    def parse_dir_elements(env_dir):
        # Assuming env_dir is of the form
        # /data/AbandonedCable_Windows/P001
        # Refer to https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html

        all_parts = []
        while True:
            parts = os.path.split(env_dir)
            if parts[0] == env_dir:  # sentinel for absolute paths
                all_parts.insert(0, env_dir)
                break
            elif parts[1] == env_dir: # sentinel for relative paths
                all_parts.insert(0, parts[1])
                break
            else:
                env_dir = parts[0]
                all_parts.insert(0, parts[1])
        return all_parts

    @staticmethod
    def search_trajectory_names(env_dir):
        # Assuming that there is a "cam_paths.csv" file loated under every trajectory directory.
        # Assuming that there is only one level of subdirectories under env_dir, meaning that
        # the directory structure is <environment dir>/<trajectory dir>/cam_paths.csv.
        cam_paths_files = glob.glob( os.path.join(env_dir, '**', 'cam_paths.csv'), recursive=True )
        assert len(cam_paths_files) > 0, f'No cam_paths.csv file found under {env_dir}. '
        return [ os.path.split( os.path.dirname(f) )[-1] for f in cam_paths_files ]

    def parse_manifest_for_camera_models(self):
        self.camera_models = dict()
        for key, value in self.manifest["camera_models"].items():
            self.camera_models[key] = make_camera_model(CAMERA_MODELS, value)

    def parse_manifest_for_samplers(self):
        self.sampler_dicts = list()
        for sampler_dict in self.manifest["samplers"]:
            # Make a copy of sampler_dict.
            sampler_dict_copy = copy.deepcopy(sampler_dict)
            
            # Make a copy of the sampler object argument.
            sampler_spec = copy.deepcopy( sampler_dict_copy['sampler'] )
            sampler_dict_copy.pop('sampler')
            
            # Convert orientatioin to rotation matrix.
            q = sampler_spec["orientation"]
            R_raw_fisheye = FTensor( parse_orientation(q), f0=q['f0'], f1=q['f1'], rotation=True ).to(dtype=torch.float32)
            # sampler_spec["R_raw_fisheye"] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
            sampler_spec["R_raw_fisheye"] = R_raw_fisheye
            sampler_spec.pop("orientation")
            
            # Get the camera model.
            sampler_spec["camera_model"] = self.camera_models[ sampler_spec["cam_model_key"] ]
            sampler_spec.pop("cam_model_key")
            
            # Create the the sampler.
            sampler = make_sampler(SAMPLERS, sampler_spec)
            sampler.device = 'cuda'

            # Re-populate the 'sampler' key of sampler_dict_copy.
            sampler_dict_copy['sampler'] = sampler

            self.sampler_dicts.append( sampler_dict_copy)

        self.sampler_out_dirs = \
            sorted( list( set( [ sampler_dict['output_dir'] for sampler_dict in self.sampler_dicts ] ) ) )

        self.sampler_csv_columns = \
            [ sampler_dict['table_header'] for sampler_dict in self.sampler_dicts ]

    def create_sampler_output_dirs(self, env_traj_dir):
        for sampler_out_dir in self.sampler_out_dirs:
            os.makedirs( os.path.join(env_traj_dir, sampler_out_dir), exist_ok=True )

    def parse_manifest_for_envs_and_trajs(self, args):
        # Extract important paths and the datasets to process from the manifest JSON.
        self.preproc_name = self.manifest["dataset_name"] + "_" + datetime.date.today().strftime("%m%d%Y")
        self.storage_path = self.manifest["storage_path"]
        self.raw_env_dirs = self.manifest["raw_environment_dirs"]

        # Create the output directory.
        self.processed_dataset_path = os.path.join(self.storage_path, self.preproc_name)
        if not os.path.exists(self.processed_dataset_path):
            os.makedirs(self.processed_dataset_path)

        # NOTE: This should happen for every trajectory.
        shutil.copy( args.manifest_path, os.path.join(self.processed_dataset_path,"manifest.json") )

        # environment - trajectory - number string.
        # This is a list of discts.
        self.env_traj_num = list()

        # For each dataset in the manifest.json.
        # E.g., training and testing.
        self.total_frames = 0
        for env_dir in self.raw_env_dirs:
            # Figure out the environment name.
            env_name = Fisheye_Preprocessor.parse_dir_elements( env_dir )[-1]

            # Create the environment output directory
            env_out_path = os.path.join( self.processed_dataset_path, env_name )
            if not os.path.isdir(env_out_path):
                os.makedirs(env_out_path)
            
            # Repurpose a metadata reader to create the target directory structure.
            metadata_reader = MetadataReader( env_dir)

            # TODO: Should every pre_set (envrionment) have its own metadata and frame_graph?
            # TODO: How to deal with multiple trajectories in a single environment?
            metadata_reader.read_metadata_and_initialize_dirs(args.metadata_path, args.frame_graph_path, create_dirs=False)

            # Dump important metadata for each dataset at the subset level
            # TODO: Similar to the above. metadata and frame graph might be directly associated with an environment.
            shutil.copy( args.metadata_path, os.path.join(env_dir, "metadata.json") )
            shutil.copy( args.frame_graph_path, os.path.join(env_dir, "frame_graph.json") )

            # Initialize the first level of the cascade dictionary.
            self.env_traj_num.append( {
                'name': env_name,
                'dir': env_dir,
                'metadata_reader': metadata_reader,
                'trajectories': list() # List of dicts.
            } )

            # Search for all the trajectories.
            traj_names = Fisheye_Preprocessor.search_trajectory_names( env_dir )

            for traj_name in traj_names:
                # Create the trajectory output directory.
                traj_out_path = os.path.join( env_out_path, traj_name )
                if not os.path.isdir( traj_out_path ):
                    os.makedirs( traj_out_path )

                # Create the output directories for the samplers.
                self.create_sampler_output_dirs( traj_out_path )

                #Load the image names and paths from the cam_paths.csv file
                # It's a string array like:
                # [ [  'cam0'   'cam1'   'cam2' ]
                #   ['000000' '000000' '000000' ]
                #   ['000001' '000001' '000001' ]
                #   ['000002' '000002' '000002' ] ]
                img_index_arr = np.genfromtxt(
                            os.path.join(env_dir, traj_name, 'cam_paths.csv'), delimiter=',', dtype=str )
                self.total_frames += img_index_arr.shape[0] - 1

                # Create an index from camera name to a list of partial absolute paths to the images that need processing. 
                # The missing parts of the image path include the ImageType (CubeDistance or CubeScene) and 
                # the file type (.png or .npy).                        
                img_index_arr = img_index_arr[1:self.debug_n+1, 0] if self.debug_n > 0 else img_index_arr[1:, 0]
                self.env_traj_num[-1]['trajectories'].append( {
                    'name': traj_name, 
                    'index_strings': img_index_arr } )

        print("----------------------")
        for env_dict in self.env_traj_num:
            print(f'{env_dict["name"]}: ')
            for traj_dict in env_dict['trajectories']:
                print(f'    { traj_dict["name"] }: {traj_dict["index_strings"].shape[0]}')
        print("----------------------")
        print(f'Total number of envs: {len(self.env_traj_num)}')
        print(f'self.total_frames = {self.total_frames}')
        print(f'self.debug_n = {self.debug_n}')

    def generate_file_list_for_trajectory( self, out_dir, img_num_str_list ):
        # Save the table header and columns separately.
        data_frame_dict = dict()
        for sampler_dict in self.sampler_dicts:
            out_sub_dir = sampler_dict['output_dir']
            out_img_suffix = sampler_dict['output_image_suffix']

            data_frame_dict[ sampler_dict['table_header'] ] = \
                [ os.path.join( out_sub_dir, f'{img_num_str}_{out_img_suffix}.png' ) 
                  for img_num_str in img_num_str_list ]

        # Create a pandas dataframe.
        df = pd.DataFrame( data=data_frame_dict )

        # Save the list as a csv file.
        df.to_csv( os.path.join(out_dir, 'default_file_list.csv'),
                   columns=self.sampler_csv_columns,
                   index=False )

    @staticmethod
    def get_raw_and_output_shapes(env_dir, traj_name, sampler_dict, img_num_str_list):
        # Figure out the shapes of the raw and output images.
        img = IMAGE_READERS[ sampler_dict['image_reader'] ](
            os.path.join( 
                env_dir, 
                traj_name, 
                sampler_dict['raw_camera'], 
                f'{img_num_str_list[0]}_{sampler_dict["raw_image_suffix"]}.png' ) 
        )

        raw_image_shape = img.shape
        flag_float = img.dtype == np.float32

        if len(raw_image_shape) == 3:
            out_image_shape = ( *sampler_dict['sampler'].shape[:2], raw_image_shape[2] )
        else:
            out_image_shape = sampler_dict['sampler'].shape[:2]

        return raw_image_shape, out_image_shape, flag_float

    def upate_batch_size(self, raw_image_shape, out_image_shape, flag_float):
        channel_depth = 4 if flag_float else 1
        
        self.batch_size = self.target_batch_size

        raw_image_mem = np.prod( raw_image_shape ) * channel_depth
        out_image_mem = np.prod( out_image_shape ) * channel_depth
        image_mem = raw_image_mem + out_image_mem
        if self.gpu_mem_size_byte / image_mem < self.batch_size:
            self.batch_size = int( np.floor( self.gpu_mem_size_byte / image_mem ) )
            print(f'Batch size reduced to {self.batch_size} from {self.target_batch_size} to fit in targe GPU memory limit {self.gpu_mem_size_byte / 1024**2 }MB. ')

    def prepare_shared_memory(self, raw_image_shape, out_image_shape, flag_float):
        channel_depth = 4 if flag_float else 1
        processor_in  = float_as_c4_uint8 if flag_float else pass_forward
        processor_out = c4_uint8_as_float if flag_float else pass_forward

        # Prepare the shared memory for inputs.
        shm_size_byte_in = shm_size_from_img_shape( raw_image_shape, channel_depth, self.batch_size )
        self.shm_in = shared_memory.SharedMemory(create=True, size=shm_size_byte_in)

        # Get a SharedMemoryImage object for inputs.
        self.shm_img_in = SharedMemoryImage( self.shm_in.name, raw_image_shape, channel_depth, processor_in, processor_out)
        self.shm_img_in.initialize()

        # Prepare the shared memory for outputs.
        shm_size_byte_out = shm_size_from_img_shape( out_image_shape, channel_depth, self.batch_size )
        self.shm_out = shared_memory.SharedMemory(create=True, size=shm_size_byte_out)

        # Get a SharedMemoryImage object for outputs.
        self.shm_img_out = SharedMemoryImage( self.shm_out.name, out_image_shape, channel_depth, processor_in, processor_out )
        self.shm_img_out.initialize()

    def cleanup_shared_memory(self):
        self.shm_img_out.finalize()
        self.shm_out.close()
        self.shm_out.unlink()

        self.shm_img_in.finalize()
        self.shm_in.close()
        self.shm_in.unlink()

    def process_trajectory(self, env_dir, env_name, traj_name, img_num_str_list):
        # Initialize the list of flags for tracking the event of mask generation.
        mask_generation_flags = [ False for _ in self.sampler_dicts ]

        for sampler_index, sampler_dict in enumerate(self.sampler_dicts):
            # sampler_dict: {
            #     "description": "cam0 fisheye RGB sampler",
            #     "table_header": "cam0_rgb_fisheye",
            #     "raw_camera": "cam0",
            #     "raw_image_suffix": "CubeScene",
            #     "output_dir": "cam0",
            #     "output_image_suffix": "CubeScene",
            #     "interpolation": "linear",
            #     "write_valid_mask": true,
            #     "mvs_main_cam_model_for_cam": true,
            #     "sampler": sampler
            # }

            print()
            print(f'>>> {env_name}, {traj_name}, {sampler_dict["description"]}')

            # if sampler_dict['description'] == 'rig rotated panorama distance sampler':
            #     import ipdb; ipdb.set_trace()
            #     print('Debug')

            # Get the raw and output image shapes.
            raw_image_shape, out_image_shape, flag_float = \
                self.get_raw_and_output_shapes(env_dir, traj_name, sampler_dict, img_num_str_list)
            
            # Update the batch size if necessary.
            self.upate_batch_size(raw_image_shape, out_image_shape, flag_float)

            # Prepare the shared memory.
            self.prepare_shared_memory(raw_image_shape, out_image_shape, flag_float)

            # Figure out the batch splits.
            N = len(img_num_str_list)
            raw_fn_indices = np.arange( N, dtype=int )
            batch_idx_splits = np.array_split(raw_fn_indices, N // self.batch_size + 1) \
                if N != self.batch_size else [ raw_fn_indices ]

            count = 0
            try:
                for batch_indices in batch_idx_splits:
                    # The list of images.
                    # print(f'Batch read {batch_indices.size} images.')

                    # Prepare the parallel processing arguments for reading images.
                    rep_raw_image_dir = ReplicatedArgument( os.path.join( env_dir, traj_name, sampler_dict['raw_camera'] ), batch_indices.size )
                    rep_raw_image_suffix = ReplicatedArgument( sampler_dict['raw_image_suffix'], batch_indices.size )
                    img_num_str_list_batch = [ img_num_str_list[idx] for idx in batch_indices ]
                    zipped_args = zip( range(batch_indices.size), rep_raw_image_dir, img_num_str_list_batch, rep_raw_image_suffix )

                    # Read the images in parallel.
                    with PoolWithLogger(self.image_io_job_num, image_io_job_initializer, 'mvs', None, 
                            (self.shm_in.name, raw_image_shape, sampler_dict['image_reader'], flag_float)) as pool:
                        results = pool.map( read_image_job, zipped_args )

                    # Re-arange the images as a list. This should not have any copies.
                    raw_images = [ self.shm_img_in[i] for i in range(batch_indices.size) ]

                    # print(f'Batch read done. ')

                    # Batch sample using GPU.
                    sampled_images, valid_mask = sampler_dict['sampler']( raw_images, interpolation=sampler_dict['interpolation'] )

                    # Copy the sampled images to the shared memory.
                    for i, img in enumerate(sampled_images):
                        # self.shm_img_out[i][:, :, ...] = img
                        self.shm_img_out[i] = img

                    # Prepare the arguments for parallel writing.
                    rep_out_dir = ReplicatedArgument( os.path.join(
                                                        self.processed_dataset_path, 
                                                        env_name, 
                                                        traj_name,
                                                        sampler_dict['output_dir'] ), batch_indices.size )
                    rep_out_image_suffix = ReplicatedArgument( sampler_dict['output_image_suffix'], batch_indices.size )
                    zipped_args = zip( range(batch_indices.size), rep_out_dir, img_num_str_list_batch, rep_out_image_suffix)

                    # Write the sampled images.
                    with PoolWithLogger(self.image_io_job_num, image_io_job_initializer, 'mvs', None, 
                            (self.shm_out.name, out_image_shape, sampler_dict['image_reader'], flag_float)) as pool:
                        results = pool.map( write_image_job, zipped_args )

                    # Handle the mask generation.
                    if sampler_dict['write_valid_mask']:
                        if not mask_generation_flags[sampler_index]:
                            valid_mask = valid_mask.astype(np.uint8) * 255
                            cv2.imwrite( os.path.join(
                                            self.processed_dataset_path, 
                                            env_name, 
                                            traj_name, 
                                            sampler_dict['output_dir'], 
                                            'mask.png' ),
                                        valid_mask )
                            mask_generation_flags[sampler_index] = True

                    # Update the counter.
                    count += batch_indices.size

                    print(f'{count}/{N} fisheye images written... ')
            finally:
                self.cleanup_shared_memory()

        # Generate the file list for this trajectory. 
        self.generate_file_list_for_trajectory( 
            os.path.join( self.processed_dataset_path, env_name, traj_name ),
            img_num_str_list )

    def preprocess_requested_imgs(self):
        # env_traj_num: [
        #     'name': env_name,
        #     'dir': env_dir,
        #     'metadata_reader': metadata_reader,
        #     'trajectories': [
        #         {
        #             'name': traj_name,
        #             'index_strings': img_index_arr[1:, 0]
        #         },
        #         ...
        #     ],
        #     ... }
        time_start = time.time()
        for env_dict in self.env_traj_num:
            env_name        = env_dict['name']
            env_dir         = env_dict['dir']
            # metadata_reader = env_dict['metadata_reader']

            print(f'Processing {env_name}: ')

            for traj_dict in env_dict['trajectories']:
                traj_name = traj_dict['name']
                img_num_str_list = traj_dict['index_strings']

                print(f'Trajectory: {traj_name}')
                self.process_trajectory( env_dir, env_name, traj_name, img_num_str_list )
                

        time_end = time.time()
        total_time_secs = time_end - time_start
        print(f'Total processing time: {total_time_secs:.2f} s ({total_time_secs/3600:.2f} hrs)')
        print(f'Total number of envs: {len(self.env_traj_num)}')
        print(f'self.total_frames = {self.total_frames}')
        print(f'self.debug_n = {self.debug_n}')

        if self.debug_n > 0:
            secs_per_frame = total_time_secs / ( self.debug_n * len(self.env_traj_num) )
            projected_total_secs = secs_per_frame * self.total_frames
            print(f'Average processing time per frame: { secs_per_frame:.2f} s')
            print(f'Projected total processing time: { projected_total_secs:.2f} s ({projected_total_secs / 3600} hrs)')
