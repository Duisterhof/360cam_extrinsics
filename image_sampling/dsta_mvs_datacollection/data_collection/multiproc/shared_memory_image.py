from multiprocessing import shared_memory
import numpy as np

def shm_size_from_img_shape( img_shape, channel_depth, n_img ):
    return int( np.prod( img_shape ) * channel_depth * n_img )

def pass_forward(img):
    return img

def c4_uint8_as_float(img):
    assert img.ndim == 3 and img.shape[2] == 4, f'img.shape = {img.shape}. Expecting image.shape[2] == 4. '
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)

    return img.view('<f4')

def float_as_c4_uint8(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    elif img.ndim == 3 and img.shape[2] != 1:
        raise Exception(f'img.shape = {img.shape}. Expecting image.shape[2] == 1. ')
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)

    return img.view('<u1')

class SharedMemoryImage(object):
    def __init__(self, name: str, img_shape: tuple, channel_depth: int=1, processor_in=None, processor_out=None):
        '''
        This is a wrapper class for using shared memory to store images across
        mulitple processes. The shared memory object is created by the user and
        refered by the name.

        The user specifies the shape of the image. And the total number of images
        in the shared memory is then determined by the size of the shared memory. 
        
        Note that the user is supposed to call initialize() and finalize() to properly
        handle the shared resources.

        Note that internally, this class treats all image as uint8. The user must properly
        set the channel_depth value and the associated processor_in and processor_out functions. 
        porcessor_in is used to convert the input image to uint8 type with channel_depth. 
        Inversely, processor_out is used to convert the image from uint8 type to the original type.
        Use None for processor_in and processor_out if no conversion is needed.

        name: The name of the shared memory object.
        img_shape: The shape of the image, including channel number.
        channel_depth: The number of bytes per channel. Default is 1. For single-precision
            floating point values, use 4.
        processor_in: A callable object that performs the conversion to uint8.
        processor_out: A callable object that performs the conversion from uint8.
        '''
        super().__init__()

        self.name = name
        self.img_shape = ( *img_shape[:2], img_shape[2] * channel_depth) \
            if len(img_shape) == 3 else ( *img_shape[:2], channel_depth )
        self.img_capacity = np.prod( self.img_shape )

        self.prorcessor_in = processor_in
        self.prorcessor_out = processor_out

        self.shm = None # User must call initialize() to assign a valid value.
        self._n_img = 0 # This should not be assigned buy the user.

    @property
    def n_img(self):
        return self._n_img

    def initialize(self):
        if ( self.shm is not None ):
            raise Exception('Already initialized. ')

        self.shm = shared_memory.SharedMemory(name=self.name)

        # Check the size of the shared memory.
        assert ( self.shm.size > 0 and self.shm.size % self.img_capacity == 0 ), \
            'Incompatible shared memory size. self.shm.size = {}, self.img_capacity = {}, self.img_shape = {}'.format( 
                self.shm.size, self.img_capacity, self.img_shape )
        self._n_img = self.shm.size // self.img_capacity

    def finalize(self):
        if ( self.shm is not None ):
            self.shm.close()

    def __del__(self):
        self.finalize()

    def seek(self, idx):
        assert ( idx < self.n_img and idx >= 0 ), 'self.n_img = {}, idx = {}'.format( self.n_img, idx )
        return self.shm.buf[ idx * self.img_capacity : (idx+1) * self.img_capacity ]

    def __len__(self):
        return self.n_img

    def as_array_at_idx(self, idx):
        return np.ndarray( 
            ( self.img_capacity, ), 
            dtype=np.uint8, 
            buffer=self.seek(idx) 
            ).reshape( self.img_shape )

    def __getitem__(self, idx):
        array = self.as_array_at_idx(idx)

        if self.prorcessor_out is not None:
            return self.prorcessor_out(array)
        else:
            return array

    def __setitem__(self, idx, img):
        if self.prorcessor_in is not None:
            img = self.prorcessor_in(img)

        array = self.as_array_at_idx(idx)
        array[:, :, ...] = img
        