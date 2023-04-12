import signal
import numpy as np
import scipy.ndimage
from multiprocessing import Pool, cpu_count
from PIL import Image
from tools import set_contrast


def Initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class MIPGenerator:

    numpy_array: np.ndarray
    frames: int
    delay: int
    projection: int

    def __init__(self, numpy_array: np.ndarray, frames: int, delay: int, projection: int = 360):
        self.numpy_array = numpy_array
        self.frames = frames
        self.delay = delay
        self.projection = projection

    def _project(self, angle: int) -> np.ndarray:
        vol_angle = scipy.ndimage.rotate(
            self.numpy_array, angle=angle, reshape=False, axes=(1, 2), order=0, mode='constant', cval=0.0, prefilter=False)
        MIP = np.amax(vol_angle, axis=1)
        MIP = np.flip(MIP, axis=0)
        return MIP

    def _create_projection_list(self) -> list:
        angles = np.linspace(0, self.projection, self.frames)
        nbCores = min(cpu_count(), 4) if cpu_count() > 2 else 1
        pool = Pool(nbCores, initializer=Initializer)
        projection_list = pool.map(self._project, angles)
        pool.close()
        pool.join()
        return projection_list

    def create_gif(self, output: bytes) -> None:
        projections = np.stack(self._create_projection_list())
        projections = set_contrast(projections)
        image = Image.fromarray(projections[0])
        projections = np.delete(projections, 0, axis=0)
        images = [Image.fromarray(projection) for projection in projections]
        image.save(output, format='GIF', append_images=images,
                   save_all=True, duration=self.delay, loop=0)
