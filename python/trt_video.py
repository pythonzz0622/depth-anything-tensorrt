import argparse
import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from torchvision.transforms import Compose
import torch 
import torch.nn.functional as F
# from CVutils.Timer import FrameTimer
import time

class FrameTimer:
    '''Frame단위 시간 측정하는 class'''
    def __init__(self, frame_count):
        self.frame_count = frame_count
        self.rate = 1 /  self.frame_count
        self.times = []

    def set_timer(self):
        self.start_time = time.time()

    def _step(self):
        '''시간을 list에다가 저장함'''
        elapsed_time = time.time() - self.start_time
        self.times.append(elapsed_time)
        if len(self.times) > self.frame_count:
            self.times.pop(0)

    def _get_avg(self ):
        '''시간의 평균을 계산'''
        if len(self.times) < self.frame_count:
            return {'fps' : 0 , 'msec' : 0}
        avg_time = sum(self.times) / len(self.times)
        fps_avg = round(1 / avg_time, 1)
        msec= round(avg_time * 1000 , 1)

        return {'fps' : fps_avg , 'msec' : msec}
    
    def draw_frame_rate(self , frame ):
        self._step()
        times = self._get_avg()
        cv2.putText(frame, f"Avg msec : {times['msec']:.2f}msec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        cv2.putText(frame, f"Avg FPS : {times['fps']:.2f} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def print_frame_rate(self):
        self._step()
        times = self._get_avg()
        print( f"Avg msec : {times['msec']:.2f}msec", end = '\r')

    def frame_rate_control(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time < self.rate:  # 40 msec 미만이면 나머지 시간만큼 대기
            # print(f'{elapsed_time * 1000}msec')
            time.sleep(self.rate - elapsed_time)

class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample

class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # resize sample
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(
                    sample["disparity"],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )

            if "semseg_mask" in sample:
                # sample["semseg_mask"] = cv2.resize(
                #     sample["semseg_mask"], (width, height), interpolation=cv2.INTER_NEAREST
                # )
                sample["semseg_mask"] = F.interpolate(torch.from_numpy(sample["semseg_mask"]).float()[None, None, ...], (height, width), mode='nearest').numpy()[0, 0]
                
            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                # sample["mask"] = sample["mask"].astype(bool)

        # print(sample['image'].shape, sample['depth'].shape)
        return sample

class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
            
        if "semseg_mask" in sample:
            sample["semseg_mask"] = sample["semseg_mask"].astype(np.float32)
            sample["semseg_mask"] = np.ascontiguousarray(sample["semseg_mask"])

        return sample
    
    
transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

def run(args):
    timer = FrameTimer(30)
    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.rtsp)  # Open rtsp stream

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the camera.")
        return
    
    # Loop to read frames from the camera
    while True:
        timer.set_timer()
        ret, frame = cap.read()
        
        # Check if frame is empty
        if not ret:
            print("Error: Couldn't read frame.")
            break

        orig_shape = frame.shape[:2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        frame = transform({"image": frame})["image"]  # C, H, W
        frame = frame[None]  # B, C, H, W

        # Create logger and load the TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(args.engine, 'rb') as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        with engine.create_execution_context() as context:
            input_shape = context.get_tensor_shape('input')
            output_shape = context.get_tensor_shape('output')
            h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
            h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            d_input = cuda.mem_alloc(h_input.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)
            stream = cuda.Stream()
            
            # Copy the input image to the pagelocked memory
            np.copyto(h_input, frame.ravel())
            
            # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
            cuda.memcpy_htod_async(d_input, h_input, stream)
            context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            depth = h_output
            
        # Process the depth output
        depth = np.reshape(depth, output_shape[2:])
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        timer.frame_rate_control()
        timer.draw_frame_rate(depth)
        print(depth.shape)
        # Show the depth map
        cv2.imshow('Depth Map', depth)
        
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation with a TensorRT engine.')
    parser.add_argument('--rtsp', type=str, required=True, help='RTSP stream URL')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Output directory for the depth map')
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine')
    
    args = parser.parse_args()
    run(args)
