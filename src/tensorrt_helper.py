import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import torch
from collections import OrderedDict, namedtuple
from PIL import Image


class TRTModel:
    def __init__(self, engine_path, device="cuda"):
        self.device = device
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.INFO)
        dummy_tensor = torch.tensor([0.0]).to(self.device)  # workaround for some TRT engine bugs

        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())

        self.context = self.model.create_execution_context()
        self.bindings = OrderedDict()
        self.output_names = []
        self.fp16 = False
        self.dynamic = False

        is_trt10 = not hasattr(self.model, "num_bindings")
        num = range(self.model.num_io_tensors) if is_trt10 else range(self.model.num_bindings)

        for i in num:
            if is_trt10:
                name = self.model.get_tensor_name(i)
                dtype = trt.nptype(self.model.get_tensor_dtype(name))
                is_input = self.model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                if is_input:
                    if -1 in self.model.get_tensor_shape(name):
                        self.dynamic = True
                        self.context.set_input_shape(name, self.model.get_tensor_profile_shape(name, 0)[2])
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_names.append(name)
                shape = tuple(self.context.get_tensor_shape(name))
            else:
                name = self.model.get_binding_name(i)
                dtype = trt.nptype(self.model.get_binding_dtype(i))
                is_input = self.model.binding_is_input(i)
                if is_input:
                    if -1 in self.model.get_binding_shape(i):
                        self.dynamic = True
                        self.context.set_binding_shape(i, self.model.get_profile_shape(0)[i][2])
                    if dtype == np.float16:
                        self.fp16 = True
                else:
                    self.output_names.append(name)
                shape = tuple(self.context.get_binding_shape(i))

            torch_dtype = torch.float16 if self.fp16 else torch.float32
            tensor = torch.empty(shape, dtype=torch_dtype, device=self.device)
            self.bindings[name] = Binding(name, dtype, shape, tensor, int(tensor.data_ptr()))

        self.binding_addrs = OrderedDict((n, b.ptr) for n, b in self.bindings.items())
        self.batch_size = self.bindings["images"].shape[0]

    def _update_output_bindings(self):
        for name in self.output_names:
            idx = self.model.get_binding_index(name)
            new_shape = tuple(self.context.get_binding_shape(idx))
            dtype = torch.float16 if self.fp16 else torch.float32
            new_tensor = torch.empty(new_shape, dtype=dtype, device=self.device)
            self.bindings[name] = self.bindings[name]._replace(
                data=new_tensor, shape=new_shape, ptr=int(new_tensor.data_ptr())
            )
            self.binding_addrs[name] = int(new_tensor.data_ptr())

    def __call__(self, input_data):
        input_data = input_data.contiguous()
        assert isinstance(input_data, torch.Tensor) and input_data.device.type == "cuda"

        name = "images"
        if self.dynamic and input_data.shape != self.bindings[name].shape:
            idx = self.model.get_binding_index(name)
            self.context.set_binding_shape(idx, input_data.shape)
            self.bindings[name] = self.bindings[name]._replace(
                shape=input_data.shape, data=input_data, ptr=int(input_data.data_ptr())
            )
            self.binding_addrs[name] = int(input_data.data_ptr())
            self._update_output_bindings()

        assert input_data.shape == self.bindings[name].shape
        self.binding_addrs[name] = int(input_data.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        cuda.Context.synchronize()

        return [self.bindings[n].data for n in self.output_names]

def resize_and_center_crop(img_array, target_size=(640, 640)):
    img = Image.fromarray(img_array)
    shorter = min(img.size)
    scale = target_size[0] / shorter
    new_size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(new_size, Image.LANCZOS)

    # Center crop
    left = (img_resized.width - target_size[0]) // 2
    top = (img_resized.height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]
    img_cropped = img_resized.crop((left, top, right, bottom))

    return np.array(img_cropped)

if __name__ == "__main__":
    engine_path = "tph-yolov5/0311multi_classes_boat.engine"
    model = TRTModel(engine_path, device="cuda")

    img = Image.open("example.png").convert("RGB")
    img_array = resize_and_center_crop(np.array(img), target_size=(2016, 2016))
    img_array = img_array.transpose(2, 0, 1)  # CHW
    img_tensor = torch.from_numpy(img_array).float().div(255.0).unsqueeze(0).to("cuda")  # normalize and add batch dim

    if model.fp16:
        img_tensor = img_tensor.half()

    output_data = model(img_tensor)

    print(f"Output tensors: {len(output_data)}")
    for i, out in enumerate(output_data):
        print(f"[{i}] shape: {out.shape}")
    print("Done.")
