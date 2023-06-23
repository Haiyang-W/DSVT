# Copyright (c) OpenMMLab. All rights reserved.
# import logging
from typing import Dict, Sequence, Union

import onnx
import tensorrt as trt
import torch
from packaging import version


def create_trt_engine(
        onnx_model: str, # onnx file path
        input_shapes: Dict[str, Sequence[int]],
        log_level: trt.Logger.Severity = trt.Logger.WARNING,
        fp16_mode: bool = True,
        # int8_mode: bool = False,
        # int8_param: dict = None,
        max_workspace_size: int = 1 << 31,  # default 2GB
        device: Union[int, str, torch.device] = 0,
        **kwargs
) -> trt.IHostMemory:
    """Create a tensorrt engine from ONNX.

    Args:
        onnx_model (str or onnx.ModelProto): Input onnx model to convert from.
        input_shapes (Dict[str, Sequence[int]]): The min/opt/max shape of
            each input.
        log_level (trt.Logger.Severity): The log level of TensorRT. Defaults to
            `trt.Logger.WARNING`.
        fp16_mode (bool): Specifying whether to enable fp16 mode.
            Defaults to `True`.
        int8_mode (bool): Specifying whether to enable int8 mode.
            Defaults to `False`.
        int8_param (dict): A dict of parameter  int8 mode. Defaults to `None`.
        max_workspace_size (int): To set max workspace size of TensorRT engine.
            some tactics and layers need large workspace. Defaults to `1 << 30`.
        device_id (int): Choice the device to create engine. Defaults to `0`.

    Returns:
        tensorrt.IHostMemory: The TensorRT serialized engine
            created from onnx_model.

    Example:
        >>> from mmdeploy.apis.tensorrt import create_trt_engine
        >>> engine = create_trt_engine(
        >>>             "onnx_model.onnx",
        >>>             {'input': {"min_shape" : [1, 3, 160, 160],
        >>>                        "opt_shape" : [1, 3, 320, 320],
        >>>                        "max_shape" : [1, 3, 640, 640]}},
        >>>             log_level=trt.Logger.WARNING,
        >>>             fp16_mode=True,
        >>>             max_workspace_size=1 << 30,
        >>>             device_id=0)
        >>>             })
    """
    # load_tensorrt_plugin()
    # device = torch.device('cuda:{}'.format(device_id))
    if not isinstance(device, torch.device):
        device = torch.device(device)
    # create builder and network
    logger = trt.Logger(log_level)
    # NOTE: default
    trt.init_libnvinfer_plugins(logger, namespace="")
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_model)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    assert success, f"tensorrt parse from file {onnx_model} failed!"

    config = builder.create_builder_config()
    # NOTE [DEPRECATED]
    # config.max_workspace_size = max_workspace_size
    # Based on tensorrt official Python API:
    # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#build_engine_python
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    profile = builder.create_optimization_profile()

    for input_name, param in input_shapes.items():
        min_shape = param['min_shape']
        opt_shape = param['opt_shape']
        max_shape = param['max_shape']
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    # create engine
    with torch.cuda.device(device):
        # output trt.IHostMemory
        serialized_engine = builder.build_serialized_network(network, config)
        # output trt.ICudaEngine NOTE: DeprecationWarning
        # engine = builder.build_engine(network, config)

    assert serialized_engine is not None, 'Failed to create TensorRT engine'
    return serialized_engine


def save_trt_engine(engine: trt.IHostMemory, path: str) -> None:
    """Save serialized TensorRT engine to disk.

    Args:
        engine (tensorrt.IHostMemory): TensorRT serialized engine to be saved.
        path (str): The absolute disk path to write the engine.
    """
    with open(path, mode='wb') as f:
        f.write(engine)
    print(f">>> TensorRT inference engine saved to {path}.")


def load_trt_engine(path: str) -> trt.ICudaEngine:
    """Deserialize TensorRT engine from disk.

    Args:
        path (str): The disk path to read the engine.

    Returns:
        tensorrt.ICudaEngine: The TensorRT engine loaded from disk.
    """
    # load_tensorrt_plugin()
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        print(f"TensorRT engine {path} successfully loaded.")
        return engine


def torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    """Convert pytorch dtype to TensorRT dtype.

    Args:
        dtype (str.DataType): The data type in tensorrt.

    Returns:
        torch.dtype: The corresponding data type in torch.
    """

    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f'{dtype} is not supported by torch')


def torch_device_from_trt(device: trt.TensorLocation):
    """Convert pytorch device to TensorRT device.

    Args:
        device (trt.TensorLocation): The device in tensorrt.
    Returns:
        torch.device: The corresponding device in torch.
    """
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError(f'{device} is not supported by torch')
