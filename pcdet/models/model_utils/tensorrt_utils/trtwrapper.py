# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Optional, Sequence, Union

import tensorrt as trt
import torch

from .utils import load_trt_engine, torch_device_from_trt, torch_dtype_from_trt


class TRTWrapper(torch.nn.Module):
    """TensorRT engine wrapper for inference.

    Args:
        engine (tensorrt.ICudaEngine): TensorRT engine to wrap.
        output_names (Sequence[str] | None): Names of model outputs  in order.
            Defaults to `None` and the wrapper will load the output names from
            model.

    Note:
        If the engine is converted from onnx model. The input_names and
        output_names should be the same as onnx model.

    Examples:
        >>> from mmdeploy.backend.tensorrt import TRTWrapper
        >>> engine_file = 'resnet.engine'
        >>> model = TRTWrapper(engine_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(
            self,
            engine: Union[str, trt.ICudaEngine],
            input_names: Sequence[str],
            output_names: Sequence[str],
    ):
        super().__init__()
        # NOTE use TensorRT default one
        # load_tensorrt_plugin()
        trt.init_libnvinfer_plugins(None, '')
        self.engine = engine
        if isinstance(self.engine, str):
            self.engine = load_trt_engine(engine)

        if not isinstance(self.engine, trt.ICudaEngine):
            raise TypeError(f'`engine` should be str or trt.ICudaEngine, \
                but given: {type(self.engine)}')

        self._input_names = input_names
        self._output_names = output_names

        # self._register_state_dict_hook(TRTWrapper.__on_state_dict)
        self.context = self.engine.create_execution_context()

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): The input name and tensor pairs.

        Return:
            Dict[str, torch.Tensor]: The output name and tensor pairs.
        """
        bindings = [None] * (len(self._input_names) + len(self._output_names))

        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(eval(repr(profile[0])), input_tensor.shape,
                                             eval(repr(profile[2]))):
                assert s_min <= s_input <= s_max, \
                    f'Input shape of {input_name} should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            # assert 'cuda' in input_tensor.device.type
            # input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
                print(input_name)
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = eval(repr(self.context.get_binding_shape(idx)))

            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()

        self.__trt_execute(bindings=bindings)

        return outputs

    # @TimeCounter.count_time()
    def __trt_execute(self, bindings: Sequence[int]):
        """Run inference with TensorRT.

        Args:
            bindings (list[int]): A list of integer binding the input/output.
        """
        self.context.execute_async_v2(
            bindings,
            torch.cuda.current_stream().cuda_stream,
        )
