import numpy as np
import math


class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params

class ModelOpsAttrsExtractor(object):
    def __init__(self, model, subgraph):

        try:
            from tflite.BuiltinOperator import BuiltinOperator
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self.model = model
        self.subgraph = subgraph
        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())
        self.prefetched_nodes = {}
        self.ret_dict = {}
        self.ret_list = []
        self.op_type_in_model = []

        # for openvino
        self.op_cnt = 0  # count the number of ops in the model, and index the op by op_string_count
        self.data_node_map = {}  # for Const and Placeholder data such as weights, bias, and inputs
        self.data_dict = {}  # data node dict
        self.param_value_dict = {}
        self.ordered_ops_list = []  


        self.op_extractors_map = {
            "CONV_2D": self.conv2d_extractor,
            "DEPTHWISE_CONV_2D": self.depthwise_conv2d_extractor,
            "AVERAGE_POOL_2D": self.average_pool2d_extractor,
            "MAX_POOL_2D": self.max_pool_2d_extractor,
            "RESHAPE": self.reshape_extractor,
            "SOFTMAX": self.softmax_extractor,
            "FULLY_CONNECTED": self.fully_connected_extractor
        }

    def extract_model_ops_and_attrs(self):
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            output_tensors = self.get_output_tensors(op)
            # print(op_code_str)
            if not op_code_str in self.op_type_in_model:
                self.op_type_in_model.append(op_code_str)
            # print(get_tensor_name(self.subgraph, output_tensors[0].tensor_idx))

            try:
                from tflite.Operator import Operator
            except ImportError:
                raise ImportError("The tflite package must be installed")

            assert isinstance(op, Operator)
            self.op_extractors_map[op_code_str](op)

        return self.ordered_ops_list, self.data_dict, self.param_value_dict

    def unique_id(self, op_code_str):
        id = op_code_str + '_' + str(self.op_cnt)
        self.op_cnt += 1
        return id

    def get_op_code_str(self, op):
        """Get TFLite ops string representation"""
        try:
            from tflite.BuiltinOperator import BuiltinOperator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        op_code_list_idx = op.OpcodeIndex()

        op_c = self.model.OperatorCodes(op_code_list_idx)
        # In TFlite 2.4.x there was a change where the type of the field that contained
        # the builtin code changed from int8 to int32 in the flat buffer representation.
        # However to retain support for old flat buffers that were created, they retained
        # the original 8 bit encoding for the operator but in a new field accessed by the
        # DeprecatedBuiltinCode method.
        # This means that the API function BuiltinCode() is used on an operator
        # which was originally encoded as an 8 bit quantity it would look for the
        # code in the new int32 field in the schema and this creates the need
        # for the check for the magic number of 127 which is indicated by
        # BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
        # Remember however that this value came into existence only after Tensorflow
        # lite 2.4.x and hence encase it in a try -except block.
        # Phew !
        try:
            if op_c.BuiltinCode() < BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES:
                opc = op_c.DeprecatedBuiltinCode()
            else:
                opc = op_c.BuiltinCode()
        except AttributeError:
            opc = op_c.BuiltinCode()

        op_code_id = opc
        try:
            op_code_str = self.builtin_op_code[op_code_id]
        except KeyError:
            raise NotImplementedError(
                "TFLite operator with code "
                + str(op_code_id)
                + " is not supported by this version of the fbs schema."
            )
        if op_code_id == BuiltinOperator.CUSTOM:
            # Custom operator
            custom_op_code_str = self.model.OperatorCodes(op_code_list_idx).CustomCode()
            if custom_op_code_str == b"TFLite_Detection_PostProcess":
                return "DETECTION_POSTPROCESS"

            raise NotImplementedError("Custom operators are currently not supported")
        return op_code_str

    def get_input_tensors(self, op):
        operator_inputs = op.InputsAsNumpy()
        return self.get_tensors(operator_inputs)

    def get_output_tensors(self, op):
        operator_outputs = op.OutputsAsNumpy()
        return self.get_tensors(operator_outputs)


    # whole function not modified 
    def get_tensors(self, tensors_idx_list):
        """Get tensor wrapper list from given TFLite tensor index list"""
        return_list = list()
        for tensor_idx in tensors_idx_list:
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, 0, 0))
                continue

            tensor = self.subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                # TFLite supports both per-tensor and per-axis (aka channel) quantization.  For
                # per-tensor quantization, scale and zero points are scalar values.  For per-axis
                # quantization, scale and zero points for the weights are tensors (activations are
                # per-tensor quantized). However, the TFLite quantization spec puts restrictions on
                # zero points for per-axis quantization.  Specifically, the zero point is a tensor
                # but all values are 0. More information can be found here -
                # https://www.tensorflow.org/lite/performance/quantization_spec

                tflite_scale = tflite_qnn_params.ScaleAsNumpy()
                tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
                is_qnn_params_valid = True

                # Handle Per-axis and per-tensor cases
                if isinstance(tflite_scale, np.ndarray):
                    assert isinstance(tflite_zero_point, np.ndarray)

                    # Tensor - Per-axis quantization
                    if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                        scale = tflite_scale
                        # Ensure that all zero points are zeros
                        zero_point = tflite_zero_point
                        if not np.all(zero_point == 0):
                            raise tvm.error.OpAttributeInvalid(
                                "TFLite per-axis quantization restricts all zero points to be"
                                + " 0, but a non-zero value is observed"
                            )
                        zero_point = int(zero_point[0])

                    # Scalar - Per-tensor quantization
                    elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                        scale = float(tflite_scale[0])
                        zero_point = int(tflite_zero_point[0])

                    else:
                        raise NotImplementedError(
                            "Quantized type {} (scale) and  {} (zero point) not supported".format(
                                type(tflite_scale), type(tflite_zero_point)
                            )
                        )
                elif tflite_scale == 0 and tflite_zero_point == 0:
                    # Handle corner case for ops like quantized reshape whose second operand (shape)
                    # has zero scale and zero zero point. This is not used.
                    is_qnn_params_valid = False
                else:
                    raise NotImplementedError(
                        "Quantized type {} not supported".format(type(tflite_scale))
                    )

                # Check that the scale and zero points are valid.
                if is_qnn_params_valid:
                    qnn_params = dict()
                    qnn_params["scale"] = scale
                    qnn_params["zero_point"] = zero_point
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
        return return_list

    def get_tensor_type_str(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_type == TensorType.INT8:
            return "int8"
        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT16:
            return "float16"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        if tensor_type == TensorType.BOOL:
            return "bool"
        raise NotImplementedError(
            "Tensor type {} is currently not supported".format(str(tensor_type))
        )

    def get_tensor_type_as_numpy(self, tensor_wrapper):
        """Returns np.dtype out of TensorType"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        try:
            from tflite.TensorType import TensorType

            return {
                TensorType.UINT8: np.uint8,
                TensorType.INT8: np.int8,
                TensorType.FLOAT16: np.float16,
                TensorType.FLOAT32: np.float32,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.BOOL: np.bool_,
            }[tensor_wrapper.tensor.Type()]
        except ImportError:
            raise ImportError("The tflite package must be installed")
        except KeyError:
            raise NotImplementedError(
                "Tensor type '{}' currently not supported".format(tensor_wrapper.tensor.Type())
            )

    def get_tensor_value(self, tensor_wrapper, is_sparse=False):
        """Get tensor buffer value from given tensor wrapper"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        dtype = self.get_tensor_type_as_numpy(tensor_wrapper)
        data = tensor_wrapper.buffer.DataAsNumpy()

        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = to_int_list(self.get_tensor_shape(tensor_wrapper))
        else:
            shape = []

        if is_sparse:
            return np.frombuffer(data, dtype=dtype)
        else:
            return np.frombuffer(data, dtype=dtype).reshape(shape)


    def initialize_op_list_element(self, op):
        dict_to_be_appended = {}
        input_tensors_name_list = []
        output_tensors_name_list =[]

        for in_tensor in op.InputsAsNumpy():
            input_tensors_name_list.append(get_tensor_name(self.subgraph, in_tensor))

        for out_tensor in op.OutputsAsNumpy():
            output_tensors_name_list.append(get_tensor_name(self.subgraph, out_tensor))

        id = self.unique_id(self.get_op_code_str(op))
        dict_to_be_appended['id'] = id 
        dict_to_be_appended['tf_op_type'] = self.get_op_code_str(op)
        dict_to_be_appended['in'] = input_tensors_name_list
        dict_to_be_appended['out'] = output_tensors_name_list
        dict_to_be_appended['temp_inter'] = [id + '_output']  # when handling fused activation, this element will be pop and used.
        
        return dict_to_be_appended

    def conv2d_extractor(self, op):
        self.conv_extractor(op, "conv2d")

    def depthwise_conv2d_extractor(self, op):
        self.conv_extractor(op, "depthwise")

    def average_pool2d_extractor(self, op):
        self.pool2d_extractor(op, "average")

    def max_pool_2d_extractor(self, op):
        return {'max_pool_2d': 2}


    def reshape_extractor(self, op):
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ReshapeOptions import ReshapeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (1, 2), "input tensors should not be empty"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "There should be only 1 output tensor"


        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        if len(input_tensors) == 2:
            shape_tensor = input_tensors[1]
            target_shape = self.get_tensor_value(shape_tensor)
            # convert to flattened list
            from itertools import chain

            try:
                target_shape = list(chain(*target_shape))
            except TypeError:
                target_shape = list(chain(target_shape))

        else:
            assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
            op_options = op.BuiltinOptions()
            reshape_options = ReshapeOptions()
            reshape_options.Init(op_options.Bytes, op_options.Pos)
            target_shape = reshape_options.NewShapeAsNumpy()

        input_tensor_name = get_tensor_name(self.subgraph, input_tensors[0].tensor_idx)
        output_tensor_name = get_tensor_name(self.subgraph, output_tensors[0].tensor_idx)

        target_shape = np.array(target_shape, dtype=np.int64)
        reshape_node_id = self.unique_id('RESHAPE')
        reshape_const_name = reshape_node_id + '_const'
        self.data_dict[reshape_const_name] = {'shape': (2,), 'dtype': 'int64', 'value':target_shape}
        self.param_value_dict[reshape_const_name] = target_shape
        self.ordered_ops_list.append({'id':reshape_node_id, 'tf_op_type':'RESHAPE', 'in':[input_tensor_name, reshape_const_name], 'out':[output_tensor_name], 'attrs':{'target_shape':target_shape}})



    def softmax_extractor(self, op):
        """Convert TFLite softmax"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        dict_to_be_appended = self.initialize_op_list_element(op)

        params = {"axis": 1}  # 1 is channel
        dict_to_be_appended['attrs'] = params

        self.ordered_ops_list.append(dict_to_be_appended)


    # now only implement avg_pool #
    def pool2d_extractor(self, op, pool_type):
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Pool2DOptions import Pool2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors should be 1"

        dict_to_be_appended = self.initialize_op_list_element(op)

        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool2d_options = Pool2DOptions()
        pool2d_options.Init(op_options.Bytes, op_options.Pos)
        stride_h = pool2d_options.StrideH()
        stride_w = pool2d_options.StrideW()
        padding = pool2d_options.Padding()
        filter_h = pool2d_options.FilterHeight()
        filter_w = pool2d_options.FilterWidth()
        fused_activation_fn = pool2d_options.FusedActivationFunction()

        params = {
            "pool_size": (filter_h, filter_w),
            "strides": (stride_h, stride_w),
            "padding": [0, 0],
            "layout": "NHWC",
        }

        _, input_h, input_w, _ = to_int_list(self.get_tensor_shape(input_tensor))

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, filter_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, filter_w, stride_w)
            params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
        else:
            raise Exception(
                "Padding format {} for operator Pool2D is not supported.".format(padding)
            )

        if pool_type == "average":
            dict_to_be_appended['attrs'] = params


        # elif pool_type == "max":
        #     pass
        # elif pool_type == "l2":
        #     pass

        else:
            raise Exception(
                "Operator {} is not yet supported for TFLite in our parser.".format(pool_type + " pool")
            )

        self.convert_fused_activation_function(fused_activation_fn, dict_to_be_appended)

        
    def fully_connected_extractor(self, op):
        """Convert TFLite fully connected"""
        try:
            from tflite.FullyConnectedOptions import FullyConnectedOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (2, 3), "input tensors length should be two or three"

        input_tensor = input_tensors[0]
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        weight_tensor_shape = to_int_list(self.get_tensor_shape(weight_tensor))

        # Weight should have only 2 dimensions(TFLite convention)
        assert len(weight_tensor_shape) == 2, "Weight should be only 2-dim"

        # dict_to_be_appended is used differently from other ops in fc op
        # it is rather a initializer
        dict_to_be_appended = self.initialize_op_list_element(op)   

        # Input shape: [i_batch_size, ..., n_inputs]
        # Filter shape: [n_inputs, n_units]
        #
        # As we will transform Fully_Connected Input to Dense Op inputs as below
        # Dense expected Input shape: [batch_size, n_units]
        # Dense expected Weight shape: [out_dim, n_units]
        # Dense output shape: [batch_size, out_dim]
        target_shape = tuple((-1, weight_tensor_shape[1]))
        target_shape = np.array([-1, weight_tensor_shape[1]])

        # when encountering fully-connected op, first do things as handling a RESHAPE node, 
        # that is, add a const node to data_dict, and append a reshape op to ordered_op_list
        reshape_node_id = self.unique_id('RESHAPE')
        input_tensor_name = get_tensor_name(self.subgraph, input_tensor.tensor_idx)
        reshape_const_name = reshape_node_id + '_const'
        self.data_dict[reshape_const_name] = {'shape': (2,), 'dtype': 'int64', 'value':target_shape}
        self.param_value_dict[reshape_const_name] = target_shape
        fc_reshape_output_name = dict_to_be_appended['id'] + '_reshape_result'
        self.ordered_ops_list.append({'id':reshape_node_id, 'tf_op_type':'RESHAPE', 'in':[input_tensor_name, reshape_const_name], 'out':[fc_reshape_output_name], 'attrs':{'target_shape':target_shape}})
        
        dict_to_be_appended['in'] = [fc_reshape_output_name, dict_to_be_appended['in'][1], dict_to_be_appended['in'][2]]
        dict_to_be_appended['attrs'] = {}
        ### done handling fc's reshape part ###
        
        assert op.BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions
        op_options = op.BuiltinOptions()
        fully_connected_options = FullyConnectedOptions()
        fully_connected_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = fully_connected_options.FusedActivationFunction()

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)
        weight_tensor_name = get_tensor_name(self.subgraph, weight_tensor.tensor_idx)
        weight_value = self.get_tensor_value(weight_tensor)

        self.data_dict[weight_tensor_name] = {'shape':weight_value.shape, 'dtype':weight_tensor_type_str, 'value':weight_value}
        self.param_value_dict[weight_tensor_name] = weight_value

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (quantization) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.FLOAT32)
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            bias_value = self.get_tensor_value(bias_tensor)
            bias_name = get_tensor_name(self.subgraph, bias_tensor.tensor_idx)

            self.param_value_dict[bias_name] = bias_value
            self.data_dict[bias_name] = {'shape': bias_value.shape, 'dtype': bias_tensor_type_str, 'value':bias_value}

        # Handle fused activation
        self.convert_fused_activation_function(fused_activation_fn, dict_to_be_appended)





    def convert_fused_activation_function(self, fused_activation_fn, dict_to_be_appended):
        """Convert TFLite fused activation function"""
        try:
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        inter_result = dict_to_be_appended.pop('temp_inter')  # a string to connect op and its following activation function  

        if fused_activation_fn == ActivationFunctionType.NONE:
            self.ordered_ops_list.append(dict_to_be_appended)

        if fused_activation_fn == ActivationFunctionType.RELU6:
            act_fn_out = dict_to_be_appended['out']
            dict_to_be_appended['out'] = inter_result
            self.ordered_ops_list.append(dict_to_be_appended)
            self.ordered_ops_list.append({'id':self.unique_id('RELU6'), 'tf_op_type':'RELU6', 'in':inter_result, 'out':act_fn_out, 'attrs':{}})
        
        if fused_activation_fn == ActivationFunctionType.RELU:
            act_fn_out = dict_to_be_appended['out']
            dict_to_be_appended['out'] = inter_result
            self.ordered_ops_list.append(dict_to_be_appended)
            self.ordered_ops_list.append({'id':self.unique_id('RELU'), 'tf_op_type':'RELU', 'in':inter_result, 'out':act_fn_out, 'attrs':{}})


    def conv_extractor(self, op, conv_type):
        """convolution implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.TensorType import TensorType
            from tflite.Conv2DOptions import Conv2DOptions
            from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"


        dict_to_be_appended = self.initialize_op_list_element(op)


        input_tensor = input_tensors[0]
        weight_tensor = input_tensors[1]

        # handle conv attrs
        is_depthwise_conv = False
        if conv_type == "conv2d":
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == "depthwise":
            is_depthwise_conv = True
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
            depth_multiplier = conv_options.DepthMultiplier()
        else:
            raise Exception(
                "Operator {} is not supported for frontend TFLite in our parser.".format(conv_type)
            )

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        _, input_h, input_w, input_c = to_int_list(self.get_tensor_shape(input_tensor))

        if is_depthwise_conv:
            # TFLite depthwise convolution kernel layout is:
            # 1 KH KW C(input_c * depth_multiplier)
            _, kernel_h, kernel_w, in_channels = to_int_list(self.get_tensor_shape(weight_tensor))
            assert in_channels == input_c * depth_multiplier
        else:
            output_channels, kernel_h, kernel_w, _ = to_int_list(
                self.get_tensor_shape(weight_tensor)
            )

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {
            "tf_kernel_size": [kernel_h, kernel_w],
            "tf_strides": [stride_h, stride_w],
            "tf_dilation": [dilation_h, dilation_w],
            "tf_padding": [0, 0],
            "tf_data_layout": "NHWC",
        }

        if is_depthwise_conv:
            params["tf_channels"] = int(in_channels)
            params["tf_groups"] = int(input_c)
            # If number of input channels is 1, treat as normal
            # convolution.
            params["tf_kernel_layout"] = "HWIO" if input_c == 1 else "HWOI"
        else:
            params["tf_channels"] = int(output_channels)
            params["tf_kernel_layout"] = "HWIO"

            # I add this key-value with the purpose to specify common conv2d's group num to be 1. 
            # Whether this is correct needs to be further confirmed.
            params["tf_groups"] = int(1)


        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, dilated_kernel_h, stride_h)

            pad_left, pad_right = get_pad_value(input_w, dilated_kernel_w, stride_w)
            do_pad = not (pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0)
            if do_pad:
                params["tf_padding"] = [pad_top, pad_left, pad_bottom, pad_right]

        else:
            raise Exception(
                "Padding format {} is not supported for operator Conv.".format(padding)
            )


        dict_to_be_appended['attrs'] = params
        # now all info needed to generate a node for the op are prepared in the dict. 
        # we can append this dict to ordered-ops list.


        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        weight_value = self.get_tensor_value(weight_tensor)

        # TVM implementation #
        # if is_depthwise_conv:
        #     weight_value = weight_value.reshape(kernel_h, kernel_w, input_c, depth_multiplier)
        # else:
        #     weight_value = weight_value.transpose((1, 2, 3, 0))


        # [Eason] fit weight layout to onnx data layout, that is, OIHW
        if is_depthwise_conv:
            weight_value = weight_value.transpose((3, 0, 1, 2))  
        else:
            weight_value = weight_value.transpose((0, 3, 1, 2))

        weight_name = get_tensor_name(self.subgraph, weight_tensor.tensor_idx)

        self.param_value_dict[weight_name] = weight_value
        self.data_dict[weight_name] = {'shape': weight_value.shape, 'dtype': weight_tensor_type_str, 'value':weight_value}



        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (quantization) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.FLOAT32)
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            bias_value = self.get_tensor_value(bias_tensor)
            bias_name = get_tensor_name(self.subgraph, bias_tensor.tensor_idx)

            self.param_value_dict[bias_name] = bias_value
            self.data_dict[bias_name] = {'shape': bias_value.shape, 'dtype': bias_tensor_type_str, 'value':bias_value}


        # Handle fused activation
        self.convert_fused_activation_function(fused_activation_fn, dict_to_be_appended)


        
    def get_tensor_shape(self, tensor_wrapper):
        """Returns tensor shape. Infers shape if the shape is empty."""
        assert isinstance(tensor_wrapper, TensorWrapper), "Expecting TensorWrapper here"
        if tensor_wrapper.tensor.ShapeLength() > 0:
            return tensor_wrapper.tensor.ShapeAsNumpy()

        


def to_int_list(np_array):
    """Convert a np array to a python int list.

    Note: This function converts np.int32 to python's int.
    If we don't do this conversion, numpy's automatic upcast will make
    the shape / parameters be converted to int64 IntImm in relay and
    cause problems in relay/TOPI.
    """
    return [int(x) for x in np_array]

# SAME padding: https://www.tensorflow.org/api_guides/python/nn
def get_pad_value(data, kernel, stride):
    """Get the pad tuple of value for SAME padding

    Parameters
    ----------
    data:
        1D input data

    kernel:
        1D input kernel

    stride:
        1D input stride

    Returns
    -------
        pad tuple of value
    """

    out = int(math.ceil(float(data) / float(stride)))
    pad = max(0, (out - 1) * stride + kernel - data)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after

def build_str_map(obj):
    """Build string map of TFLite enum int value

    Parameters
    ----------
    obj:
        TFLite class which contains enum int value, such as BuiltInOptions

    Returns
    -------
        String representation map of TFLite class enum int value
    """
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith("_"):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret

def get_tensor_name(subgraph, tensor_idx):
    """Get the tensor name.

    Parameters
    ----------
    subgraph:
        tflite.Subgraph.Subgraph

    tensor:
        tensor index in subgraph

    Returns
    -------
        tensor name in UTF-8 encoding
    """
    return subgraph.Tensors(tensor_idx).Name().decode("utf-8")


# def _decode_type(n):
#     _tflite_m = {
#         0: "float32",
#         1: "float16",
#         2: "int32",
#         3: "uint8",
#         4: "int64",
#         5: "string",
#         6: "bool",
#         7: "int16",
#         8: "complex64",
#         9: "int8",
#     }
#     return _tflite_m[n]

def _decode_type(n):
    _tflite_m = {
        0: np.float32,
        1: np.float16,
        2: np.int32,
        3: np.uint8,
        4: np.int64,
        # 5: "string",
        6: np.bool,
        7: np.int16,
        # 8: "complex64",
        # 9: "int8",
    }
    return _tflite_m[n]

def _input_type(model):
    subgraph_count = model.SubgraphsLength()
    assert subgraph_count > 0
    shape_dict = {}
    dtype_dict = {}
    for subgraph_index in range(subgraph_count):
        subgraph = model.Subgraphs(subgraph_index)
        inputs_count = subgraph.InputsLength()
        assert inputs_count >= 1
        for input_index in range(inputs_count):
            input_ = subgraph.Inputs(input_index)
            assert subgraph.TensorsLength() > input_
            tensor = subgraph.Tensors(input_)
            input_shape = tensor.ShapeAsNumpy()  # in NHWC format
            input_shape = tuple(input_shape[np.array([0, 3, 1, 2])])  # to NCHW format that is suitable for openvino
            tensor_type = tensor.Type()
            input_name = tensor.Name().decode("utf8")
            shape_dict[input_name] = input_shape
            dtype_dict[input_name] = _decode_type(tensor_type)

    return shape_dict, dtype_dict

def tflite_parser(model):
    try:
        import tflite.SubGraph
        import tflite.BuiltinOperator
    except ImportError:
        raise ImportError("The tflite package must be installed")

    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite

        assert isinstance(model, tflite.Model)
    except TypeError:
        import tflite.Model

    assert model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"

    subgraph = model.Subgraphs(0)
    _shape_dict, _dtype_dict = _input_type(model)  # {'input': (1, 224, 224, 3)} {'input': 'float32'}

    
    model_ops_attrs_extractor = ModelOpsAttrsExtractor(model, subgraph)

    # model inputs / outputs
    model_inputs = subgraph.InputsAsNumpy()  # [0]
    model_outputs = subgraph.OutputsAsNumpy()  # [9]

    model_input_dict = {}
    for model_input in model_inputs:
        model_input_name = get_tensor_name(subgraph, model_input)
        shape = _shape_dict[model_input_name] if model_input_name in _shape_dict else None
        dtype = _dtype_dict[model_input_name] if model_input_name in _dtype_dict else np.float32
        model_input_dict[model_input_name] = {'shape':shape, 'dtype':dtype}

    model_output_dict = {}
    for model_output in model_outputs:
        model_output_name = get_tensor_name(subgraph, model_output)
        shape = _shape_dict[model_output_name] if model_output_name in _shape_dict else None
        dtype = _dtype_dict[model_output_name] if model_output_name in _dtype_dict else np.float32
        model_output_dict[model_output_name] = {'shape':shape, 'dtype':dtype}

    ordered_ops_list, data_dict, param_value_dict = model_ops_attrs_extractor.extract_model_ops_and_attrs()
    
    return ordered_ops_list, data_dict, param_value_dict, model_input_dict, model_output_dict