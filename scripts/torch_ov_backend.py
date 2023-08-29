import os
import torch
from hashlib import sha256
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.execute import execute, execute_cached
from openvino.frontend.pytorch.torchdynamo.compile import cached_model_name, cache_root_path, get_device, openvino_compile_cached_model
from openvino.runtime import Core, Type, PartialShape
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx
from scripts.ov_model_state import model_state

##hack eval_frame.py for windows support
def check_if_dynamo_supported():
    import sys
    # Skip checking for Windows support for the OpenVINO backend
    if sys.version_info >= (3, 12):
        raise RuntimeError("Python 3.12+ not yet supported for torch.compile")

torch._dynamo.eval_frame.check_if_dynamo_supported = check_if_dynamo_supported

def get_cached_file_name(*args, model_hash_str, device, cache_root):
    file_name = None
    if model_hash_str is not None:
        model_cache_dir = cache_root + "/model/"
        try:
            os.makedirs(model_cache_dir, exist_ok=True)
            file_name = model_cache_dir + model_hash_str + "_" + device
            for input_data in args:
                if file_name is not None:
                    file_name += "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "")
        except OSError as error:
            print("Cache directory ", cache_root, " cannot be created. Model caching is disabled. Error: ", error)
            file_name = None
            model_hash_str = None

    return file_name

@register_backend
@fake_tensor_unsupported
def openvino_fx(subgraph, example_inputs):
    try:
        executor_parameters = None
        inputs_reversed = False
        if os.getenv("OPENVINO_TORCH_MODEL_CACHING") is not None:
            # Create a hash to be used for caching
            model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest() + "_" + model_state["model_hash"]
            executor_parameters = {"model_hash_str": model_hash_str}
            # Check if the model was fully supported and already cached
            example_inputs.reverse()
            inputs_reversed = True
            maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", get_device(), example_inputs, cache_root_path())
            print(maybe_fs_cached_name)
            if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
                # Model is fully supported and already cached. Run the cached OV model directly.
                compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, *example_inputs)
                def _call(*args):
                    res = execute_cached(compiled_model, *args)
                    return res
                return _call
        print("First Time Model Compilation-----")
        if inputs_reversed:
            example_inputs.reverse()
        model = make_fx(subgraph)(*example_inputs)
        with torch.no_grad():
            model.eval()
        partitioner = Partitioner()
        compiled_model = partitioner.make_partitions(model)

        if executor_parameters is not None and 'model_hash_str' in executor_parameters:
            # Check if the model is fully supported.
            fully_supported = partitioner.check_fully_supported(compiled_model)
            if fully_supported:
                executor_parameters["model_hash_str"] += "_fs"

        def _call(*args):
            res = execute(compiled_model, *args, executor="openvino",
                          executor_parameters=executor_parameters)
            return res
        return _call
    except Exception as e:
        print(f"Failed in OpenVINO execution: {e}")
        return compile_fx(subgraph, example_inputs)



def openvino_fx_bak(subgraph, example_inputs):
    print(model_state)
    try:
        executor_parameters = None
        core = Core()
        if os.getenv("OPENVINO_TORCH_MODEL_CACHING") != "0":
            model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest() + "_" + model_state["model_hash"]
            #model_hash_str_file = model_hash_str + str(model_state["partition_id"])
            model_hash_str_file = model_hash_str + "_fs"
            #model_state["partition_id"] = model_state["partition_id"] + 1
            executor_parameters = {"model_hash_str": model_hash_str}

        example_inputs.reverse()
        cache_root = "./cache/"
        if os.getenv("OPENVINO_TORCH_CACHE_DIR") is not None:
            cache_root = os.getenv("OPENVINO_TORCH_CACHE_DIR")

        # device = "CPU"
        device = "CPU"

        if os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is not None:
            device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")
            assert device in core.available_devices, "Specified device " + device + " is not in the list of OpenVINO Available Devices"

        file_name = get_cached_file_name(*example_inputs, model_hash_str=model_hash_str_file, device=device, cache_root=cache_root)

        print(file_name)

        if file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin"):
            print("reading model from xml")
            om = core.read_model(file_name + ".xml")

            dtype_mapping = {
                torch.float32: Type.f32,
                torch.float64: Type.f64,
                torch.float16: Type.f16,
                torch.int64: Type.i64,
                torch.int32: Type.i32,
                torch.uint8: Type.u8,
                torch.int8: Type.i8,
                torch.bool: Type.boolean
            }

            for idx, input_data in enumerate(example_inputs):
                om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
                om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
            om.validate_nodes_and_infer_types()

            if model_hash_str is not None:
                core.set_property({'CACHE_DIR': cache_root + '/blob'})

            compiled_model = core.compile_model(om, device)
            def _call(*args):
                ov_inputs = [a.detach().cpu().numpy() for a in args]
                ov_inputs.reverse()
                res = compiled_model(ov_inputs)
                result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
                return result
            return _call
        else:
            print("First Time Model Compilation------------")
            example_inputs.reverse()
            model = make_fx(subgraph)(*example_inputs)
            with torch.no_grad():
                model.eval()
            partitioner = Partitioner()
            compiled_model = partitioner.make_partitions(model)
            if executor_parameters is not None and 'model_hash_str' in executor_parameters:
                # Check if the model is fully supported.
                fully_supported = partitioner.check_fully_supported(compiled_model)
            if fully_supported:
                executor_parameters["model_hash_str"] += "_fs"
            def _call(*args):
                res = execute(compiled_model, *args, executor="openvino",
                              executor_parameters=executor_parameters)
                return res
            return _call
    except Exception as error:
        print("exception section" + str(error))
        return compile_fx(subgraph, example_inputs)





