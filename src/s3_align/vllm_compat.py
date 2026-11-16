"""vLLM0.9.2 NVML UUID mapping fix; retain CUDA_VISIBLE_DEVICES UUIDs.

Upstream Platform.device_id_to_physical_device_id calls int() on GPU UUIDs.
Resolve only this NVML physical-index lookup, without altering CUDA visibility,
model code, generation or scoring. Remove when upgrading to native UUID support.
"""
import os


def install():
    from vllm.platforms.interface import Platform
    if getattr(Platform,'_qwen3_uuid_lookup_fixed',False):return
    original=Platform.device_id_to_physical_device_id.__func__
    @classmethod
    def resolve(cls,device_id):
        visible=os.environ.get(cls.device_control_env_var,'')
        if visible and visible.split(',')[device_id].startswith('GPU-'):
            from vllm.utils import import_pynvml
            pynvml=import_pynvml()
            pynvml.nvmlInit()
            try:
                handle=pynvml.nvmlDeviceGetHandleByUUID(visible.split(',')[device_id])
                return pynvml.nvmlDeviceGetIndex(handle)
            finally:pynvml.nvmlShutdown()
        return original(cls,device_id)
    Platform.device_id_to_physical_device_id=resolve
    Platform._qwen3_uuid_lookup_fixed=True
