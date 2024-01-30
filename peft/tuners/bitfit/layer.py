# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn


class BitFitLayer:
    # indicates whether all adapters should be disabled
    _disable_adapters: bool = False
    
    # the currently active adapters(s)
    _active_adapter: Union[str, List[str]] = "default"
    
    # save the original bias
    _bias_original = None
    
    def __init__(self, base_layer: nn.Module, **kwargs):
        self.bitfit_base_layer = base_layer
        self._disable_adapters = False
        self.kwargs = kwargs
        # For bias
        self.bias_adapters = nn.ParameterDict({})
        
        # TODO: we have to implement the layer that don't have register_parameter method 
        # such as T5LayerNorm. Later, Implement the customization, then also complete adapter_on/off
        if isinstance(base_layer, nn.Linear):
            out_features = base_layer.out_features
        elif isinstance(base_layer, nn.LayerNorm):
            out_features = base_layer.normalized_shape[0]
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        
        if hasattr(base_layer, "bias") and base_layer.bias is not None:
            self.bias_original = base_layer.bias
        
        self.out_features = out_features

    def set_adapter(self, adapter_names: Union[str, List[str]]) -> None:
        """
        Set the active adapter(s).

        Args:
            adapter_names (`str` or `List[str]`): Name of the adapter(s) to be activated.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        
        # deactivate grads on the inactive adapter
        for key, bias in self.bias_adapters.items():
            if key in adapter_names:
                # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                # happen if a completely different adapter layer is being activated.
                bias.requires_grads_(True)
            else:
                bias.requires_grad_(False)
        
        self._active_adapter = adapter_names

    def enable_adapter(self, enabled: bool, adapter_name: str = "default") -> None:
        """Toggle the enableing and disabling of adapters
        
        Takes care of setting the requires_grad flag for the adapter weights
        
        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if enabled:
            self.set_adapter(active_adapter)
            self.adapter_on(active_adapter)
            self._disable_adapters = False
        else:
            # disable grads on all adapter layer
            self.bias_adapters.requires_grad_(False)
            self.adapter_off()
            self._disable_adapters = True

    def update_layer(self, adapter_name):
        # Add trainable parameter as bias
        if self.bias_original is not None:
            self.bias_adapters[adapter_name] = copy.deepcopy(bias_original)
        else:
            self.bias_adapters[adapter_name] = nn.Parameter(torch.zeros(self.out_features))
        
        weight = getattr(self.bitfit_base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(self.weight.device)
        
        self.set_adapter(self.active_adapter)

    # Adapter (bias) on base layer
    def adapter_on(self, adapter_name):
        if has_register():
            self.bitfit_base_layer.register_parameter("bias", self.bias_adapters[adapter_name])
        else:
            raise ValueError(f"Unsupported layer type {type(bitfit_base_layer)}")

    # Adapter (bias) off base layer. replace adapter to original bias
    def adapter_off(self):
        if has_register():
            self.bitfit_base_layer.register_parameter("bias", self.bias_original)
        else:
            raise ValueError(f"Unsupported layer type {type(bitfit_base_layer)}")

    @property
    def has_register(self) -> bool:
        # if layer has register_parameter
        return hasattr(self.bitfit_base_layer, "register_parameter")

    @property
    def disable_adapter(self) -> bool:
        # use a property to ensure that disable_adapter is not set directly, instead use the enable_adapters
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter
        return self._active_adapter

    @property
    def active_adapters(self):
        if isinstance(self.active_adapter, str):
            return [self.active_adapter]
        # is already a list of str
        return self.active_adapter

    @property
    def bias_original(self):
        return self._bias_original


class BitFitModule(nn.Module, BitFitLayer):
    # BitFit implemented in a dense layer
    def __init__(self, base_layer, adapter_name: str, **kwargs):
        super().__init__()
        BitFitLayer.__init__(self, base_layer=base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name)
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.bitfit_base_layer(x, *args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "bitfit." + rep
