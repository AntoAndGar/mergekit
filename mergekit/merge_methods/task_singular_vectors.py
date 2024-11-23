import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod
from mergekit.merge_methods.generalized_task_arithmetic import get_task_vectors


class TSVMerge(MergeMethod, BaseModel, frozen=True):
    default_normalize: bool
    default_rescale: bool

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(
                name="normalize", required=False, default_value=self.default_normalize
            ),
            ConfigParameterDef(
                name="rescale", required=False, default_value=self.default_rescale
            ),
            ConfigParameterDef(name="probabilistic", required=True, default_value=True),
            ConfigParameterDef(name="sv_reduction", required=False, default_value=0.2),
            ConfigParameterDef(name="num_iterations", required=False, default_value=2),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        res = [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
        ]

        return res

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        return TaskSingularVector(
            method=self,
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            normalize=parameters["normalize"],
            rescale=parameters["rescale"],
            weight_info=output_weight,
            sv_reduction=parameters["sv_reduction"],
            num_iterations=parameters["num_iterations"],
            probabilistic=parameters["probabilistic"],
        )


class TaskSingularVector(Task[torch.Tensor]):
    method: TSVMerge
    tensors: GatherTensors
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, Any]
    normalize: bool
    rescale: bool
    sv_reduction: float
    num_iterations: int
    probabilistic: bool

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base = get_task_vectors(
            self.weight_info,
            self.base_model,
            tensors,
            tensor_parameters=self.tensor_parameters,
        )

        if not tvs:
            return base

        # svd_dict = {}
        for i, tv in enumerate(tvs):
            # svd_dict[tv["model"]] = {}
            device = tv["delta"].device
            vec = tv["delta"].float().to(device)
            if tv["delta"].dim() == 2:
                # dtype = tv["delta"].dtype
                if self.probabilistic:
                    u, s, v = torch.svd_lowrank(
                        vec,
                        q=min(tv["delta"].shape[0], tv["delta"].shape[1]),
                        niter=self.num_iterations,
                    )
                    v = v.mT
                else:
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                # not performed here because done at the end but could be done here if needed
                # u = u.to(dtype)
                # s = s.to(dtype)
                # v = v.to(dtype)

                # reduced_index_s = int(s.shape[0] * self.sv_reduction)

                # temp_u = torch.zeros_like(u, device=device)
                # # select only the first reduced_index_s columns of u and place them
                # temp_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                #     :, :reduced_index_s
                # ]
                # svd_dict[tv["model"]]["u"] = temp_u

                # temp_s = torch.zeros_like(s, device=device)
                # temp_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                #     :reduced_index_s
                # ]
                # svd_dict[tv["model"]]["s"] = temp_s  # s_reduced

                # # select only the first reduced_index_s rows of v and place them
                # temp_v = torch.zeros_like(v, device=device)
                # temp_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                #     :reduced_index_s, :
                # ]
                # svd_dict[tv["model"]]["v"] =
                if i == 0:
                    sum_u = torch.zeros_like(u, device=device)
                    sum_s = torch.zeros_like(s, device=device)
                    sum_v = torch.zeros_like(v, device=device)
                reduced_index_s = int(s.shape[0] * self.sv_reduction)

                # select only the first reduced_index_s columns of u and place them
                sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                    :, :reduced_index_s
                ]
                sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                    :reduced_index_s
                ]
                # select only the first reduced_index_s rows of v and place them
                sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                    :reduced_index_s, :
                ]
            else:
                if i == 0:
                    mixed_delta = vec.clone()
                else:
                    mixed_delta += (vec - mixed_delta) / (i + 1)
                # svd_dict[tv["model"]]["dim1"] = tv["delta"]

        if tv["delta"].dim() == 2:
            # sum_u = sum([svd_dict[tv["model"]]["u"] for tv in tvs])
            # sum_s = sum([svd_dict[tv["model"]]["s"] for tv in tvs])
            # sum_v = sum([svd_dict[tv["model"]]["v"] for tv in tvs])
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            mixed_delta = torch.linalg.multi_dot(
                (
                    u_u,
                    v_u,
                    torch.diag_embed(sum_s),
                    u_v,
                    v_v,
                )
            )
        # else:
        #     # calculate running mean
        #     for i, tv in enumerate(tvs, start=1):
        #         if i == 1:
        #             mixed_delta = svd_dict[tv["model"]]["dim1"]
        #         else:
        #             mixed_delta += (svd_dict[tv["model"]]["dim1"] - mixed_delta) / i

        return (base + mixed_delta).to(base.dtype)

    def group_label(self) -> Optional[str]:
        return self.tensors.group_label()
