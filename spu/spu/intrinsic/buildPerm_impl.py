__all__ = ["buildPerm"]

from functools import partial

from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla

# from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# Public facing interface
def buildPerm(in1, in2):
    # Add necessary preprocessing code
    return _buildPerm_prim.bind(in1, in2)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _buildPerm_abstract(in1, in2):
    shape = in1.shape
    dtype = dtypes.canonicalize_dtype(in1.dtype)
    return ShapedArray(shape, dtype)


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
def _buildPerm_lowering(ctx, in1, in2):
    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(in1.type)
    shape = dtype.shape
    result_type = mlir.ir.RankedTensorType.get(shape, dtype.element_type)

    return custom_call(
        "buildPerm",
        # Output types
        result_types=[result_type],
        # The inputs:
        operands=[in1, in2],
    ).results


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************


def _buildPerm_jvp(args, tangents):
    raise NotImplementedError()


# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************


# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
def _buildPerm_batch(args, axes):
    raise NotImplementedError()


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_buildPerm_prim = core.Primitive("buildPerm")
# Change this to True if there are more than 1 output
_buildPerm_prim.multiple_results = False
_buildPerm_prim.def_impl(partial(xla.apply_primitive, _buildPerm_prim))
_buildPerm_prim.def_abstract_eval(_buildPerm_abstract)

mlir.register_lowering(_buildPerm_prim, _buildPerm_lowering)

# Connect the JVP and batching rules
ad.primitive_jvps[_buildPerm_prim] = _buildPerm_jvp
batching.primitive_batchers[_buildPerm_prim] = _buildPerm_batch
