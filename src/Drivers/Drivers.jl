module Drivers

# Julia modules used in the drivers
using Gridap
using Gridap.Helpers
using Gridap.Geometry
using Gridap.Arrays
using Gridap.CellData
using Gridap.MultiField: ConsecutiveMultiFieldStyle
using GridapFSI.WeakForms
using TimerOutputs
using WriteVTK
using LineSearches: BackTracking, HagerZhang
using ForwardDiff
using Test

using .WeakForms: lame_parameters
using .WeakForms: MeshStrategy
using .WeakForms: Coupling
using .WeakForms: residual_Ωf, residual_Ωs, residual_Γi
using .WeakForms: jacobian_Ωf, jacobian_Ωs, jacobian_Γi
using .WeakForms: jacobian_t_Ωf, jacobian_t_Ωs

export Problem
export FSIProblem
export PotentialFlowProblem

export execute
export get_problem

abstract type Problem end
struct FSIProblem{Kind} <: Problem end
struct PotentialFlowProblem{Kind} <: Problem end

function get_problem(problemName::String, kwargs)
  ptype = _get_kwarg(:ptype,kwargs,"FSI")
  if ptype == "FSI"
    return FSIProblem{Symbol(problemName)}()
  elseif ptype == "PotentialFlow"
    return PotentialFlowProblem{Symbol(problemName)}()
  else
    @notimplemented("The problem type: $ptype is not implemented")
  end
end

execute(problem::Problem; kwargs...) = @notimplemented("The driver for problem: $problem is not implemented")

function _get_kwarg(kwarg,kwargs)
  try
      return kwargs[kwarg]
  catch
      s = "The key-word argument $(kwarg) is mandatory in the $problem driver"
      error(s)
  end
end

function _get_kwarg(kwarg,kwargs,value)
  try
      return kwargs[kwarg]
  catch
      return value
  end
end

include("FSIDrivers.jl")
include("PotentialFlowDrivers.jl")

end
