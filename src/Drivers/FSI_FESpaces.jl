function get_FE_spaces(
  strategy::MeshStrategy,
  coupling::Coupling{:strong},
  Tₕ,
  order,
  bconds;
  constraint=nothing)

  # Reference FE
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

  # Test FE Spaces
  Vu_FSI = TestFESpace(Tₕ[:Ω], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vu_tags])
  Vv_FSI = TestFESpace(Tₕ[:Ω], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vv_tags])
  Vu_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vu_tags])
  Vv_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vv_tags])
  Q = TestFESpace(Tₕ[:Ωf], reffeₚ, constraint=constraint, conformity=:C0)

  # Trial FE Spaces
  Uu_ST = TrialFESpace(Vu_ST,bconds[:ST_Vu_values])
  Uv_ST = TrialFESpace(Vv_ST,bconds[:ST_Vv_values])
  Uu_FSI = TransientTrialFESpace(Vu_FSI,bconds[:FSI_Vu_values])
  Uv_FSI = TransientTrialFESpace(Vv_FSI,bconds[:FSI_Vv_values])
  P = TrialFESpace(Q)

  # Multifield FE Spaces
  fe_spaces = (
    Y_ST = MultiFieldFESpace([Vu_ST,Vv_ST,Q]),
    X_ST = MultiFieldFESpace([Uu_ST,Uv_ST,P]),
    Y_FSI = MultiFieldFESpace([Vu_FSI,Vv_FSI,Q]),
    X_FSI = MultiFieldFESpace([Uu_FSI,Uv_FSI,P])
  )
end

function get_FE_spaces(
  strategy::MeshStrategy{:biharmonic},
  coupling::Coupling{:strong},
  Tₕ,
  order,
  bconds;
  constraint=nothing)

  # Reference FE
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

  # Test FE Spaces
  Vw_FSI = TestFESpace(Tₕ[:Ω], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vu_tags])
  Vu_FSI = TestFESpace(Tₕ[:Ω], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vu_tags])
  Vv_FSI = TestFESpace(Tₕ[:Ω], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vv_tags])
  Vw_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vu_tags])
  Vu_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vu_tags])
  Vv_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vv_tags])
  Q = TestFESpace(Tₕ[:Ωf], reffeₚ, constraint=constraint, conformity=:C0)

  # Trial FE Spaces
  Uw_ST = TrialFESpace(Vu_ST,bconds[:ST_Vu_values])
  Uu_ST = TrialFESpace(Vu_ST,bconds[:ST_Vu_values])
  Uv_ST = TrialFESpace(Vv_ST,bconds[:ST_Vv_values])
  Uw_FSI = TrialFESpace(Vu_FSI,bconds[:ST_Vu_values])
  Uu_FSI = TransientTrialFESpace(Vu_FSI,bconds[:FSI_Vu_values])
  Uv_FSI = TransientTrialFESpace(Vv_FSI,bconds[:FSI_Vv_values])
  P = TrialFESpace(Q)

  # Multifield FE Spaces
  fe_spaces = (
    Y_ST = MultiFieldFESpace([Vw_ST,Vu_ST,Vv_ST,Q]),
    X_ST = MultiFieldFESpace([Uw_ST,Uu_ST,Uv_ST,P]),
    Y_FSI = MultiFieldFESpace([Vw_FSI,Vu_FSI,Vv_FSI,Q]),
    X_FSI = MultiFieldFESpace([Uw_FSI,Uu_FSI,Uv_FSI,P])
  )
end

function get_FE_spaces(
  strategy::MeshStrategy,
  coupling::Coupling{:weak},
  Tₕ,
  order,
  bconds;
  constraint=nothing)

  # Reference FE
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

  # Test FE Spaces
  Vu_FSI_f = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vu_f_tags])
  Vv_FSI_f = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vv_f_tags])
  Vu_FSI_s = TestFESpace(Tₕ[:Ωs], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vu_s_tags])
  Vv_FSI_s = TestFESpace(Tₕ[:Ωs], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vv_s_tags])
  Vu_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vu_tags])
  Vv_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vv_tags])
  Q = TestFESpace(Tₕ[:Ωf], reffeₚ, constraint=constraint, conformity=:C0)

  # Trial FE Spaces
  Uu_ST = TrialFESpace(Vu_ST,bconds[:ST_Vu_values])
  Uv_ST = TrialFESpace(Vv_ST,bconds[:ST_Vv_values])
  Uu_FSI_f = TransientTrialFESpace(Vu_FSI_f,bconds[:FSI_Vu_f_values])
  Uv_FSI_f = TransientTrialFESpace(Vv_FSI_f,bconds[:FSI_Vv_f_values])
  Uu_FSI_s = TransientTrialFESpace(Vu_FSI_s,bconds[:FSI_Vu_s_values])
  Uv_FSI_s = TransientTrialFESpace(Vv_FSI_s,bconds[:FSI_Vv_s_values])
  P = TrialFESpace(Q)

  # Multifield FE Spaces
  fe_spaces = (
    Y_ST = MultiFieldFESpace([Vu_ST,Vv_ST,Q]),
    X_ST = MultiFieldFESpace([Uu_ST,Uv_ST,P]),
    Y_FSI = MultiFieldFESpace([Vu_FSI_f,Vv_FSI_f,Q,Vu_FSI_s,Vv_FSI_s]),
    X_FSI = MultiFieldFESpace([Uu_FSI_f,Uv_FSI_f,P,Uu_FSI_s,Uv_FSI_s])
  )
end

function get_FE_spaces(
  strategy::MeshStrategy{:biharmonic},
  coupling::Coupling{:weak},
  Tₕ,
  order,
  bconds;
  constraint=nothing)

  # Reference FE
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffeₚ = ReferenceFE(lagrangian,Float64,order-1)

  # Test FE Spaces
  Vw_FSI_f = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vw_f_tags])
  Vu_FSI_f = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vu_f_tags])
  Vv_FSI_f = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vv_f_tags])
  Vu_FSI_s = TestFESpace(Tₕ[:Ωs], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vu_s_tags])
  Vv_FSI_s = TestFESpace(Tₕ[:Ωs], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:FSI_Vv_s_tags])
  Vw_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vu_tags])
  Vu_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vu_tags])
  Vv_ST = TestFESpace(Tₕ[:Ωf], reffeᵤ, conformity=:H1, dirichlet_tags=bconds[:ST_Vv_tags])
  Q = TestFESpace(Tₕ[:Ωf], reffeₚ, constraint=constraint, conformity=:C0)

  # Trial FE Spaces
  Uw_ST = TrialFESpace(Vu_ST,bconds[:ST_Vu_values])
  Uu_ST = TrialFESpace(Vu_ST,bconds[:ST_Vu_values])
  Uv_ST = TrialFESpace(Vv_ST,bconds[:ST_Vv_values])
  Uw_FSI_f = TrialFESpace(Vw_FSI_f,bconds[:FSI_Vw_f_values])
  Uu_FSI_f = TransientTrialFESpace(Vu_FSI_f,bconds[:FSI_Vu_f_values])
  Uv_FSI_f = TransientTrialFESpace(Vv_FSI_f,bconds[:FSI_Vv_f_values])
  Uu_FSI_s = TransientTrialFESpace(Vu_FSI_s,bconds[:FSI_Vu_s_values])
  Uv_FSI_s = TransientTrialFESpace(Vv_FSI_s,bconds[:FSI_Vv_s_values])
  P = TrialFESpace(Q)

  # Multifield FE Spaces
  fe_spaces = (
    Y_ST = MultiFieldFESpace([Vw_ST,Vu_ST,Vv_ST,Q]),
    X_ST = MultiFieldFESpace([Uw_ST,Uu_ST,Uv_ST,P]),
    Y_FSI = MultiFieldFESpace([Vw_FSI_f,Vu_FSI_f,Vv_FSI_f,Q,Vu_FSI_s,Vv_FSI_s]),
    X_FSI = MultiFieldFESpace([Uw_FSI_f,Uu_FSI_f,Uv_FSI_f,P,Uu_FSI_s,Uv_FSI_s])
  )
end
