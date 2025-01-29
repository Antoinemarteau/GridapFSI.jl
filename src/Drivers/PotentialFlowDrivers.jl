function execute(problem::PotentialFlowProblem{:analytical};kwargs...)

  # Parameters
  L = 2*π
  H = 1.0
  n = 20
  order = 2
  g = 9.81
  ξ = 0.1
  λ = L/2
  k = 2*π/L
  h = L/n
  ω = √(g*k*tanh(k*H))
  t₀ = 0.0
  tf = 8*π
  Δt = h/(10*λ*ω)
  θ = 0.5

  # Exact solution
  ϕₑ(x,t) = ω/k * ξ * (cosh(k*(x[2]))) / sinh(k*H) * sin(k*x[1] - ω*t)
  ηₑ(x,t) = ξ * cos(k*x[1] - ω*t)
  ϕₑ(t::Real) = x -> ϕₑ(x,t)
  ηₑ(t::Real) = x -> ηₑ(x,t)

  # Domain
  domain = (0,L,0,H)
  partition = (n,n)
  model = CartesianDiscreteModel(domain,partition;isperiodic=(true,false))

  # Boundaries
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"bottom",[1,2,5])
  add_tag_from_tags!(labels,"free_surface",[3,4,6])
  bgface_to_mask = get_face_mask(labels,[3,4,6],1)
  model_Γ =BoundaryDiscreteModel(Polytope{1},model,bgface_to_mask)

  # Triangulation
  Ω = Triangulation(model)
  Γ = Triangulation(model_Γ)
  dΩ = Measure(Ω,2*order)
  dΓ = Measure(Γ,2*order)

  # FE spaces
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe,conformity=:H1)
  V_Γ = TestFESpace(model_Γ,reffe,conformity=:H1)
  U = TransientTrialFESpace(V)
  U_Γ = TransientTrialFESpace(V_Γ)
  X = MultiFieldFESpace([U,U_Γ])
  Y = MultiFieldFESpace([V,V_Γ])

  # Weak form
  α = 2/Δt
  # mass
  m(t,(ϕt,ηt),(w,v)) = ∫( 0.5*(α/g*(w*ϕt) + v*ϕt) - (w*ηt) )dΓ
  # stiffness
  a(t,(ϕ,η),(w,v)) = ∫( ∇(ϕ)⋅∇(w) )dΩ + ∫( 0.5*(α*(w*η) + g*v*η) )dΓ
  # residual
  b(t,(ϕ,η),(w,v)) = ∫( 0.0*w )dΓ
  # TODO check that the update to Gridap 0.1.8 below is correct:
  op = TransientLinearFEOperator(a,m,b,X,Y; constant_forms=(true,true))
  # op = TransientConstantFEOperator(m,a,b,X,Y) # <= Gridap 0.1.7; GridapODEs

  # Solver
  ls = LUSolver()
  solver = ThetaMethod(ls,Δt,θ)

  # Initial solution
  x₀ = interpolate_everywhere([ϕₑ(0.0),ηₑ(0.0)],X(0.0))

  # Solution
  sol_t = solve(solver,op,t₀,tf,x₀)

  # Post-process
  l2_Ω(w) = √(∑(∫(w*w)dΩ))
  l2_Γ(v) = √(∑(∫(v*v)dΓ))
  E_kin(w) = 0.5*∑( ∫(∇(w)⋅∇(w))dΩ )
  E_pot(v) = g*0.5*∑( ∫(v*v)dΓ )
  Eₑ = 0.5*g*ξ^2*L

  folderName = "ϕFlow-results"
  if !isdir(folderName)
    mkdir(folderName)
  end
  filePath_Ω = folderName * "/fields_Ω"
  filePath_Γ = folderName * "/fields_Γ"
  pvd_Ω = paraview_collection(filePath_Ω, append=false)
  pvd_Γ = paraview_collection(filePath_Γ, append=false)
  for ((ϕn,ηn),tn) in sol_t
    E = E_kin(ϕn) + E_pot(ηn)
    error_ϕ = l2_Ω(ϕn-ϕₑ(tn))
    error_η = l2_Γ(ηn-ηₑ(tn))
    #println(E/Eₑ," ", error_ϕ," ",error_η)

    pvd_Ω[tn] = createvtk(Ω,filePath_Ω * "_$tn.vtu",cellfields = ["phi" => ϕn])
    pvd_Γ[tn] = createvtk(Γ,filePath_Γ * "_$tn.vtu",cellfields = ["eta" => ηn])
  end
  vtk_save(pvd_Ω)
  vtk_save(pvd_Γ)
end

include("PotentialFlowBeam.jl")
