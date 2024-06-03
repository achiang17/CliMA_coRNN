using SciMLBase, OrdinaryDiffEq, Plots
gr()

function lorenz96(du,u,p,t)
    n = length(u)
    F = p
    for j in 1:n
        du[j] = (u[mod1(j+1, n)] - u[mod1(j-2, n)]) * u[mod1(j-1, n)] - u[j] + F
    end
end

n = 5
F_vals = [0.9, 8]
tspan = (0.0,20.0)

for F in F_vals
    local u0 = F .+ 0.5 .* (2 .* rand(n) .- 1)
    local prob = ODEProblem(lorenz96,u0,tspan,F)
    local sol = solve(prob,Euler(),dt=0.01)
    plot(sol,idxs=(1,2), label="F = $F", title="Lorenz 96 Model Trajectory")
    savefig("Lorenz96_trajectories_$(F).png") 
end