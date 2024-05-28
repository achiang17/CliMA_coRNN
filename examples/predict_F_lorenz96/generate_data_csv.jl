using SciMLBase, OrdinaryDiffEq, DataFrames, CSV, Random

function lorenz96(du,u,p,t)
    n = length(u)
    F = p
    for j in 1:n
        du[j] = (u[mod1(j+1, n)] - u[mod1(j-2, n)]) * u[mod1(j-1, n)] - u[j] + F
    end
end

n = 5
dt = 0.01
trajectory_length = 2000
tspan = (0.0, trajectory_length * dt)
num_csvs = 2000

groups = ["train", "test"]
for group in groups
    local labels = []
    for i in 1:num_csvs
        F = 0.9 + (8.0 - 0.9) * rand()
        F = round(F, digits=2)
        push!(labels, F)
        u0 = F .+ 0.5 .* (2 .* rand(n) .- 1)
        prob = ODEProblem(lorenz96,u0,tspan,F)
        sol = solve(prob,Euler(),dt=dt)
        df = DataFrame(sol.u, :auto)
        CSV.write("$(group)/$(group)_$(i).csv", df)
    end
    df_F = DataFrame(F_vals = labels)
    CSV.write("$(group)/labels.csv", df_F)
end