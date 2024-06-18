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
num_csvs = 10
F_vals = 0.9:0.05:8.05

global group_label = 0

for F in F_vals
    if mod(group_label,2) == 0
        local group = "basin_split_train"
    else
        local group = "basin_split_test"
    end
    global group_label += 1

    for i in 1:num_csvs
        u0 = F .+ 0.5 .* (2 .* rand(n) .- 1)
        prob = ODEProblem(lorenz96,u0,tspan,F)
        sol = solve(prob,Euler(),dt=dt)
        df = DataFrame(sol.u, :auto)
        CSV.write("$(group)/$(F)_$(i).csv", df)
        CSV.write("time_split/$(F)_$(i).csv", df)
    end
end