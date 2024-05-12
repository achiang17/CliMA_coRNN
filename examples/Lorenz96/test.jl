using SciMLBase, OrdinaryDiffEq, Plots, DataFrames, CSV
gr()

function lorenz96(du,u,p,t)
    n = length(u)
    F = p
    for j in 1:n
        du[j] = (u[mod1(j+1, n)] - u[mod1(j-2, n)]) * u[mod1(j-1, n)] - u[j] + F
    end
end

n = 5
F = 0.9
dt  = 0.01
trajectory_length = 10
tspan = (0.0, trajectory_length * dt)

u0 = F .+ 0.5 .* (2 .* rand(n) .- 1)
prob = ODEProblem(lorenz96,u0,tspan,F)
sol = solve(prob,Euler(),dt=dt)
df = DataFrame(sol.u, :auto)
CSV.write("test.csv", df)

#in python can use pd.read_csv() and the df.transpose()
