# data_generation.jl

using SciMLBase, OrdinaryDiffEq, Random, Serialization

function lorenz96(du, u, p, t)
    n = length(u)
    F = p
    for j in 1:n
        du[j] = (u[mod1(j+1, n)] - u[mod1(j-2, n)]) * u[mod1(j-1, n)] - u[j] + F
    end
end

function generate_trajectories(F::Float64, num_trajectories::Int, trajectory_length::Int; dt=0.01)
    trajectories = Vector{Matrix{Float64}}()  # Explicitly define the type here
    Random.seed!(123)
    for _ in 1:num_trajectories
        u0 = F .+ 0.5 .* (2 .* rand(5) .- 1)
        tspan = (0.0, trajectory_length * dt)
        prob = ODEProblem(lorenz96, u0, tspan, F)
        sol = solve(prob, Euler(), dt=dt)
        push!(trajectories, Array(sol))
    end
    return trajectories
end

# Example to generate and save trajectories to a file
F = 8.0
num_trajectories = 10
trajectory_length = 200
trajectories = generate_trajectories(F, num_trajectories, trajectory_length)

# Save to file
Serialization.serialize("trajectories_$F.jls", trajectories)
