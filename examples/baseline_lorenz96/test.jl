using SciMLBase, OrdinaryDiffEq
gr()

function lorenz96(du,u,p,t)
    n = length(u)
    F = p
    for j in 1:n
        du[j] = (u[mod1(j+1, n)] - u[mod1(j-2, n)]) * u[mod1(j-1, n)] - u[j] + F
    end
end

n = 5
F_vals = 0.9
tspan = (0.0,20.0)

u0 = F .+ 0.5 .* (2 .* rand(n) .- 1)
prob = ODEProblem(lorenz96,u0,tspan,F)
sol = solve(prob,Euler(),dt=0.01)
print(sol.u)
xs = [point[1] for point in sol.u]
ys = [point[2] for point in sol.u]

scatter(xs,ys)

#plot(sol,idxs=(1,2), label="F = $F", title="Lorenz 96 Model Trajectory")
savefig("test.png") 
