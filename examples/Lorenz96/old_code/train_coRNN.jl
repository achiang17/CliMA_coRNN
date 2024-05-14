using Flux, Serialization, Statistics

F = 8.0
raw_trajectories = Serialization.deserialize("trajectories_$F.jls")
trajectories = Vector{Matrix{Float64}}(raw_trajectories)

function prepare_rnn_dataset(trajectories::Vector{Matrix{Float64}}, pred_step::Int)
    X = Vector{Matrix{Float64}}()
    Y = Vector{Vector{Float64}}()
    for traj in trajectories
        for i in 1:(size(traj, 2) - pred_step)
            push!(X, traj[:, i:(i+pred_step-1)])
            push!(Y, traj[:, i+pred_step])
        end
    end
    return X, Y
end

pred_step = 25
X_train, Y_train = prepare_rnn_dataset(trajectories[1:7], pred_step)
X_val, Y_val = prepare_rnn_dataset(trajectories[8:9], pred_step)
X_test, Y_test = prepare_rnn_dataset(trajectories[9:10], pred_step)

function to_rnn_input(X::Vector{Matrix{Float64}})
    return cat(X...; dims=3)
end

X_train = to_rnn_input(X_train)
Y_train = hcat(Y_train...)
X_val = to_rnn_input(X_val)
Y_val = hcat(Y_val...)
X_test = to_rnn_input(X_test)
Y_test = hcat(Y_test...)

function train_rnn(model, X_train, Y_train, X_val, Y_val; epochs=50, lr=0.001)
    loss(x, y) = Flux.mse(model(x), y)
    opt = Flux.ADAM(lr)
    eval_cb = throttle(() -> @info("Validation Loss: $(loss(X_val, Y_val))"), 10)

    @epochs epochs Flux.train!(loss, Flux.params(model), [(X_train, Y_train)], opt, cb=eval_cb)
    return model
end
