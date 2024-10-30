using AdvancedVI
using BlackBIRDS
using DiffABM
using DifferentiationInterface
using ForwardDiff
using Functors
using Optimisers
using ProgressMeter
using StochasticAD
using PyPlot
using Flux
using Random
using Zygote

##
@functor DiffABM.RandomAxtellAgentInitializer ()
@functor AxtellFirmsParams (utility_function,)
function make_abm_params_nn(nn)
    a = 1.5
    b = 3.0
    n_agents = 1000
    n_timesteps = 20
    neighbors = [sample(1:n_agents, rand(2:6), replace = false) for _ in 1:n_agents]
    utility_function = NeuralNetworkUtility(nn)
    thetas_bounds = [0.0, 1.0]
    initial_efforts_bounds = [0.0, 1.0]
    agent_initializer = DiffABM.RandomAxtellAgentInitializer(
        n_agents, thetas_bounds, initial_efforts_bounds, neighbors)
    gradient_horizon = 200
    return DiffABM.AxtellFirmsParams(
        agent_initializer, utility_function, [a], [b], 0.5, 1.0, n_timesteps, gradient_horizon)
end
function generate_true_data()
    a = 1.5
    b = 3.0
    n_agents = 1000
    n_timesteps = 20
    neighbors = [sample(1:n_agents, rand(2:6), replace = false) for _ in 1:n_agents]
    utility_function = CobbDouglasUtility()
    thetas_bounds = [0.0, 1.0]
    initial_efforts_bounds = [0.0, 1.0]
    agent_initializer = DiffABM.RandomAxtellAgentInitializer(
        n_agents, thetas_bounds, initial_efforts_bounds, neighbors)
    gradient_horizon = 200
    params = DiffABM.AxtellFirmsParams(
        agent_initializer, utility_function, [a], [b], 0.5, 1.0, n_timesteps, gradient_horizon)
    return abm_run(params)
end
nn = Flux.Chain(
    Flux.Dense(5, 32, relu),
    Flux.Dense(32,32, relu),
    Flux.Dense(32, 1, relu),
    #x -> 20 .* Flux.sigmoid(x)
) 
nn = fmap(f64, nn)
data = generate_true_data()
chunksize = length(Flux.destructure(nn)[1])
abm = ABM(make_abm_params_nn(nn), AutoForwardDiff(chunksize=chunksize), GaussianMMDLoss(1.0, 5))

fig, ax = plt.subplots(1, 3, figsize = (12, 4))
pred_nn = rand(abm)
ax[1].plot(pred_nn[1, :], label = "nn")
ax[1].set_ylim(0, 0.6)
ax[2].plot(pred_nn[2, :], label = "nn")
ax[3].plot(pred_nn[3, :], label = "nn")
ax[1].plot(data[1, :], color = "black", label = "data")
ax[1].set_ylabel("Mean Agent Effort")
ax[2].plot(data[2, :], color = "black", label = "data")
ax[2].set_ylabel("Mean Firm Output")
ax[3].plot(data[3, :], color = "black", label = "data")
ax[3].set_ylabel("Mean Firm Size")
ax[1].legend()
fig

##
function evaluate(params, n)
    return -mean(fetch.([Threads.@spawn logpdf(abm(params), data) for _ in 1:n]))
end

function train_model(initial_params, n_epochs, lr, n_samples)
    opt = Optimisers.Adam(lr)
    state = Optimisers.setup(opt, initial_params)
    
    params = copy(initial_params)
    losses = Float64[]
    best_loss = Inf
    best_params = copy(params)
    
    p = Progress(n_epochs, desc="Training: ", showspeed=true)
    
    for epoch in 1:n_epochs
        loss, grads = DifferentiationInterface.value_and_gradient(
            x -> evaluate(x, n_samples), 
            AutoForwardDiff(chunksize=chunksize), 
            params
        )
        
        state, params = Optimisers.update(state, params, grads)
        
        push!(losses, loss)
        ProgressMeter.next!(p; showvalues = [(:epoch, epoch), (:loss, loss)])
        
        if loss < best_loss
            best_loss = loss
            best_params = copy(params)
        end
    end
    
    return best_params, losses
end

initial_params, _ = Flux.destructure(abm)
n_epochs = 1000
lr = 3e-3
n_samples = 8

best_params, losses = train_model(initial_params, n_epochs, lr, n_samples)

##
fig, ax = plt.subplots()
ax.plot(losses)
fig

##
# predictions
n_samples = 10
fig, ax = plt.subplots(1, 3, figsize = (12, 4))
for i in 1:n_samples
    trained_pred = rand(abm(best_params))
    prior_pred = rand(abm(initial_params))
    true_pred = generate_true_data()
    ax[1].plot(trained_pred[1, :], color = "C0", alpha = 0.5)
    ax[2].plot(trained_pred[2, :], color = "C0", alpha = 0.5)
    ax[3].plot(trained_pred[3, :], color = "C0", alpha = 0.5)
    ax[1].plot(prior_pred[1, :], color = "C3", alpha = 0.5)
    ax[2].plot(prior_pred[2, :], color = "C3", alpha = 0.5)
    ax[3].plot(prior_pred[3, :], color = "C3", alpha = 0.5)
    ax[1].plot(true_pred[1, :], color = "grey", alpha = 0.5)
    ax[2].plot(true_pred[2, :], color = "grey", alpha = 0.5)
    ax[3].plot(true_pred[3, :], color = "grey", alpha = 0.5)
end
ax[1].set_title("Mean Agent Effort")
ax[2].set_title("Mean Firm Output")
ax[3].set_title("Mean Firm Size")
ax[1].plot(data[1, :], color = "black", label = "data")
ax[2].plot(data[2, :], color = "black", label = "data")
ax[3].plot(data[3, :], color = "black", label = "data")
ax[1].plot([], [], color = "C0", alpha = 0.5, label = "trained")
ax[1].plot([], [], color = "C3", alpha = 0.5, label = "prior")
ax[1].legend()
fig

##
# Symbolic Regression
import SymbolicRegression: Options, equation_search

# generate data from the trained neural network and fit with symbolic regression
function generate_data_from_nn(nn, N; firms_output_dist, firms_size_dist, theta_agent_dist, effort_agent_dist)
    ret = Float64[]
    params = []
    for i in 1:N
        firms_output = rand(firms_output_dist)
        firms_size = rand(firms_size_dist)
        theta_agent = rand(theta_agent_dist)
        effort_agent = rand(effort_agent_dist)
        push!(ret, nn([firms_output, firms_size, theta_agent, 1.0, effort_agent])[1])
        push!(params, [firms_output, firms_size, theta_agent, effort_agent])
    end
    return hcat(params...), ret
end

X, y = generate_data_from_nn(nn, 1000; 
        firms_output_dist = Uniform(0.0, 3.0), 
        firms_size_dist = Uniform(0.0, 10.0), 
        theta_agent_dist = Uniform(0.0, 1.0), 
        effort_agent_dist = Uniform(0.0, 1.0))

options = Options(
    binary_operators=[+, *, /, -, ^],
    unary_operators=[],
    populations=20
)
hall_of_fame = equation_search(
    X, y, niterations=40, options=options,
    parallelism=:multithreading
)

##
import SymbolicRegression: calculate_pareto_frontier

dominating = calculate_pareto_frontier(hall_of_fame)
trees = [member.tree for member in dominating]
tree = trees[end]
output = tree(X)

##
using Zygote
function ftest(array, x)
    p = 2 * x
    array[2] = p
    return array
end
Zygote.
