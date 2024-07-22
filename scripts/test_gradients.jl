using BlackBIRDS
using DiffSIR
using Functors
using FiniteDifferences
using DifferentiationInterface
using StatsBase

using BlackBIRDS.SIRJune
using BlackBIRDS.RandomWalk
using BlackBIRDS.BrockHommes

using PyPlot

##
function make_value_plot(values_ad, values_fd; figsize)
    n_plots = size(values_ad, 1)
    fig, ax = plt.subplots(1, n_plots, figsize = figsize)
    if n_plots == 1
        ax = [ax]
    end
    for i in 1:n_plots
        ax[i].plot(values_ad[i, :], label = "AD", linewidth=2, alpha=0.5)
        ax[i].plot(values_fd[i, :], label = "FD", linewidth=2, alpha=0.5)
        ax[i].legend()
    end
    fig
end
function make_jacobian_plot(jacobians_ad, jacobian_fd, n_model_dim, n_timesteps; figsize)
    j_ad = cat(jacobians_ad..., dims = ndims(jacobians_ad[1]) + 1)
    #j_ad = reduce(cat, jacobians_ad, dims = ndims(jacobians_ad[1]) + 1)
    j_ad_mean = sum(jacobians_ad) / length(jacobians_ad)
    n_plots = size(j_ad_mean, 2)
    fig, ax = plt.subplots(n_model_dim, n_plots, figsize = figsize)
    if n_plots == 1
        ax = [ax]
    end
    if n_model_dim == 1
        ax = [ax]
    end
    for j in 1:n_plots
        j_ad_j = reshape(j_ad[:, j, :], n_model_dim, n_timesteps, :)
        j_fd_j = reshape(jacobian_fd[:, j], n_model_dim, n_timesteps, :)
        for i in 1:n_model_dim
            ax[i, j].boxplot(j_ad_j[i, :, :]', showmeans=true)
            #ax[i, j].plot(j_ad_mean[:,j], label = "AD")
            ax[i, j].plot(j_fd_j[i, :], label = "FD")
            ax[i, j].legend()
        end
    end
    fig
end
## RandomWalk

n_timesteps = 100
x = [0.3]

model = RandomWalkModel(n_timesteps, x, MSELoss(1.0));

fdm = central_fdm(50, 1, max_range=0.2)
n_samples_ad = 100
n_samples_fd = 10000

## StochasticAD
ad = AutoForwardDiff()
v_ad, j_ad, v_fd, j_fd = test_model_gradients(
    model, ad, x, n_samples_ad = n_samples_ad,
    n_samples_fd = n_samples_fd, fdm = fdm);

##

v_ad_mean = reshape(sum(v_ad) / n_samples_ad, 1, :)
v_fd_mean = reshape(v_fd, 1, :)
make_value_plot(v_ad_mean, v_fd_mean, figsize = (15, 5))

fig = make_jacobian_plot(j_ad, j_fd, figsize = (10, 10))
fig



## BrockHommes
n_timesteps = 10
time_horizon = n_timesteps
g2, g3, b2, b3 = 0.9, 0.9, 0.2, -0.2
x = [g2, g3, b2, b3]

model = BrockHommesModel(n_timesteps, x, MSELoss(1.0), time_horizon);
fdm = central_fdm(50, 1, max_range=0.2)
n_samples_ad = 10000
n_samples_fd = 10000

## 
ad = AutoForwardDiff()
v_ad, j_ad, v_fd, j_fd = test_model_gradients(
    model, ad, x, n_samples_ad = n_samples_ad,
    n_samples_fd = n_samples_fd, fdm = fdm);


## 
v_ad_mean = reshape(sum(v_ad) / n_samples_ad, 1, :)
v_fd_mean = reshape(v_fd, 1, :)
make_value_plot(v_ad_mean, v_fd_mean, figsize = (5, 5))
##

fig = make_jacobian_plot(j_ad, j_fd, figsize = (25, 7))
fig

## SIRJune
n_agents = 250
venues = ["household"] #, "company", "school", "leisure"]
fraction_population_per_venue = [1.0] #, 0.4, 0.4, 1.0]
number_per_venue = [1] #Int.(floor.([1/5, 1/10, 1/10, 1/2] .* n_agents))
graph = generate_random_world_graph(
    n_agents, venues, fraction_population_per_venue, number_per_venue)
initial_infected = 0.1
gamma = 0.05
betas = [0.5] #, 0.2, 0.2, 0.1]
n_timesteps = 30
loss = KDELoss(20, BlackBIRDS.MMDKernel())
discrete_sampler = DiffSIR.SAD()
model = SIRJuneModel(
    graph, initial_infected, betas, gamma, n_timesteps, discrete_sampler, loss);
@functor SIRJuneModel (venue_betas, gamma)
fdm = central_fdm(50, 1, max_range=0.05)
n_samples_ad = 10
n_samples_fd = 1000

##
fig, ax = plt.subplots()
d = rand(model)
ax.plot(d[1, :], label = "Infected")
ax.plot(d[2, :], label = "Recovered")
fig
##

#x = [0.1, 0.5, 0.1]
x = [0.5, 0.05]

## 
ad = AutoStochasticAD(10)
v_ad, j_ad, v_fd, j_fd = test_model_gradients(
    model, ad, x, n_samples_ad = n_samples_ad,
    n_samples_fd = n_samples_fd, fdm = fdm);

## 
v_ad_mean = reshape(sum(v_ad) / n_samples_ad, 2, :)
v_fd_mean = reshape(v_fd, 2, :)
make_value_plot(v_ad_mean, v_fd_mean, figsize = (10, 5))
##

fig = make_jacobian_plot(j_ad, j_fd, 2, n_timesteps, figsize = (25, 7))
fig
