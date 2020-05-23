import torch
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import os

import neuro_shooting.figure_settings as figure_settings
import neuro_shooting.figure_utils as figure_utils

# Visualization
# TODO: revamp

def plot_particles(q1,p_q1,ax):

    quiver_scale = 1.0 # to scale the magnitude of the quiver vectors for visualization

    q1 = q1[:,0,:]
    p_q1 = p_q1[:,0,:]

    # let's first just plot the positions
    ax.plot(q1[:,0], q1[:,1],'k+',markersize=12)
    ax.quiver(q1[:,0], q1[:,1], p_q1[:,0], p_q1[:,1], color='r') #, scale=quiver_scale)

    ax.set_title('q1 and p_q1; q1 in [{:.1f},{:.1f}],\np_q1 in [{:.1f},{:.1f}]'.format(
        q1.min(), q1.max(),
        p_q1.min(), p_q1.max(),
    ))

def plot_higher_order_state(q2,p_q2,ax):
    quiver_scale = 1.0 # to scale the magnitude of the quiver vectors for visualization

    sz = q2.shape
    if sz[1]!=1:
        raise ValueError('Expected size 1 for dimensions 1.')

    nr_of_particles = sz[0]

    # we do PCA if there are more than two components
    q2 = q2[:,0,:]
    p_q2 = p_q2[:,0,:]

    if (sz[2]>2) and (nr_of_particles>1):

        # project it to two dimensions
        pca = decomposition.PCA(n_components=2)
        pca_q2 = pca.fit_transform(q2)
        #pca_p_q2 = pca.transform(p_q2)
        # want to be able to see what the projection is afer the addition
        pca_p_q2 = pca.transform(q2+p_q2)-pca_q2

        overall_explained_variance_in_perc = 100*pca.explained_variance_ratio_.sum()

        ax.plot(pca_q2[:, 0], pca_q2[:, 1], 'k+', markersize=12)
        ax.quiver(pca_q2[:, 0], pca_q2[:, 1], pca_p_q2[:, 0], pca_p_q2[:, 1], color='r') #, scale=quiver_scale)

        ax.set_title('q2 and p_q2; q2 in [{:.1f},{:.1f}],\np_q2 in [{:.1f},{:.1f}] \nPCA explained_var={:.1f}%'.format(pca_q2.min(),pca_q2.max(),
                                                                                                                     pca_p_q2.min(),pca_p_q2.max(),
                                                                                                                     overall_explained_variance_in_perc))

    else:
        # we can just plot it directly
        # let's first just plot the positions
        ax.plot(q2[:,0], q2[:,1],'k+',markersize=12)
        ax.quiver(q2[:,0], q2[:,1], p_q2[:,0], p_q2[:,1], color='r') # scale=quiver_scale)

        ax.set_title('q2 and p_q2; q2 in [{:.1f},{:.1f}],\np_q2 in [{:.1f},{:.1f}]'.format(
            q2.min(), q2.max(),
            p_q2.min(), p_q2.max()))


def plot_trajectories(val_y, val_pred_y, val_sim_time=None, train_y=None, train_pred_y=None, train_t=None, itr=None, itr_name='iter', ax=None, losses_to_print=None, nr_of_pars=None, print=False):

    if val_y is not None:
        for n in range(val_y.shape[1]):
            ax.plot(val_y[:, n, 0, 0], val_y[:, n, 0, 1], 'g-')
            ax.plot(val_pred_y[:, n, 0, 0], val_pred_y[:, n, 0, 1], 'b--+')

    if train_y is not None:
        for n in range(train_y.shape[1]):
            ax.plot(train_y[:, n, 0, 0], train_y[:, n, 0, 1], 'k-')
            ax.plot(train_pred_y[:, n, 0, 0], train_pred_y[:, n, 0, 1], 'r--')

    if losses_to_print is not None:
        # losses_to_print = {'model_name': args.shooting_model, 'loss': loss.item(), 'sim_loss': sim_loss.item(),
        #                    'norm_loss': norm_loss.item(), 'par_norm': norm_penalty.item()}
        current_title = 'Trajectories of model {}:\n{}={}; loss={:.4f}; sim_loss={:.4f};\nnorm_loss={:.4f}; par_norm={:.4f}'.format(losses_to_print['model_name'],
                                                                                                                itr_name,
                                                                                                                itr,
                                                                                                                losses_to_print['loss'],
                                                                                                                losses_to_print['sim_loss'],
                                                                                                                losses_to_print['norm_loss'],
                                                                                                                losses_to_print['par_norm'])
    else:
        current_title = 'trajectories: {} = {}'.format(itr_name, itr)

    if nr_of_pars is not None:
        current_title += '\n#pars={} = {}(fixed) + {}(optimized)'.format(nr_of_pars['overall'],nr_of_pars['fixed'],nr_of_pars['optimized'])

    if print:
        current_title = figure_utils.escape_latex_special_characters(current_title)

    ax.set_title(current_title)

def convert_to_numpy_from_torch_if_needed(v):
    if v is None:
        return v
    try:
        v_c = v.detach().cpu().numpy()
    except:
        v_c = v
    return v_c

def _basic_visualize(shooting_block, val_y, val_pred_y, val_sim_time, train_y=None, train_pred_y=None, train_t=None, itr=None,
                    uses_particles=True, losses_to_print=None, nr_of_pars=None,
                    args=None, visualize=True, print=False):

    if visualize and print:
        raise ValueError('Only visualize or print can be set to True at the same time.')

    if not visualize and not print:
        return

    if print:
        previous_backend, rcsettings = figure_settings.setup_pgf_plotting()

    # do conversions to numpy if necessary. This is for convencience. Allows calling with or without torch tensor.
    val_y = convert_to_numpy_from_torch_if_needed(val_y)
    val_pred_y = convert_to_numpy_from_torch_if_needed(val_pred_y)
    val_sim_time = convert_to_numpy_from_torch_if_needed(val_sim_time)
    train_y = convert_to_numpy_from_torch_if_needed(train_y)
    train_pred_y = convert_to_numpy_from_torch_if_needed(train_pred_y)
    train_t = convert_to_numpy_from_torch_if_needed(train_t)

    if uses_particles:

        fig = plt.figure(figsize=(12, 4), facecolor='white')

        ax = fig.add_subplot(131, frameon=False)
        ax_lo = fig.add_subplot(132, frameon=False)
        ax_ho = fig.add_subplot(133, frameon=False)

        # plot it without any additional information
        plot_trajectories(val_y, val_pred_y, val_sim_time, train_y, train_pred_y, train_t, itr, ax=ax, losses_to_print=losses_to_print, nr_of_pars=nr_of_pars, print=print)

        # get all the parameters that we are optimizing over
        pars = shooting_block.state_dict()
        q1 = pars['q1'].detach().cpu().numpy()
        p_q1 = pars['p_q1'].detach().cpu().numpy()
        q2 = pars['q2'].detach().cpu().numpy()
        p_q2 = pars['p_q2'].detach().cpu().numpy()

        # now plot the information from the state variables
        plot_particles(q1=q1,p_q1=p_q1,ax=ax_lo)
        plot_higher_order_state(q2=q2,p_q2=p_q2,ax=ax_ho)

    else:

        fig = plt.figure(figsize=(4,4), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plot_trajectories(val_y, val_pred_y, val_sim_time, train_y, train_pred_y, train_t, itr, ax=ax, losses_to_print=losses_to_print, nr_of_pars=nr_of_pars, print=print)

    if print:
        figure_utils.save_all_formats(output_directory=args.output_directory,
                                      filename='basic-viz-{}-iter-{:04d}'.format(args.output_basename,itr))
        plt.close()

    if visualize:
        plt.show()

    if print:
        figure_settings.reset_pgf_plotting(backend=previous_backend, rcsettings=rcsettings)


def basic_visualize(shooting_block, val_y, val_pred_y, val_sim_time, train_y=None, train_pred_y=None, train_t=None, itr=None,
                    uses_particles=True, losses_to_print=None, nr_of_pars=None, args=None):

    if args is None:
        _basic_visualize(shooting_block=shooting_block,
                         val_y=val_y, val_pred_y=val_pred_y, val_sim_time=val_sim_time,
                         train_y=train_y, train_pred_y=train_pred_y, train_t=train_t, itr=itr,
                         uses_particles=uses_particles, losses_to_print=losses_to_print, nr_of_pars=nr_of_pars, args=args)
    else:
        if args.viz:
            _basic_visualize(shooting_block=shooting_block,
                             val_y=val_y, val_pred_y=val_pred_y, val_sim_time=val_sim_time,
                             train_y=train_y, train_pred_y=train_pred_y, train_t=train_t, itr=itr,
                             uses_particles=uses_particles, losses_to_print=losses_to_print, nr_of_pars=nr_of_pars, args=args,
                             visualize=True,print=False)
        if args.save_figures:
            _basic_visualize(shooting_block=shooting_block,
                             val_y=val_y, val_pred_y=val_pred_y, val_sim_time=val_sim_time,
                             train_y=train_y, train_pred_y=train_pred_y, train_t=train_t, itr=itr,
                             uses_particles=uses_particles, losses_to_print=losses_to_print, nr_of_pars=nr_of_pars, args=args,
                             visualize=False,print=True)

def convert_list_of_np_arrays_into_np_array(lofnp):
    sz = lofnp[0].shape
    nr = len(lofnp)

    new_sz = [nr] + list(sz)
    new_array = np.zeros(shape=new_sz,dtype=lofnp[0].dtype)

    for i,v in enumerate(lofnp):
        new_array[i,...] = v

    return new_array

def create_animated_gif(filter,out_filename):

    if os.system('which convert')==0:

        cmd = 'convert -delay 10 {} {}'.format(filter,out_filename)

        if os.path.isfile(out_filename):
            os.remove(out_filename)

        print('Creating animated gif: {}'.format(out_filename))
        os.system(cmd)
    else:
        print('WARNING: Could not create animated gif, because convert command was not found. Ignoring.')

def visualize_time_evolution(val_y, data,block_name, max_display=10, save_to_directory=None, file_prefix='evolution'):

    # time
    t = np.asarray(data['t'])

    val_y = convert_to_numpy_from_torch_if_needed(val_y)

    q1 = convert_list_of_np_arrays_into_np_array(data['{}.state.q1'.format(block_name)])
    p_q1 = convert_list_of_np_arrays_into_np_array(data['{}.costate.p_q1'.format(block_name)])
    q2 = convert_list_of_np_arrays_into_np_array(data['{}.state.q2'.format(block_name)])
    p_q2 = convert_list_of_np_arrays_into_np_array(data['{}.costate.p_q2'.format(block_name)])

    data_q1 = convert_list_of_np_arrays_into_np_array(data['{}.data.q1'.format(block_name)])
    data_q2 = convert_list_of_np_arrays_into_np_array(data['{}.data.q2'.format(block_name)])

    desired_to_time_pts = list(range(len(t)))
    step_size = 1
    desired_to_time_pts = desired_to_time_pts[0::step_size]

    if save_to_directory is not None:
        if not os.path.exists(save_to_directory):
            print('Creating output directory: {:s}'.format(save_to_directory))
            os.mkdir(save_to_directory)
        else:
            print('WARNING: Output directory {:s} already exists. Files might get overwritten.'.format(save_to_directory))

    for itr, to_time_pt in enumerate(desired_to_time_pts):

        fig = plt.figure(figsize=(12, 4), facecolor='white')

        ax = fig.add_subplot(131, frameon=False)
        ax_lo = fig.add_subplot(132, frameon=False)
        ax_ho = fig.add_subplot(133, frameon=False)

        current_data_q1 = data_q1[0:to_time_pt+1,...]
        current_q1 = q1[to_time_pt,...]
        current_p_q1 = p_q1[to_time_pt,...]
        current_q2 = q2[to_time_pt,...]
        current_p_q2 = p_q2[to_time_pt,...]

        # plot it without any additional information
        plot_trajectories(val_y, current_data_q1, itr='{:.2f}'.format(t[to_time_pt]), itr_name = 'time', ax=ax)

        # now plot the information from the state variables
        plot_particles(q1=current_q1, p_q1=current_p_q1, ax=ax_lo)
        plot_higher_order_state(q2=current_q2, p_q2=current_p_q2, ax=ax_ho)

        if save_to_directory is not None:
            current_filename = '{}-{:05d}.png'.format(file_prefix,itr)
            print('Saving: {}'.format(current_filename))
            plt.savefig(os.path.join(save_to_directory,current_filename))
            plt.close()
        else:
            plt.show()
            if itr>=max_display:
                print('Reached maximum number of figure to display on screen: {}'.format(max_display))
                break

    if save_to_directory is not None:
        animated_gif_filename = 'animated-{}.gif'.format(file_prefix)
        create_animated_gif(filter='{}/{}-*.png'.format(save_to_directory,file_prefix), out_filename=animated_gif_filename)

def visualize(true_y, pred_y, sim_time, odefunc, itr, is_higher_order_model=True):

    quiver_scale = 2.5 # to scale the magnitude of the quiver vectors for visualization

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')

    for n in range(true_y.size()[1]):
        ax_traj.plot(sim_time.numpy(), true_y.detach().cpu().numpy()[:, n, 0, 0], sim_time.numpy(), true_y.numpy()[:, n, 0, 1],
                 'g-')
        ax_traj.plot(sim_time.numpy(), pred_y.detach().cpu().numpy()[:, n, 0, 0], '--', sim_time.numpy(),
                 pred_y.detach().cpu().numpy()[:, n, 0, 1],
                 'b--')

    ax_traj.set_xlim(sim_time.min(), sim_time.max())
    ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    ax_phase.cla()
    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')

    for n in range(true_y.size()[1]):
        ax_phase.plot(true_y.detach().cpu().numpy()[:, n, 0, 0], true_y.detach().cpu().numpy()[:, n, 0, 1], 'g-')
        ax_phase.plot(pred_y.detach().cpu().numpy()[:, n, 0, 0], pred_y.detach().cpu().numpy()[:, n, 0, 1], 'b--')

    try:
        q = (odefunc.q_params)
        p = (odefunc.p_params)

        q_np = q.cpu().detach().squeeze(dim=1).numpy()
        p_np = p.cpu().detach().squeeze(dim=1).numpy()

        ax_phase.scatter(q_np[:,0],q_np[:,1],marker='+')
        ax_phase.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)
    except:
        pass

    ax_phase.set_xlim(-2, 2)
    ax_phase.set_ylim(-2, 2)


    ax_vecfield.cla()
    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]

    current_y = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))

    # print("q_params",q_params.size())

    x_0 = current_y.unsqueeze(dim=1)

    #viz_time = t[:5] # just 5 timesteps ahead
    viz_time = sim_time[:5] # just 5 timesteps ahead

    odefunc.set_integration_time_vector(integration_time_vector=viz_time,suppress_warning=True)
    dydt_pred_y,_,_,_ = odefunc(x=x_0)

    if is_higher_order_model:
        dydt = (dydt_pred_y[-1,...]-dydt_pred_y[0,...]).detach().cpu().numpy()
        dydt = dydt[:,0,...]
    else:
        dydt = dydt_pred_y[-1,0,...]

    mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")

    try:
        ax_vecfield.scatter(q_np[:, 0], q_np[:, 1], marker='+')
        ax_vecfield.quiver(q_np[:,0],q_np[:,1], p_np[:,0],p_np[:,1],color='r', scale=quiver_scale)
    except:
        pass

    ax_vecfield.set_xlim(-2, 2)
    ax_vecfield.set_ylim(-2, 2)

    fig.tight_layout()

    plt.show()

def _plot_temporal_data(data, block_name, args=None, visualize=True, print=False):

    if visualize and print:
        raise ValueError('Only visualize or print can be set to True at the same time.')

    if not visualize and not print:
        return

    if print:
        previous_backend, rcsettings = figure_settings.setup_pgf_plotting()

    # time
    t = np.asarray(data['t'])
    # energy
    energy = np.asarray(data['energy'])

    # first plot the energy over time
    plt.figure()
    plt.plot(t,energy)
    plt.xlabel('time')
    plt.ylabel('energy')

    if print:
        figure_utils.save_all_formats(output_directory=args.output_directory,
                                      filename='{}-energy'.format(args.output_basename))
        plt.close()

    if visualize:
        plt.show()

    # exclude list (what not to plot, partial initial match is fine)
    do_not_plot = ['t', 'energy', 'dot_state','dot_costate','dot_data']

    for k in data:

        # first check if we should plot this
        do_plotting = True
        for dnp in do_not_plot:
            if k.startswith(dnp) or k.startswith('{}.{}'.format(block_name,dnp)):
                do_plotting = False

        if do_plotting:
            plt.figure()

            cur_vals = np.asarray(data[k]).squeeze()
            cur_shape = cur_vals.shape
            if len(cur_shape)==3: # multi-dimensional state
                for cur_dim in range(cur_shape[2]):
                    plt.plot(t,cur_vals[:,:,cur_dim])
            else:
                plt.plot(t,cur_vals)

            plt.xlabel('time')
            if print:
                plt.ylabel(figure_utils.escape_latex_special_characters(k))
            else:
                plt.ylabel(k)

            if print:
                figure_utils.save_all_formats(output_directory=args.output_directory,
                                              filename='{}-{}'.format(args.output_basename, k))
                plt.close()

            if visualize:
                plt.show()

    if print:
        figure_settings.reset_pgf_plotting(backend=previous_backend, rcsettings=rcsettings)


def plot_temporal_data(data, block_name, args=None):
    if args is None:
        _plot_temporal_data(data=data, block_name=block_name, args=args)
    else:
        if args.viz:
            _plot_temporal_data(data=data, block_name=block_name, args=args, visualize=True, print=False)
        if args.save_figures:
            _plot_temporal_data(data=data, block_name=block_name, args=args, visualize=False, print=True)
