
def video_overlay(pa,ned,pos_des,output_filename):
    # Generate video of the performance during loiter, this
    # includes the loiter position, current estimate of
    # position and raw gps position

    #output_filename = 'out.mp4'

    import matplotlib.animation as animation
    import scipy.signal as signal
    import matplotlib.pyplot as plt
    from numpy import nan, arange
    from numpy.lib.function_base import diff
    from matplotlib.pyplot import plot, xlabel, ylabel, xlim, ylim, draw
    from matplotlib.mlab import find

    ####### plot section

    fig, ax = plt.subplots(1,sharex=True)
    fig.set_size_inches([8,4])
    fig.patch.set_facecolor('green')

    t0_s = pa['time'][0]
    t1_s = pa['time'][-1]

    # nan out gaps between flights
    idx = find(diff(pos_des['time']) > 5000)
    pos_des['End'][idx,2] = nan

    # Plot position
    plot(pa['East'][:,0], pa['North'][:,0], linewidth=3, color=(0.2,0.2,0.2), label="Position")
    plot(pa['East'][:,0], pa['North'][:,0], linewidth=2, color=(0.8,0.8,0.8), label="Position")
    desired_marker = plot(pos_des['End'][0,1], pos_des['End'][0,0], '*k', markersize=15, label="Desired")
    ned_marker = plot(pa['East'][0,0], pa['North'][0,0], '.b', markersize=8)
    current_marker = plot(pa['East'][0,0], pa['North'][0,0], '.r', markersize=13)

    xlabel('East (m)')
    ylabel('North (m)')
    xlim(min(pa['East'][:,0]-20), max(pa['East'][:,0])+20)
    ylim(min(pa['North'][:,0]-10), max(pa['North'][:,0])+10)

    ax.set_axis_bgcolor('green')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('green')
    ax.spines['right'].set_color('green')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    fig.set_facecolor('green')
    fig.subplots_adjust(left=0.15)
    fig.subplots_adjust(bottom=0.15)

    draw()

    def init(fig=fig):
        fig.set_facecolor('green')

    # Plot a segment of the path
    def update_img(t, pa=pa, pos_des=pos_des, ned=ned, desired_marker=desired_marker, current_marker=current_marker, ned_marker=ned_marker, t0_s=t0_s):

        import numpy as np
        import matplotlib.pyplot

        # convert to minutes
        t = t / 60

        idx = np.argmin(abs(np.double(pa['time']) / 60 - t))
        x = pa['East'][idx,0]
        y = pa['North'][idx,0]
        current_marker[0].set_xdata(x)
        current_marker[0].set_ydata(y)

        idx = np.argmin(abs(np.double(ned['time']) / 60 - t))
        ned_marker[0].set_xdata(ned['East'][idx])
        ned_marker[0].set_ydata(ned['North'][idx])

        idx = np.argmin(abs(np.double(pos_des['time']) / 60 - t))
        delta = abs(np.double(pos_des['time'][idx]) / 60 - t)
        if delta < (1/60.0):
            desired_marker[0].set_xdata(pos_des['End'][idx,1])
            desired_marker[0].set_ydata(pos_des['End'][idx,0])


    fig.patch.set_facecolor('green')

    fps = 30.0
    dpi = 150
    t = arange(t0_s, t1_s, 1/fps)

    ani = animation.FuncAnimation(fig,update_img,t,init_func=init,interval=0,blit=False)
    writer = animation.writers['ffmpeg'](fps=30)
    ani.save(output_filename,dpi=dpi,fps=30,writer=writer,savefig_kwargs={'facecolor':'green'})


def main():
        import sys, os
        sys.path.insert(1, os.path.dirname(sys.path[0]))
        from taulabs import telemetry
        uavo_list = telemetry.GetUavoBasedOnArgs()
        from taulabs.uavo import UAVO_PositionActual, UAVO_NEDPosition, UAVO_PathDesired

        pa = uavo_list.as_numpy_array(UAVO_PositionActual)
        ned = uavo_list.as_numpy_array(UAVO_NEDPosition)
        pos_des = uavo_list.as_numpy_array(UAVO_PathDesired)

        out = "position_overlay.mp4"
        video_overlay(pa,ned,pos_des,out)


if __name__ == "__main__":
        main()
