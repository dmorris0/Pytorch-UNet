import shutil, os
from run_params import get_run_dirs

if __name__=="__main__":

    output_dir = 'out_eggs'

    dir_plot = os.path.join(os.path.dirname(__file__), output_dir, 'Plots')
    print(dir_plot)

    for run in range(34):
        dir_run,_ = get_run_dirs(output_dir, run)
        plot_name = os.path.join(dir_run,f'scores_{run:03d}.png')
        if os.path.exists(plot_name):
            # print(plot_name)
            shutil.copy2(plot_name, dir_plot)

    print('Done')
    