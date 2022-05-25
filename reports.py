import wandb
from image import ImageDataHistogram

class HistogramChart:
    @staticmethod
    def generate_chart():
        working_folders = ['dataset/covid', 'dataset/normal', 'cov_processed', 'non_cov_processed']
        mean_data = [ImageDataHistogram.hist_mean(i) for i in working_folders]
        median_data = [ImageDataHistogram.hist_median(i) for i in working_folders]

        wandb.log({
            'cov_hist_mean': mean_data[0],
            'non_cov_hist_mean': mean_data[1],
            'cov_proc_hist_mean': mean_data[2],
            'non_cov_proc_hist_mean': mean_data[3],
            'cov_hist_median': median_data[0],
            'non_cov_hist_median': median_data[1],
            'cov_proc_hist_median': median_data[2],
            'non_cov_proc_hist_median': median_data[3]
        })

        plots = []

        for data in [[mean_data, "Mean Comparison"], [median_data, "Median Comparison"]]:
            xs = [i for i in range(1, 256)]
            x_axis_label = "Intensity"

            plots.append(wandb.plot.line_series(xs=xs,
                                                ys=[data[0][0], data[0][1]],
                                                keys=["cov", "non_cov"], title=data[1], xname=x_axis_label))
            plots.append(wandb.plot.line_series(xs=xs,
                                                ys=[data[0][2], data[0][3]],
                                                keys=["cov_proc", "non_cov_proc"], title=data[1], xname=x_axis_label))

        # Log custom tables, which will show up in customizable charts in the UI
        wandb.log({'line_' + str(i+1): plots[i] for i in range(0, len(plots))})