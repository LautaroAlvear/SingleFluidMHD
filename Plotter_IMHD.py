import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as manimation

taskname = 'p'

with h5py.File("snapshots/snapshots_s1.h5", mode='r') as file:
        # Load datasets
        task = file['tasks'][taskname]
        t = task.dims[0]['sim_time']
        phi = task.dims[1][0]
        theta = task.dims[2][0]
        r = task.dims[3][0]


        Theta, R = np.meshgrid(theta, r)
        

        step = 40

        #FFMpegWriter = manimation.writers['ffmpeg']
        #metadata = dict(title='Movie Test', artist='Matplotlib',
        #            comment='a red circle following a blue sine wave')
        #writer = FFMpegWriter(fps=10, metadata=metadata)

        for i in range(0,len(task),step):
            print(f"it {i} de {len(task)}")
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            # Graficar
            matrix = task[i][0][:][:]
            pcm = ax.pcolormesh(Theta,R,np.transpose(matrix), shading="nearest", cmap='inferno')

            # Opcional: agregar barra de color
            fig.colorbar(pcm, ax=ax, orientation='vertical', label=taskname)

            plt.savefig(f"frames/frame_{i}.jpg")
            plt.clf()
            # Mostrar
            #ax.clear()