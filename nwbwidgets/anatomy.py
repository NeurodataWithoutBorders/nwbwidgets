import matplotlib
import numpy as np


COLORS = [x['color'] for x in list(matplotlib.rcParams["axes.prop_cycle"])]


def show_brainrender(electrodes_table=None, r=40, **kwargs):
    from vtkplotter import embedWindow, show
    embedWindow('itkwidgets')
    from brainrender.scene import Scene
    from vedo import Spheres

    scene = Scene(verbose=False)

    xs, ys, zs, group_names = [np.array(electrodes_table[col][:])
                               for col in ('x', 'y', 'z', 'group_name')]
    all_pos = np.c_[xs, ys, zs]
    ugroups = np.unique(group_names)

    for group, c in zip(ugroups, COLORS):
        group_pos = all_pos[group_names == group, :]
        spheres = Spheres(group_pos, r=r, c=c)
        spheres.name = group
        scene.actors.append(spheres)

    scene.render()

    widget = show(scene.actors)
    widget.background = (1, 1, 1)

    return widget