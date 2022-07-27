import numpy as np

import ipywidgets as widgets
import plotly.graph_objects as go

from pynwb.base import DynamicTable

import trimesh

from .base import df_to_hover_text


def make_cylinder_mesh(
    radius, height, sections=32, position=(0, 0, 0), direction=(1, 0, 0), **kwargs
):
    new_normal = direction / np.linalg.norm(direction)
    cosx, cosy = new_normal[:2]
    sinx = np.sqrt(1 - cosx ** 2)
    siny = np.sqrt(1 - cosy ** 2)

    yaw = [[cosx, -sinx, 0, 0], [sinx, cosx, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    pitch = [[cosy, 0, siny, 0], [0, 1, 0, 0], [-siny, 0, cosy, 0], [0, 0, 0, 1]]

    transform = np.dot(yaw, pitch)

    transform[:3, 3] = position

    cylinder = trimesh.primitives.Cylinder(
        radius=radius, height=height, sections=sections, transform=transform
    )

    x, y, z = cylinder.vertices.T
    i, j, k = cylinder.faces.T

    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)


def make_cylinders(
    positions, directions, radius=1, height=1, sections=32, name="cylinders", **kwargs
):

    return [
        make_cylinder_mesh(
            position=position,
            direction=direction,
            radius=radius,
            height=height,
            sections=sections,
            showlegend=not i,
            legendgroup=name,
            name=name,
            **kwargs
        )
        for i, (position, direction) in enumerate(zip(positions, directions))
    ]


class HumanElectrodesPlotlyWidget(widgets.VBox):
    def __init__(self, electrodes: DynamicTable, **kwargs):

        super().__init__()
        self.electrodes = electrodes

        slider_kwargs = dict(
            value=1.0, min=0.0, max=1.0, style={"description_width": "initial"}
        )

        left_opacity_slider = widgets.FloatSlider(
            description="left hemi opacity", **slider_kwargs
        )

        right_opacity_slider = widgets.FloatSlider(
            description="right hemi opacity", **slider_kwargs
        )

        color_by_dropdown = widgets.Dropdown(
            options=list(electrodes.colnames),
            value="group_name",
            description="Color By:",
            disabled=False,
        )

        color_by_dropdown.observe(self.color_electrode_by)
        left_opacity_slider.observe(self.observe_left_opacity)
        right_opacity_slider.observe(self.observe_right_opacity)

        self.fig = go.FigureWidget()
        self.plot_human_brain()
        self.show_electrodes(electrodes, color_by_dropdown.value)
        sliders = widgets.HBox([left_opacity_slider, right_opacity_slider])
        self.children = [self.fig, widgets.VBox([sliders, color_by_dropdown])]

    @staticmethod
    def find_normals(points, k=3):
        normals = []
        for point in points:
            from skspatial.objects import Points, Plane

            distance = np.linalg.norm(points - point, axis=1)
            # closest_inds = np.argpartition(distance, 3)
            # x0, x1, x2 = points[closest_inds[:3]]
            # normal = np.cross((x1 - x0), (x2 - x0))
            closest_inds = np.argpartition(distance, k)
            close_points = points[closest_inds[:k]]
            normal = np.asarray(Plane.best_fit(close_points).normal)
            normals.append(normal)
        return normals

    def show_electrodes(self, electrodes: DynamicTable, color_by):

        positions = np.c_[electrodes.x[:], electrodes.y[:], electrodes.z[:]]

        if (
            isinstance(electrodes[color_by][0], bytes)
            or isinstance(electrodes[color_by][0], str)
            or isinstance(electrodes[color_by][0], np.bool_)
        ):
            ugroups, group_inv = np.unique(electrodes[color_by][:], return_inverse=True)
            colors = group_inv
            show_leg = True
            show_scale = False
        elif isinstance(electrodes[color_by][0], np.ndarray) or isinstance(
            electrodes[color_by][0], np.float
        ):
            colors = np.ravel(electrodes[color_by][:])
            ugroups, group_inv = [0], np.array([0] * len(colors))
            show_leg = False
            show_scale = True

        else:
            print("Not a valid data type")
            return

        c_max = np.max(colors)
        c_min = np.min(colors)

        for i, group in enumerate(ugroups):
            sel_positions = positions[group_inv == i]
            c = colors[group_inv == i]
            x, y, z = sel_positions.T

            if isinstance(group, bytes):
                group = group.decode()

                """
                if 'GRID' in group:
                    normals = self.find_normals(sel_positions, 5)
                    with self.fig.batch_update():
                        [self.fig.add_trace(trace) for trace in make_cylinders(
                            positions=sel_positions,
                            directions=normals,
                            radius=2,
                            height=.5,
                            color=c,
                            name=group
                    )]
                else:


                """
            self.fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    x=x,
                    y=y,
                    z=z,
                    name=str(group),
                    legendgroup=str(group),
                    marker=dict(
                        color=c,
                        cmax=c_max,
                        cmin=c_min,
                        colorscale="Viridis",
                        colorbar=dict(title="Colorbar"),
                        showscale=show_scale,
                    ),
                    text=df_to_hover_text(electrodes.to_dataframe()),
                    hoverinfo="text",
                    showlegend=show_leg,
                )
            ),

        self.fig.update_layout(
            legend=dict(
                x=0,
                y=1,
            ),
        )

    def plot_human_brain(self, left_opacity=1.0, right_opacity=1.0):

        from nilearn import datasets, surface

        mesh = datasets.fetch_surf_fsaverage("fsaverage5")

        def create_mesh(name, **kwargs):
            vertices, triangles = surface.load_surf_mesh(mesh[name])
            x, y, z = vertices.T
            i, j, k = triangles.T

            return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)

        kwargs = dict(
            color="lightgray",
            lighting=dict(specular=1, ambient=0.9, roughness=0.9, diffuse=0.9),
            hoverinfo="skip",
        )

        self.fig.add_trace(create_mesh("pial_left", opacity=left_opacity, **kwargs))
        self.fig.add_trace(create_mesh("pial_right", opacity=right_opacity, **kwargs))

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            ),
            height=500,
            margin=dict(t=20, b=0),
        )

    def observe_left_opacity(self, change):
        if "new" in change and isinstance(change["new"], float):
            self.fig.data[0].opacity = change["new"]

    def observe_right_opacity(self, change):
        if "new" in change and isinstance(change["new"], float):
            self.fig.data[1].opacity = change["new"]

    def color_electrode_by(self, change):
        if "new" in change and isinstance(change["new"], str):
            with self.fig.batch_update():
                self.fig.data = None
                self.plot_human_brain()
                self.show_electrodes(self.electrodes, change["new"])
