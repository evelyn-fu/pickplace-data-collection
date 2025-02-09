import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

class FramePlacerApp:
    def __init__(self, pcd):
        self.pcd = pcd
        self.frames = []         # List to store finalized 4x4 transformation matrices.
        self.temp_origin = None  # The origin (3D point) chosen for the temporary frame.
        self.temp_angles = [0.0, 0.0, 0.0]  # [roll, pitch, yaw] in degrees.
        self.is_placing = False  # Whether we are waiting for a click to place a frame.
        self.temp_frame_name = "TempFrame"

        # Create the main window.
        self.window = gui.Application.instance.create_window("Coordinate Frame Placer", 1024, 768)
        self.window.set_on_layout(self.on_layout)

        # Create the SceneWidget.
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        # Create a control panel (a vertical container) for your buttons and sliders.
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em))
        # Add "Place Frame" button.
        self.place_button = gui.Button("Place Frame")
        self.place_button.horizontal_padding_em = 0.5
        self.place_button.vertical_padding_em = 0.5
        self.place_button.set_on_clicked(self.on_place_frame)
        self.panel.add_child(self.place_button)
        # Add Roll slider.
        self.panel.add_child(gui.Label("Roll (deg)"))
        self.roll_slider = gui.Slider(gui.Slider.DOUBLE)
        self.roll_slider.set_limits(-180.0, 180.0)
        self.roll_slider.double_value = 0.0
        self.roll_slider.set_on_value_changed(self.on_angle_changed)
        self.panel.add_child(self.roll_slider)
        # Add Pitch slider.
        self.panel.add_child(gui.Label("Pitch (deg)"))
        self.pitch_slider = gui.Slider(gui.Slider.DOUBLE)
        self.pitch_slider.set_limits(-180.0, 180.0)
        self.pitch_slider.double_value = 0.0
        self.pitch_slider.set_on_value_changed(self.on_angle_changed)
        self.panel.add_child(self.pitch_slider)
        # Add Yaw slider.
        self.panel.add_child(gui.Label("Yaw (deg)"))
        self.yaw_slider = gui.Slider(gui.Slider.DOUBLE)
        self.yaw_slider.set_limits(-180.0, 180.0)
        self.yaw_slider.double_value = 0.0
        self.yaw_slider.set_on_value_changed(self.on_angle_changed)
        self.panel.add_child(self.yaw_slider)
        # Add Confirm Frame button.
        self.confirm_button = gui.Button("Confirm Frame")
        self.confirm_button.horizontal_padding_em = 0.5
        self.confirm_button.vertical_padding_em = 0.5
        self.confirm_button.set_on_clicked(self.on_confirm_frame)
        self.panel.add_child(self.confirm_button)
        # Add an Exit button.
        self.exit_button = gui.Button("Exit")
        self.exit_button.horizontal_padding_em = 0.5
        self.exit_button.vertical_padding_em = 0.5
        self.exit_button.set_on_clicked(lambda: gui.Application.instance.quit())
        self.panel.add_child(self.exit_button)

        # Add the control panel as a separate child of the window.
        self.window.add_child(self.panel)

        # Create and configure the material for the point cloud.
        pcd_material = rendering.MaterialRecord()
        pcd_material.shader = "defaultUnlit"
        pcd_material.point_size = 3.0  # Set a point size.
        self.pcd_material = pcd_material

        # Add the point cloud geometry.
        self.scene.scene.add_geometry("PointCloud", self.pcd, pcd_material)
        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())
        self.scene.scene.set_background([0.2, 0.2, 0.2, 1.0])

        # Set a mouse callback on the scene.
        self.scene.set_on_mouse(self.on_mouse)

    def on_layout(self, layout_context):
        # Get the available content rectangle.
        r = self.window.content_rect
        # Define a fixed width for the control panel (e.g., 200 pixels).
        panel_width = 200
        # Position the control panel on the right.
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        # Position the scene to fill the rest of the window.
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
        # WORKAROUND: Re-add the point cloud geometry on every layout update.
        try:
            self.scene.scene.remove_geometry("PointCloud")
        except Exception:
            pass
        self.scene.scene.add_geometry("PointCloud", self.pcd, self.pcd_material)

    def on_place_frame(self):
        print("Entering placement mode: click on the point cloud to set frame origin.")
        self.is_placing = True
        self.temp_origin = None
        self.temp_angles = [0.0, 0.0, 0.0]
        self.roll_slider.double_value = 0.0
        self.pitch_slider.double_value = 0.0
        self.yaw_slider.double_value = 0.0
        try:
            self.scene.scene.remove_geometry(self.temp_frame_name)
        except Exception:
            pass
        self.scene.force_redraw()

    def on_angle_changed(self, value):
        if self.temp_origin is None:
            return
        self.temp_angles = [
            self.roll_slider.double_value,
            self.pitch_slider.double_value,
            self.yaw_slider.double_value,
        ]
        T = self.compute_transformation(self.temp_origin, self.temp_angles)
        try:
            self.scene.scene.remove_geometry(self.temp_frame_name)
        except Exception:
            pass
        temp_frame_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        temp_frame_geom.transform(T)
        self.scene.scene.add_geometry(self.temp_frame_name, temp_frame_geom, rendering.MaterialRecord())
        self.scene.force_redraw()

    def on_confirm_frame(self):
        if self.temp_origin is None:
            print("No frame origin selected; cannot confirm frame.")
            return
        T = self.compute_transformation(self.temp_origin, self.temp_angles)
        self.frames.append(T)
        name = f"Frame_{len(self.frames)}"
        frame_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame_geom.transform(T)
        self.scene.scene.add_geometry(name, frame_geom, rendering.MaterialRecord())
        print(f"Frame confirmed: {T}")
        self.temp_origin = None
        self.is_placing = False
        try:
            self.scene.scene.remove_geometry(self.temp_frame_name)
        except Exception:
            pass
        self.scene.force_redraw()

    def compute_transformation(self, origin, angles):
        roll, pitch, yaw = np.radians(angles)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,           0,            1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0,             1, 0],
            [-np.sin(pitch),0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0,            0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        R = Rz @ Ry @ Rx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = origin
        return T

    def pick_point_from_click(self, x, y, threshold=10.0):
        camera = self.scene.scene.camera
        proj_mat = camera.get_projection_matrix()
        view_mat = camera.get_view_matrix()
        M = proj_mat @ view_mat
        points = np.asarray(self.pcd.points)
        N = points.shape[0]
        points_h = np.hstack([points, np.ones((N, 1))])
        clip = (M @ points_h.T).T
        valid = clip[:, 3] != 0
        if not np.any(valid):
            return None
        clip = clip[valid]
        ndc = clip[:, :3] / clip[:, 3:4]
        rect = self.scene.frame
        width = rect.width
        height = rect.height
        screen_x = (ndc[:, 0] + 1) / 2.0 * width
        screen_y = (1 - ndc[:, 1]) / 2.0 * height
        coords = np.stack([screen_x, screen_y], axis=1)
        click = np.array([x, y])
        dists = np.linalg.norm(coords - click, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < threshold:
            valid_indices = np.where(valid)[0]
            return points[valid_indices[min_idx]]
        else:
            return None

    def on_mouse(self, event):
        if not self.is_placing:
            return gui.Widget.EventCallbackResult.IGNORED
        if (event.type == gui.MouseEvent.Type.BUTTON_DOWN and
                event.is_button_down(gui.MouseButton.LEFT)):
            x = event.x
            y = event.y
            print(f"Mouse click at: ({x}, {y})")
            picked_point = self.pick_point_from_click(x, y)
            if picked_point is not None:
                self.temp_origin = picked_point
                print(f"Picked point: {self.temp_origin}")
                T = self.compute_transformation(self.temp_origin, self.temp_angles)
                temp_frame_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                temp_frame_geom.transform(T)
                try:
                    self.scene.scene.remove_geometry(self.temp_frame_name)
                except Exception:
                    pass
                self.scene.scene.add_geometry(self.temp_frame_name, temp_frame_geom, rendering.MaterialRecord())
                self.scene.force_redraw()
                return gui.Widget.EventCallbackResult.HANDLED
            else:
                print("No valid point was picked.")
        return gui.Widget.EventCallbackResult.IGNORED

def main():
    pcd = o3d.io.read_point_cloud(o3d.data.PCDPointCloud().path)
    gui.Application.instance.initialize()
    app = FramePlacerApp(pcd)
    gui.Application.instance.run()
    # When the GUI exits (via the Exit button or window close),
    # print the final frames.
    print("Final frames:")
    for i, T in enumerate(app.frames):
        print(f"Frame {i}:\n{T}")

if __name__ == "__main__":
    main()
