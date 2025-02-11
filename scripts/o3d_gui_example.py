import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

class FramePlacerApp:
    def __init__(self, pcd):
        self.pcd = pcd

        # Create the main window.
        self.window = gui.Application.instance.create_window("Coordinate Frame Placer", 1024, 768)
        self.window.set_on_layout(self.on_layout)

        # Create the SceneWidget and add it as a child.
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene)

        # Create a control panel as a simple vertical container.
        em = self.window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em))
        # (Add any controls you want to the panel; here we add a simple label.)
        self.panel.add_child(gui.Label("Control Panel"))
        # For example, you could add a button:
        # btn = gui.Button("Test Button")
        # self.panel.add_child(btn)
        self.window.add_child(self.panel)

        # Create and configure the material for the point cloud.
        pcd_material = rendering.MaterialRecord()
        pcd_material.shader = "defaultUnlit"
        pcd_material.point_size = 3.0
        self.pcd_material = pcd_material

        # Add the point cloud to the scene.
        self.scene.scene.add_geometry("PointCloud", self.pcd, pcd_material)
        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())
        self.scene.scene.set_background([0.2, 0.2, 0.2, 1.0])

    def on_layout(self, layout_context):
        # Get the available content rectangle of the window.
        r = self.window.content_rect

        # Define a fixed width for the control panel (e.g., 200 pixels).
        panel_width = 200

        # Position the panel on the right side of the window.
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)

        # Position the scene widget to occupy the rest of the window.
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)

if __name__ == "__main__":
    import sys
    gui.Application.instance.initialize()

    # Load a point cloud: use the first command-line argument if provided,
    # otherwise load Open3D’s default example point cloud.
    if len(sys.argv) > 1:
        pcd_path = sys.argv[1]
    else:
        pcd_data = o3d.data.PCDPointCloud()
        pcd_path = pcd_data.path
    pcd = o3d.io.read_point_cloud(pcd_path)

    app = FramePlacerApp(pcd)
    gui.Application.instance.run()
