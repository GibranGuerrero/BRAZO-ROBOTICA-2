import sys
import numpy as np
import serial
import serial.tools.list_ports

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QComboBox, QFrame,
                             QScrollArea, QGroupBox, QGridLayout, QButtonGroup, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d

class RobotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(6, 6), dpi=100, facecolor='#f8f9fa')
        self.axes = fig.add_subplot(111, projection='3d')
        super().__init__(fig)
        
        self.curr_q = [0.0, 0.0, 0.0, 0.0]
        self.target_q = [0.0, 0.0, 0.0, 0.0]
        self.joint_coords = []
        
        self.annot = self.axes.annotate("", xy=(0,0), xytext=(15,15),
                                        textcoords="offset points",
                                        bbox=dict(boxstyle="round", fc="#f1c40f", alpha=0.9),
                                        arrowprops=dict(arrowstyle="->", color='black'))
        self.annot.set_visible(False)
        
        self.mpl_connect("motion_notify_event", self.on_hover)
        self.setup_plot()
        self.update_view(0, 0, 0, 0, "Denavit-Hartenberg")

    def setup_plot(self):
        self.axes.set_xlim([-40, 40])
        self.axes.set_ylim([-40, 40])
        self.axes.set_zlim([0, 40])
        self.axes.set_xlabel('X (cm)')
        self.axes.set_ylabel('Y (cm)')
        self.axes.set_zlabel('Z (cm)')
        self.axes.set_title("Gemelo Digital Manipulador")

    def get_frame_axes(self, q1, q2, q3, q4):
        """Calcula las matrices de rotación para cada frame DH (0 a 4).
        Para frames 1-4: Y y Z están invertidos 180° respecto al cálculo DH estándar.
        """
        t1, t2, t3, t4 = np.radians([q1, q2, q3, q4])

        # Frame 0: Base - identidad (sin cambios)
        R0 = np.eye(3)

        # Frame 1
        x1 = np.array([np.cos(t1)*np.cos(t2), np.sin(t1)*np.cos(t2), np.sin(t2)])
        z1 = -np.array([-np.sin(t1), np.cos(t1), 0])  # invertido
        y1 = -np.cross(np.array([-np.sin(t1), np.cos(t1), 0]), x1)  # invertido
        R1 = np.column_stack([x1, y1, z1])

        # Frame 2
        t23 = t2 + t3
        x2 = np.array([np.cos(t1)*np.cos(t23), np.sin(t1)*np.cos(t23), np.sin(t23)])
        z2 = -np.array([-np.sin(t1), np.cos(t1), 0])
        y2 = -np.cross(np.array([-np.sin(t1), np.cos(t1), 0]), x2)
        R2 = np.column_stack([x2, y2, z2])

        # Frame 3
        t234 = t2 + t3 + t4
        x3 = np.array([np.cos(t1)*np.cos(t234), np.sin(t1)*np.cos(t234), np.sin(t234)])
        z3 = -np.array([-np.sin(t1), np.cos(t1), 0])
        y3 = -np.cross(np.array([-np.sin(t1), np.cos(t1), 0]), x3)
        R3 = np.column_stack([x3, y3, z3])

        # Frame 4
        x4 = np.array([np.cos(t1)*np.cos(t234), np.sin(t1)*np.cos(t234), np.sin(t234)])
        z4 = -np.array([-np.sin(t1), np.cos(t1), 0])
        y4 = -np.cross(np.array([-np.sin(t1), np.cos(t1), 0]), x4)
        R4 = np.column_stack([x4, y4, z4])

        return [R0, R1, R2, R3, R4]

    def get_matrices_dh(self, q1, q2, q3, q4):
        l12, l3, l4, l5, l6 = 8.7576, 1.05, 10.4, 8.8, 6.95
        t1, t2, t3, t4 = np.radians([q1, q2, q3, q4])
        
        p0 = np.array([0, 0, 0])
        p1 = np.array([l3 * np.cos(t1), l3 * np.sin(t1), l12])
        p2 = np.array([np.cos(t1)*(l3 + l4*np.cos(t2)), np.sin(t1)*(l3 + l4*np.cos(t2)), l12 + l4*np.sin(t2)])
        p3 = np.array([np.cos(t1)*(l3 + l4*np.cos(t2) + l5*np.cos(t2+t3)), 
                       np.sin(t1)*(l3 + l4*np.cos(t2) + l5*np.cos(t2+t3)), 
                       l12 + l4*np.sin(t2) + l5*np.sin(t2+t3)])
        p4 = np.array([np.cos(t1)*(l3 + l4*np.cos(t2) + l5*np.cos(t2+t3) + l6*np.cos(t2+t3+t4)),
                       np.sin(t1)*(l3 + l4*np.cos(t2) + l5*np.cos(t2+t3) + l6*np.cos(t2+t3+t4)),
                       l12 + l4*np.sin(t2) + l5*np.sin(t2+t3) + l6*np.sin(t2+t3+t4)])

        t234 = t2 + t3 + t4
        n = [np.cos(t1)*np.cos(t234), np.sin(t1)*np.cos(t234), np.sin(t234)]
        o = [-np.cos(t1)*np.sin(t234), -np.sin(t1)*np.sin(t234), np.cos(t234)]
        a = [np.sin(t1), -np.cos(t1), 0]
        return [p0, p1, p2, p3, p4], (n, o, a, p4)

    def get_matrices_geometric(self, q1, q2, q3, q4):
        l12, l3, l4, l5, l6 = 8.7576, 1.05, 10.4, 8.8, 6.95
        t1, t2, t3, t4 = np.radians([q1, q2, q3, q4])
        theta = t2 + t3 + t4 
        
        r = l3 + l4*np.cos(t2) + l5*np.cos(t2 + t3) + l6*np.cos(theta)
        px = r * np.cos(t1)
        py = r * np.sin(t1)
        pz = l12 + l4*np.sin(t2) + l5*np.sin(t2 + t3) + l6*np.sin(theta)
        p4 = np.array([px, py, pz])

        p0 = np.array([0, 0, 0])
        p1 = np.array([l3*np.cos(t1), l3*np.sin(t1), l12])
        p2 = np.array([(l3 + l4*np.cos(t2))*np.cos(t1), (l3 + l4*np.cos(t2))*np.sin(t1), l12 + l4*np.sin(t2)])
        p3 = np.array([(l3 + l4*np.cos(t2) + l5*np.cos(t2+t3))*np.cos(t1), 
                       (l3 + l4*np.cos(t2) + l5*np.cos(t2+t3))*np.sin(t1), 
                       l12 + l4*np.sin(t2) + l5*np.sin(t2+t3)])

        n = [np.cos(t1)*np.cos(theta), np.sin(t1)*np.cos(theta), np.sin(theta)]
        o = [-np.sin(theta)*np.cos(t1), -np.sin(t1)*np.sin(theta), np.cos(theta)]
        a = [np.sin(t1), -np.cos(t1), 0]
        
        return [p0, p1, p2, p3, p4], (n, o, a, p4)

    def update_view(self, q1, q2, q3, q4, method):
        self.axes.cla()
        self.setup_plot()
        
        if method == "Denavit-Hartenberg":
            self.joint_coords, noap = self.get_matrices_dh(q1, q2, q3, q4)
        else:
            self.joint_coords, noap = self.get_matrices_geometric(q1, q2, q3, q4)
        
        x = [p[0] for p in self.joint_coords]
        y = [p[1] for p in self.joint_coords]
        z = [p[2] for p in self.joint_coords]

        self.axes.plot(x, y, z, '-o', linewidth=5, markersize=8, color='#2c3e50', markerfacecolor='#e74c3c')
        self.axes.scatter(x[-1], y[-1], z[-1], color='red', s=100)

        # ---- Dibujar sistemas de coordenadas para cada frame ----
        frames = self.get_frame_axes(q1, q2, q3, q4)
        axis_len = 4.0

        for i, (origin, R) in enumerate(zip(self.joint_coords, frames)):
            ox, oy, oz = origin[0], origin[1], origin[2]
            x_dir = R[:, 0] * axis_len
            y_dir = R[:, 1] * axis_len
            z_dir = R[:, 2] * axis_len

            # Eje X → Azul
            self.axes.quiver(ox, oy, oz, x_dir[0], x_dir[1], x_dir[2],
                             color='#2196F3', arrow_length_ratio=0.15, linewidth=1.8)
            self.axes.text(ox + x_dir[0]*1.15, oy + x_dir[1]*1.15, oz + x_dir[2]*1.15,
                           f'X{i}', color='#2196F3', fontsize=7, fontweight='bold')

            # Eje Y → Verde
            self.axes.quiver(ox, oy, oz, y_dir[0], y_dir[1], y_dir[2],
                             color='#4CAF50', arrow_length_ratio=0.15, linewidth=1.8)
            self.axes.text(ox + y_dir[0]*1.15, oy + y_dir[1]*1.15, oz + y_dir[2]*1.15,
                           f'Y{i}', color='#4CAF50', fontsize=7, fontweight='bold')

            # Eje Z → Rojo
            self.axes.quiver(ox, oy, oz, z_dir[0], z_dir[1], z_dir[2],
                             color='#F44336', arrow_length_ratio=0.15, linewidth=1.8)
            self.axes.text(ox + z_dir[0]*1.15, oy + z_dir[1]*1.15, oz + z_dir[2]*1.15,
                           f'Z{i}', color='#F44336', fontsize=7, fontweight='bold')

        self.annot = self.axes.annotate("", xy=(0,0), xytext=(15,15),
                                        textcoords="offset points",
                                        bbox=dict(boxstyle="round", fc="#f1c40f", alpha=0.9),
                                        arrowprops=dict(arrowstyle="->", color='black'))
        self.annot.set_visible(False)
        self.draw()
        return noap

    def on_hover(self, event):
        if event.inaxes != self.axes:
            if self.annot.get_visible():
                self.annot.set_visible(False)
                self.draw_idle()
            return

        for i, p in enumerate(self.joint_coords):
            x2d, y2d, _ = proj3d.proj_transform(p[0], p[1], p[2], self.axes.get_proj())
            x_pix, y_pix = self.axes.transData.transform((x2d, y2d))
            dist = np.hypot(event.x - x_pix, event.y - y_pix)
            
            if dist < 20:
                label = f"Articulación Q{i}" if i > 0 else "Base"
                self.annot.set_text(f"{label}\nX: {p[0]:.2f}\nY: {p[1]:.2f}\nZ: {p[2]:.2f}")
                self.annot.xy = (x2d, y2d)
                self.annot.set_visible(True)
                self.draw_idle()
                return

        if self.annot.get_visible():
            self.annot.set_visible(False)
            self.draw_idle()

    def reset_view(self):
        self.axes.view_init(elev=30, azim=-60)
        self.draw_idle()


class ValidationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.arduino = None
        self.matrix_dh_data = None
        self.matrix_geo_data = None
        self.initUI()
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.animate_step)
        self.step_count = 0
        self.send_to_arduino = False

    def initUI(self):
        layout = QHBoxLayout()

        # ============================================================
        # PANEL IZQUIERDO: ÚNICO SCROLL con Directa + Inversa en paralelo
        # ============================================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
        """)

        scroll_content = QWidget()
        main_scroll_layout = QVBoxLayout(scroll_content)
        main_scroll_layout.setContentsMargins(10, 10, 10, 10)


        # Conexión Serial (compartida)
        main_scroll_layout.addWidget(QLabel("<b> CONEXIÓN SERIAL</b>"))
        self.combo = QComboBox()
        for p in serial.tools.list_ports.comports():
            self.combo.addItem(p.device)
        main_scroll_layout.addWidget(self.combo)
        self.btn_conn = QPushButton("CONECTAR")
        self.btn_conn.clicked.connect(self.toggle_serial)
        main_scroll_layout.addWidget(self.btn_conn)

        # Separador principal
        main_scroll_layout.addSpacing(15)
        sep_main = QFrame()
        sep_main.setFrameShape(QFrame.HLine)
        sep_main.setStyleSheet("color: #bdc3c7;")
        main_scroll_layout.addWidget(sep_main)
        main_scroll_layout.addSpacing(10)

        # Layout horizontal para las dos columnas paralelas
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(25)

        # ==================== COLUMNA IZQUIERDA: CINEMÁTICA DIRECTA ====================
        directa_widget = QWidget()
        directa_panel = QVBoxLayout(directa_widget)
        directa_panel.setContentsMargins(0, 0, 0, 0)

        title_dir = QLabel("<b>CINEMÁTICA DIRECTA</b>")
        title_dir.setAlignment(Qt.AlignCenter)
        title_dir.setStyleSheet("font-size: 13px; color: #2980b9; margin-bottom: 10px;")
        directa_panel.addWidget(title_dir)

        # Selección de Método directa
        directa_panel.addWidget(QLabel("<b> MÉTODO DE CÁLCULO</b>"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Denavit-Hartenberg", "Método Geométrico"])
        self.method_combo.setStyleSheet("height: 30px; font-weight: bold;")
        directa_panel.addWidget(self.method_combo)

        # Entradas con restricciones de límites numéricos
        directa_panel.addSpacing(10)
        directa_panel.addWidget(QLabel("<b> POSICIÓN DESEADA (DEG)</b>"))
        self.inputs = [QLineEdit("0") for _ in range(4)]
        
        self.ranges = [(0.0, 180.0), (0.0, 180.0), (-90.0, 90.0), (-90.0, 90.0)]
        labels_text = [
            "Eje q1 (0 a 180°):",
            "Eje q2 (0 a 180°):",
            "Eje q3 (-90 a 90°):",
            "Eje q4 (-90 a 90°):"
        ]
        
        for i, le in enumerate(self.inputs):
            directa_panel.addWidget(QLabel(labels_text[i]))
            low, high = self.ranges[i]
            validador = QDoubleValidator(low, high, 2)
            validador.setNotation(QDoubleValidator.StandardNotation)
            le.setValidator(validador)
            directa_panel.addWidget(le)

        directa_panel.addSpacing(10)

        # Botón SIMULAR directa
        self.btn_sim = QPushButton("SIMULAR")
        self.btn_sim.setStyleSheet("""
            background: #2980b9; color: white; 
            font-weight: bold; height: 45px; border-radius: 5px;
        """)
        self.btn_sim.clicked.connect(self.start_simulation)
        directa_panel.addWidget(self.btn_sim)

        # Botón EJECUTAR directa
        self.btn_send = QPushButton("EJECUTAR")
        self.btn_send.setStyleSheet("""
            background: #27ae60; color: white; 
            font-weight: bold; height: 45px; border-radius: 5px;
        """)
        self.btn_send.clicked.connect(self.start_motion)
        directa_panel.addWidget(self.btn_send)

        # Etiqueta de estado directa
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-style: italic; color: #7f8c8d; font-size: 9pt;")
        directa_panel.addWidget(self.status_label)

        # Panel NOAP Denavit-Hartenberg
        directa_panel.addSpacing(10)
        dh_title_layout = QHBoxLayout()
        dh_title_layout.addWidget(QLabel("<b> MATRIZ HOMOGÉNEA (NOAP) D-H</b>"))
        self.btn_copy_dh = QPushButton("COPIAR")
        self.btn_copy_dh.setCursor(Qt.PointingHandCursor)
        self.btn_copy_dh.setStyleSheet("""
            QPushButton {
                background-color: #34495e; color: white; font-size: 8pt; font-weight: bold; 
                padding: 3px 8px; border-radius: 4px; border: none;
            }
            QPushButton:hover { background-color: #2c3e50; }
            QPushButton:pressed { background-color: #1a252f; }
        """)
        self.btn_copy_dh.clicked.connect(lambda: self.copy_matrix_to_clipboard("Denavit-Hartenberg", self.btn_copy_dh))
        dh_title_layout.addWidget(self.btn_copy_dh)
        dh_title_layout.addStretch()
        directa_panel.addLayout(dh_title_layout)

        self.noap_dh_display = QLabel()
        self.noap_dh_display.setStyleSheet("""
            background-color: #fdfefe; color: #2c3e50; border-radius: 6px; 
            padding: 8px; font-family: 'Consolas', monospace; font-size: 9pt;
            border: 1px solid #bdc3c7;
        """)
        directa_panel.addWidget(self.noap_dh_display)

        # Panel NOAP Método Geométrico
        directa_panel.addSpacing(10)
        geo_title_layout = QHBoxLayout()
        geo_title_layout.addWidget(QLabel("<b> MATRIZ HOMOGÉNEA (NOAP) GEOMÉTRICA</b>"))
        self.btn_copy_geo = QPushButton("COPIAR")
        self.btn_copy_geo.setCursor(Qt.PointingHandCursor)
        self.btn_copy_geo.setStyleSheet("""
            QPushButton {
                background-color: #34495e; color: white; font-size: 8pt; font-weight: bold; 
                padding: 3px 8px; border-radius: 4px; border: none;
            }
            QPushButton:hover { background-color: #2c3e50; }
            QPushButton:pressed { background-color: #1a252f; }
        """)
        self.btn_copy_geo.clicked.connect(lambda: self.copy_matrix_to_clipboard("Método Geométrico", self.btn_copy_geo))
        geo_title_layout.addWidget(self.btn_copy_geo)
        geo_title_layout.addStretch()
        directa_panel.addLayout(geo_title_layout)

        self.noap_geo_display = QLabel()
        self.noap_geo_display.setStyleSheet("""
            background-color: #fdfefe; color: #2c3e50; border-radius: 6px; 
            padding: 8px; font-family: 'Consolas', monospace; font-size: 9pt;
            border: 1px solid #bdc3c7;
        """)
        directa_panel.addWidget(self.noap_geo_display)
        
        # ==================== COLUMNA DERECHA: CINEMÁTICA INVERSA ====================
        inversa_widget = QWidget()
        inversa_panel = QVBoxLayout(inversa_widget)
        inversa_panel.setContentsMargins(0, 0, 0, 0)

        title_inv = QLabel("<b>CINEMÁTICA INVERSA</b>")
        title_inv.setAlignment(Qt.AlignCenter)
        title_inv.setStyleSheet("font-size: 13px; color: #8e44ad; margin-bottom: 10px;")
        inversa_panel.addWidget(title_inv)

        # Selección de método inverso
        inversa_panel.addWidget(QLabel("<b> MÉTODO DE CÁLCULO</b>"))
        self.inv_method_combo = QComboBox()
        self.inv_method_combo.addItems(["Geométrica", "Analítica"])
        self.inv_method_combo.setStyleSheet("height: 30px; font-weight: bold;")
        inversa_panel.addWidget(self.inv_method_combo)

        # Selección de Configuración de Codo (Arriba/Abajo)
        inversa_panel.addSpacing(5)
        inversa_panel.addWidget(QLabel("<b> CONFIGURACIÓN DE CODO</b>"))
        elbow_layout = QHBoxLayout()
        self.btn_elbow_up = QPushButton("CODO ABAJO")
        self.btn_elbow_up.setCheckable(True)
        self.btn_elbow_up.setChecked(True)
        self.btn_elbow_up.setCursor(Qt.PointingHandCursor)
        self.btn_elbow_up.setStyleSheet("""
            QPushButton {
                background-color: #bdc3c7; color: #2c3e50; font-weight: bold; height: 35px; border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #8e44ad; color: white;
            }
        """)

        self.btn_elbow_down = QPushButton("CODO ARRIBA")
        self.btn_elbow_down.setCheckable(True)
        self.btn_elbow_down.setCursor(Qt.PointingHandCursor)
        self.btn_elbow_down.setStyleSheet("""
            QPushButton {
                background-color: #bdc3c7; color: #2c3e50; font-weight: bold; height: 35px; border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #8e44ad; color: white;
            }
        """)

        self.elbow_group = QButtonGroup(self)
        self.elbow_group.addButton(self.btn_elbow_up)
        self.elbow_group.addButton(self.btn_elbow_down)
        self.elbow_group.setExclusive(True)

        elbow_layout.addWidget(self.btn_elbow_up)
        elbow_layout.addWidget(self.btn_elbow_down)
        inversa_panel.addLayout(elbow_layout)

        # Matriz NOAP editable
        inversa_panel.addSpacing(10)
        matrix_group = QGroupBox("NOAP EDITABLE")
        matrix_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #8e44ad;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #8e44ad;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
        """)
        matrix_group_layout = QVBoxLayout(matrix_group)
        matrix_group_layout.setContentsMargins(5, 5, 5, 5)
        
        # Botón de Pegar Matriz
        pegar_layout = QHBoxLayout()
        self.btn_paste_matrix = QPushButton("PEGAR MATRIZ 4x4")
        self.btn_paste_matrix.setCursor(Qt.PointingHandCursor)
        self.btn_paste_matrix.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad; color: white; font-size: 8.5pt; font-weight: bold; 
                padding: 4px 12px; border-radius: 4px; border: none; height: 22px;
            }
            QPushButton:hover { background-color: #7d3c98; }
            QPushButton:pressed { background-color: #5b2c6f; }
        """)
        self.btn_paste_matrix.clicked.connect(self.paste_matrix_from_clipboard)
        pegar_layout.addStretch()
        pegar_layout.addWidget(self.btn_paste_matrix)
        pegar_layout.addStretch()
        matrix_group_layout.addLayout(pegar_layout)
        
        matrix_grid_widget = QWidget()
        matrix_grid = QGridLayout(matrix_grid_widget)
        matrix_grid.setContentsMargins(5, 5, 5, 5)
        matrix_grid.setSpacing(5)

        self.matrix_inputs = []
        default_vals = [
            ["1.0", "0.0", "0.0", "0.0"],
            ["0.0", "0.0", "-1.0", "0.0"],
            ["0.0", "1.0", "0.0", "8.76"]
        ]
        
        for r in range(3):
            row_inputs = []
            for c in range(4):
                le = QLineEdit(default_vals[r][c])
                le.setStyleSheet("""
                    QLineEdit {
                        background-color: #fdfefe;
                        color: #2c3e50;
                        border: 1px solid #bdc3c7;
                        border-radius: 3px;
                        font-family: 'Consolas', monospace;
                        font-size: 9pt;
                        min-width: 45px;
                        max-width: 65px;
                        height: 25px;
                        text-align: center;
                    }
                    QLineEdit:focus {
                        border: 1px solid #8e44ad;
                        background-color: #fcf3cf;
                    }
                """)
                le.setAlignment(Qt.AlignCenter)
                validador = QDoubleValidator(-180.0, 180.0, 4)
                validador.setNotation(QDoubleValidator.StandardNotation)
                le.setValidator(validador)
                matrix_grid.addWidget(le, r, c)
                row_inputs.append(le)
            self.matrix_inputs.append(row_inputs)

        # Última fila fija (0 0 0 1)
        for c, val_str in enumerate(["0", "0", "0", "1"]):
            lbl = QLabel(val_str)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("""
                font-family: 'Consolas', monospace;
                font-weight: bold;
                font-size: 9pt;
                color: #7f8c8d;
                height: 25px;
            """)
            matrix_grid.addWidget(lbl, 3, c)

        matrix_group_layout.addWidget(matrix_grid_widget)
        inversa_panel.addWidget(matrix_group)
        inversa_panel.addSpacing(10)

        # Botón SIMULAR inversa
        self.btn_inv_sim = QPushButton("SIMULAR")
        self.btn_inv_sim.setStyleSheet("""
            background: #8e44ad; color: white; 
            font-weight: bold; height: 45px; border-radius: 5px;
        """)
        self.btn_inv_sim.clicked.connect(self.start_inverse_simulation)
        inversa_panel.addWidget(self.btn_inv_sim)

        # Botón EJECUTAR inversa
        self.btn_inv_send = QPushButton("EJECUTAR")
        self.btn_inv_send.setStyleSheet("""
            background: #27ae60; color: white; 
            font-weight: bold; height: 45px; border-radius: 5px;
        """)
        self.btn_inv_send.clicked.connect(self.start_inverse_motion)
        inversa_panel.addWidget(self.btn_inv_send)

        # Etiqueta de estado inversa
        self.inv_status_label = QLabel("")
        self.inv_status_label.setAlignment(Qt.AlignCenter)
        self.inv_status_label.setStyleSheet("font-style: italic; color: #7f8c8d; font-size: 9pt;")
        inversa_panel.addWidget(self.inv_status_label)

        # Panel de ángulos de salida: Geométrica
        inversa_panel.addSpacing(10)
        inversa_panel.addWidget(QLabel("<b> ÁNGULOS GEOMÉTRICA (DEG)</b>"))
        self.inv_angles_geo_display = QLabel()
        self.inv_angles_geo_display.setStyleSheet("""
            background-color: #fdfefe; color: #2c3e50; border-radius: 6px; 
            padding: 8px; font-family: 'Consolas', monospace; font-size: 9pt;
            border: 1px solid #bdc3c7;
        """)
        inversa_panel.addWidget(self.inv_angles_geo_display)

        # Panel de ángulos de salida: Analítica
        inversa_panel.addSpacing(10)
        inversa_panel.addWidget(QLabel("<b> ÁNGULOS ANALÍTICA (DEG)</b>"))
        self.inv_angles_ana_display = QLabel()
        self.inv_angles_ana_display.setStyleSheet("""
            background-color: #fdfefe; color: #2c3e50; border-radius: 6px; 
            padding: 8px; font-family: 'Consolas', monospace; font-size: 9pt;
            border: 1px solid #bdc3c7;
        """)
        inversa_panel.addWidget(self.inv_angles_ana_display)

        # Añadir ambos widgets al layout de columnas
        columns_layout.addWidget(directa_widget, 1)
        columns_layout.addWidget(inversa_widget, 1)

        main_scroll_layout.addLayout(columns_layout)
        main_scroll_layout.addStretch()

        self.update_noap_table([1,0,0], [0,1,0], [0,0,1], [0,0,0], "Denavit-Hartenberg")
        self.update_noap_table([1,0,0], [0,1,0], [0,0,1], [0,0,0], "Método Geométrico")
        self.update_inv_angles_table([0, 0, 0, 0], "Geométrica")
        self.update_inv_angles_table([0, 0, 0, 0], "Analítica")

        # ============================================================
        # Ensamblar: scroll a la izquierda, canvas a la derecha
        # ============================================================
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 2)

        canvas_layout = QVBoxLayout()
        self.canvas = RobotCanvas(self)
        canvas_layout.addWidget(self.canvas)
        
        btn_reset_view = QPushButton("🔄 UBICAR VISTA 3D")
        btn_reset_view.setStyleSheet("""
            QPushButton {
                background: #34495e; color: white; 
                font-weight: bold; height: 40px; border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2c3e50; }
        """)
        btn_reset_view.setCursor(Qt.PointingHandCursor)
        btn_reset_view.clicked.connect(self.canvas.reset_view)
        canvas_layout.addWidget(btn_reset_view)
        
        layout.addLayout(canvas_layout, 3)
        self.setLayout(layout)

    def copy_matrix_to_clipboard(self, method, button):
        data = getattr(self, 'matrix_dh_data' if method == "Denavit-Hartenberg" else 'matrix_geo_data', None)
        if not data:
            return
        n, o, a, p = data
        rows = []
        for i in range(3):
            rows.append(f"{n[i]:.4f}\t{o[i]:.4f}\t{a[i]:.4f}\t{p[i]:.4f}")
        rows.append("0.0000\t0.0000\t0.0000\t1.0000")
        QApplication.clipboard().setText("\n".join(rows))
        
        # Retroalimentación visual
        button.setText("¡COPIADO!")
        button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; color: white; font-size: 8pt; font-weight: bold; 
                padding: 3px 8px; border-radius: 4px; border: none;
            }
        """)
        QTimer.singleShot(1000, lambda: (
            button.setText("COPIAR"),
            button.setStyleSheet("""
                QPushButton {
                    background-color: #34495e; color: white; font-size: 8pt; font-weight: bold; 
                    padding: 3px 8px; border-radius: 4px; border: none;
                }
                QPushButton:hover { background-color: #2c3e50; }
                QPushButton:pressed { background-color: #1a252f; }
            """)
        ))

    def paste_matrix_from_clipboard(self):
        text = QApplication.clipboard().text().strip()
        if not text:
            return
        
        import re
        text_clean = re.sub(r'(\d),(\d)', r'\1.\2', text)
        num_pattern = re.compile(r'[-+]?\d*\.\d+|\d+')
        tokens = num_pattern.findall(text_clean)
        
        if len(tokens) < 12:
            QMessageBox.warning(self, "Error al Pegar", 
                                f"El portapapeles no contiene suficientes números válidos para una matriz 3x4 o 4x4.\n"
                                f"Se encontraron {len(tokens)} números, se necesitan al menos 12.")
            return
        
        idx = 0
        for r in range(3):
            for c in range(4):
                val = float(tokens[idx])
                self.matrix_inputs[r][c].setText(f"{val:.4f}")
                idx += 1
                
        # Retroalimentación visual
        self.btn_paste_matrix.setText("¡PEGADO!")
        self.btn_paste_matrix.setStyleSheet("""
            QPushButton {
                background-color: #27ae60; color: white; font-size: 8.5pt; font-weight: bold; 
                padding: 4px 12px; border-radius: 4px; border: none; height: 22px;
            }
        """)
        QTimer.singleShot(1000, lambda: (
            self.btn_paste_matrix.setText("PEGAR MATRIZ 4x4"),
            self.btn_paste_matrix.setStyleSheet("""
                QPushButton {
                    background-color: #8e44ad; color: white; font-size: 8.5pt; font-weight: bold; 
                    padding: 4px 12px; border-radius: 4px; border: none; height: 22px;
                }
                QPushButton:hover { background-color: #7d3c98; }
                QPushButton:pressed { background-color: #5b2c6f; }
            """)
        ))

    def update_noap_table(self, n, o, a, p, method):
        """Genera el HTML estructurado y actualiza selectivamente el panel del método activo"""
        html = "<table width='100%' cellspacing='6' style='text-align: center;'>"
        html += "<tr><th style='color:#3498db'>n</th><th style='color:#3498db'>o</th><th style='color:#3498db'>a</th><th style='color:#3498db'>p</th></tr>"
        for i in range(3):
            html += f"<tr><td>{n[i]:.2f}</td><td>{o[i]:.2f}</td><td>{a[i]:.2f}</td><td style='color:#c0392b; font-weight:bold;'>{p[i]:.2f}</td></tr>"
        html += "<tr><td>0.00</td><td>0.00</td><td>0.00</td><td>1.00</td></tr></table>"
        
        # Filtrado de actualización según el método seleccionado
        if method == "Denavit-Hartenberg":
            self.noap_dh_display.setText(html)
            self.matrix_dh_data = (n, o, a, p)
        else:
            self.noap_geo_display.setText(html)
            self.matrix_geo_data = (n, o, a, p)

    def update_inv_angles_table(self, angles, method):
        """Genera el HTML con los ángulos resultantes de la cinemática inversa en formato descendente"""
        labels = ["q1", "q2", "q3", "q4"]
        html = "<table width='100%' cellspacing='6' style='text-align: left; padding-left: 20px; font-size: 10pt;'>"
        for i, val in enumerate(angles):
            html += f"<tr><td style='font-weight:bold; color:#8e44ad; width: 60px;'>{labels[i]}:</td><td style='font-weight:bold; color:#2c3e50;'>{val:.2f}°</td></tr>"
        html += "</table>"
        
        if method == "Geométrica":
            self.inv_angles_geo_display.setText(html)
        else:
            self.inv_angles_ana_display.setText(html)

    # ============================================================
    # CINEMÁTICA INVERSA - MÉTODOS DE CÁLCULO
    # ============================================================

    def inverse_geometric(self, T, codo_arriba=True):
        """Cinemática inversa por método geométrico.
        Entrada: T (matriz de transformación homogénea 3x4 en cm).
        Retorna: (q1, q2, q3, q4) en grados o None si no hay solución."""
        # ---- Longitudes (cm) ----
        L1 = 5.8576
        L2 = 2.9
        L3 = 1.05
        L4 = 10.4
        L5 = 8.8
        L6 = 6.95

        # Extraer vectores de T
        n_vec = T[0:3, 0].copy()
        # Normalizar n_vec por seguridad
        norm_n = np.linalg.norm(n_vec)
        if norm_n > 1e-6:
            n_vec /= norm_n
        else:
            n_vec = np.array([1.0, 0.0, 0.0])

        P = T[0:3, 3]  # [px, py, pz]

        # =====================================================
        # THETA 1
        # =====================================================
        theta1 = np.arctan2(P[1], P[0])

        # =====================================================
        # CENTRO DE MUÑECA P3 (restamos L6 en direccion n)
        # =====================================================
        P3 = P - L6 * n_vec
        P3x, P3y, P3z = P3[0], P3[1], P3[2]

        r = np.sqrt(P3x**2 + P3y**2)
        r_prima = r - L3
        z_base = L1 + L2
        dz = P3z - z_base
        h = np.sqrt(r_prima**2 + dz**2)

        # =====================================================
        # K (Ley de cosenos)
        # =====================================================
        if (2 * L4 * L5) == 0:
            return None
        K = (h**2 - L4**2 - L5**2) / (2 * L4 * L5)

        # Tolerancia numérica
        tol = 1e-5
        if K > 1.0 and (K - 1.0) < tol:
            K = 1.0
        elif K < -1.0 and (-1.0 - K) < tol:
            K = -1.0
        elif abs(K) > 1.0:
            return None  # Fuera de alcance

        # =====================================================
        # THETA 3 (Codo Arriba o Abajo)
        # =====================================================
        if codo_arriba:
            theta3 = np.arctan2(np.sqrt(max(0.0, 1.0 - K**2)), K)
        else:
            theta3 = np.arctan2(-np.sqrt(max(0.0, 1.0 - K**2)), K)

        # =====================================================
        # THETA 2
        # =====================================================
        beta  = np.arctan2(dz, r_prima)
        alpha = np.arctan2(L5 * np.sin(theta3), L4 + L5 * np.cos(theta3))
        theta2 = beta - alpha

        # =====================================================
        # THETA GLOBAL — angulo de n_vec en el plano vertical
        # del robot (plano donde viven los eslabones 2, 3 y 4)
        # =====================================================
        n_r = np.cos(theta1) * T[0, 0] + np.sin(theta1) * T[1, 0]  # componente radial
        n_z = T[2, 0]                                                 # componente vertical
        Theta_global = np.arctan2(n_z, n_r)

        # =====================================================
        # THETA 4
        # =====================================================
        theta4 = Theta_global - theta2 - theta3
        theta4 = np.arctan2(np.sin(theta4), np.cos(theta4))  # normalizar en [-pi, pi]

        # Convertir a grados
        q1 = np.degrees(theta1)
        q2 = np.degrees(theta2)
        q3 = np.degrees(theta3)
        q4 = np.degrees(theta4)

        return (q1, q2, q3, q4)
    
    def inverse_analytical(self, T, codo_arriba=True):
        """Cinemática inversa por método analítico (Método de Paul).
        Entrada: T (matriz de transformación homogénea 3x4 en cm).
        Retorna: (q1, q2, q3, q4) en grados o None si no hay solución."""
        # ---- Longitudes (cm) ----
        L1 = 5.8576
        L2 = 2.9
        L3 = 1.05
        L4 = 10.4
        L5 = 8.8
        L6 = 6.95

        nx = T[0, 0]
        ny = T[1, 0]
        nz = T[2, 0]
        px = T[0, 3]
        py = T[1, 3]
        pz = T[2, 3]

        # =====================================================
        # THETA 1
        # =====================================================
        theta1 = np.arctan2(py, px)

        # =====================================================
        # MATRIZ INTERMEDIA M = (^0A_1)^-1 * T
        # =====================================================
        M11 = np.cos(theta1)*nx + np.sin(theta1)*ny
        M21 = nz
        M14 = np.cos(theta1)*px + np.sin(theta1)*py - L3
        M24 = pz - L1 - L2

        # =====================================================
        # ORIENTACIÓN TOTAL EN EL PLANO (THETA 234)
        # =====================================================
        theta234 = np.arctan2(M21, M11)

        # =====================================================
        # COORDENADAS DESACOPLADAS (X, Y)
        # =====================================================
        X = M14 - L6 * np.cos(theta234)
        Y = M24 - L6 * np.sin(theta234)

        # =====================================================
        # K (Ley de cosenos para THETA 3)
        # =====================================================
        if (2 * L4 * L5) == 0:
            return None
        cos_t3 = (X**2 + Y**2 - L4**2 - L5**2) / (2 * L4 * L5)

        # Tolerancia numérica para evitar errores fuera del dominio de la raíz
        tol = 1e-5
        if cos_t3 > 1.0 and (cos_t3 - 1.0) < tol:
            cos_t3 = 1.0
        elif cos_t3 < -1.0 and (-1.0 - cos_t3) < tol:
            cos_t3 = -1.0
        elif abs(cos_t3) > 1.0:
            return None # Fuera de alcance

        # =====================================================
        # THETA 3 (Codo Arriba o Codo Abajo)
        # =====================================================
        if codo_arriba:
            sin_t3 = np.sqrt(max(0.0, 1.0 - cos_t3**2))
        else:
            sin_t3 = -np.sqrt(max(0.0, 1.0 - cos_t3**2))
        theta3 = np.arctan2(sin_t3, cos_t3)

        # =====================================================
        # THETA 2
        # =====================================================
        num_sin2 = Y * (L4 + L5 * cos_t3) - X * (L5 * sin_t3)
        num_cos2 = X * (L4 + L5 * cos_t3) + Y * (L5 * sin_t3)
        theta2 = np.arctan2(num_sin2, num_cos2)

        # =====================================================
        # THETA 4
        # =====================================================
        theta4 = theta234 - theta2 - theta3

        # Normalizar theta4 en el rango [-pi, pi] para evitar saltos bruscos
        theta4 = np.arctan2(np.sin(theta4), np.cos(theta4))

        # Convertir a grados
        q1 = np.degrees(theta1)
        q2 = np.degrees(theta2)
        q3 = np.degrees(theta3)
        q4 = np.degrees(theta4)

        return (q1, q2, q3, q4)

    # ============================================================
    # MANEJO DE ENTRADAS Y ACCIONES DE INVERSA
    # ============================================================

    def get_inv_inputs(self):
        """Lee y procesa la matriz NOAP editable y retorna la matriz 3x4 de numpy (en cm)"""
        mat = np.zeros((3, 4))
        for r in range(3):
            for c in range(4):
                try:
                    mat[r, c] = float(self.matrix_inputs[r][c].text())
                except ValueError:
                    mat[r, c] = 0.0
        return mat

    def compute_inverse(self):
        """Calcula la cinemática inversa con ambos métodos y actualiza los paneles de ángulos.
        Retorna los ángulos del método seleccionado actualmente o None si falla."""
        T = self.get_inv_inputs()
        codo_arriba = self.btn_elbow_up.isChecked()

        # Calcular por ambos métodos
        result_geo = self.inverse_geometric(T, codo_arriba)
        result_ana = self.inverse_analytical(T, codo_arriba)

        # Actualizar panel geométrica
        if result_geo is not None:
            self.update_inv_angles_table(list(result_geo), "Geométrica")
        else:
            self.inv_angles_geo_display.setText(
                "<p style='color:#e74c3c; text-align:center; font-weight:bold;'>⚠ Fuera de alcance</p>")

        # Actualizar panel analítica
        if result_ana is not None:
            self.update_inv_angles_table(list(result_ana), "Analítica")
        else:
            self.inv_angles_ana_display.setText(
                "<p style='color:#e74c3c; text-align:center; font-weight:bold;'>⚠ Fuera de alcance</p>")

        # Retornar el resultado del método seleccionado
        method = self.inv_method_combo.currentText()
        return result_geo if method == "Geométrica" else result_ana

    def start_inverse_simulation(self):
        """Simula el movimiento usando los ángulos calculados por cinemática inversa"""
        result = self.compute_inverse()
        if result is None:
            self.inv_status_label.setText("❌ Sin solución (fuera de alcance)")
            self.inv_status_label.setStyleSheet("color: #e74c3c; font-style: italic; font-size: 9pt;")
            return
        self.canvas.target_q = list(result)
        self.send_to_arduino = False
        self.step_count = 0
        self.inv_status_label.setText("⏳ Simulando inversa...")
        self.inv_status_label.setStyleSheet("color: #8e44ad; font-style: italic; font-size: 9pt;")
        # Usar D-H para la visualización de la animación inversa
        self.method_combo.setCurrentText("Denavit-Hartenberg")
        self.anim_timer.start(25)

    def start_inverse_motion(self):
        """Ejecuta el movimiento físico usando los ángulos calculados por cinemática inversa"""
        result = self.compute_inverse()
        if result is None:
            self.inv_status_label.setText("❌ Sin solución (fuera de alcance)")
            self.inv_status_label.setStyleSheet("color: #e74c3c; font-style: italic; font-size: 9pt;")
            return
        self.canvas.target_q = list(result)
        self.send_to_arduino = True
        self.step_count = 0
        self.inv_status_label.setText("⚙️ Ejecutando inversa en físico...")
        self.inv_status_label.setStyleSheet("color: #27ae60; font-style: italic; font-size: 9pt;")
        self.method_combo.setCurrentText("Denavit-Hartenberg")
        self.anim_timer.start(25)
        if self.arduino:
            cmd = " ".join([str(v) for v in self.canvas.target_q]) + "\n"
            self.arduino.write(cmd.encode())
            self.arduino.flush()
        else:
            self.inv_status_label.setText("⚠️ Sin Arduino conectado")
            self.inv_status_label.setStyleSheet("color: #e67e22; font-style: italic; font-size: 9pt;")

    # ============================================================
    # MÉTODOS EXISTENTES DE CINEMÁTICA DIRECTA
    # ============================================================

    def toggle_serial(self):
        if self.arduino is None:
            try:
                self.arduino = serial.Serial(self.combo.currentText(), 115200, timeout=0.1)
                self.arduino.setDTR(False)
                self.arduino.setRTS(False)
                self.btn_conn.setText("DESCONECTAR")
                self.btn_conn.setStyleSheet("background: #c0392b; color: white;")
            except:
                pass
        else:
            self.arduino.close()
            self.arduino = None
            self.btn_conn.setText("CONECTAR")
            self.btn_conn.setStyleSheet("")

    def sync_serial_ui(self):
        if self.arduino is not None:
            self.btn_conn.setText("DESCONECTAR")
            self.btn_conn.setStyleSheet("background: #c0392b; color: white;")
        else:
            self.btn_conn.setText("CONECTAR")
            self.btn_conn.setStyleSheet("")

    def get_validated_inputs(self):
        """Procesa, valida y autocorrige los límites de los ángulos q1-q4"""
        vals = []
        for i, le in enumerate(self.inputs):
            try:
                val = float(le.text())
            except ValueError:
                val = 0.0
            
            low, high = self.ranges[i]
            if val < low:
                val = low
                le.setText(str(low))
            elif val > high:
                val = high
                le.setText(str(high))
            vals.append(val)
        return vals

    def start_simulation(self):
        try:
            self.canvas.target_q = self.get_validated_inputs()
            self.send_to_arduino = False
            self.step_count = 0
            self.status_label.setText("⏳ Simulando...")
            self.status_label.setStyleSheet("color: #2980b9; font-style: italic; font-size: 9pt;")
            self.anim_timer.start(25)
        except:
            pass

    def start_motion(self):
        try:
            self.canvas.target_q = self.get_validated_inputs()
            self.send_to_arduino = True
            self.step_count = 0
            self.status_label.setText("⚙️ Ejecutando en físico...")
            self.status_label.setStyleSheet("color: #27ae60; font-style: italic; font-size: 9pt;")
            self.anim_timer.start(25)
            if self.arduino:
                cmd = " ".join([str(v) for v in self.canvas.target_q]) + "\n"
                self.arduino.write(cmd.encode())
                self.arduino.flush()
            else:
                self.status_label.setText("⚠️ Sin Arduino conectado")
                self.status_label.setStyleSheet("color: #e67e22; font-style: italic; font-size: 9pt;")
        except:
            pass

    def animate_step(self):
        self.step_count += 1
        total_steps = 40
        t = self.step_count / total_steps
        t = t * t * (3 - 2 * t)
        
        interp_q = [c + (t_obj - c) * t for c, t_obj in zip(self.canvas.curr_q, self.canvas.target_q)]
        
        method = self.method_combo.currentText()
        n, o, a, p = self.canvas.update_view(*interp_q, method)
        
        # Se envía el método para que solo la matriz seleccionada se actualice y la otra quede estática
        self.update_noap_table(n, o, a, p, method)
        
        if self.step_count >= total_steps:
            self.anim_timer.stop()
            self.canvas.curr_q = list(self.canvas.target_q)
            if self.send_to_arduino:
                self.status_label.setText("✅ Ejecutado en físico")
                self.inv_status_label.setText("✅ Ejecutado en físico")
            else:
                self.status_label.setText("✅ Simulación completada")
                self.inv_status_label.setText("✅ Simulación completada")
            self.status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 9pt;")
            self.inv_status_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 9pt;")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Multimetodo 4 GDL ROBOTICA I")
        self.setGeometry(100, 100, 1300, 850)

        self.validation_widget = ValidationWidget()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.validation_widget)
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())