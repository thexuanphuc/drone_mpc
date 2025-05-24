import numpy as np
from define_variables import Variables_Defination

class Gates(Variables_Defination):
    
    def __init__(self):
        super().gate_variables()

    def gate_points_for_visualisation(self):
        self.gate_x_points = []
        self.gate_y_points = []
        self.gate_z_points = []
        self.gate_inner_x_points = []
        self.gate_inner_y_points = []
        self.gate_inner_z_points = []
        for gate in self.gates:    
            angle = np.deg2rad(gate[-1])
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle),  0],
                                        [np.sin(angle),  np.cos(angle),  0],
                                        [            0,              0,   1]])
            outer_width = 1.5
            inner_width = 1.2
            diff = outer_width - inner_width
            outer_height = 1.5
            inner_height = 1.2
            BL = gate[0:3]
            BR = np.array([BL[0], BL[1]-outer_width, BL[2]])
            TR = np.array([BL[0], BL[1]-outer_width, BL[2]+outer_height])
            TL = np.array([BL[0], BL[1], BL[2]+outer_height])
            BL_inner = np.array([BL[0], BL[1]-diff, BL[2]+diff])
            BR_inner = np.array([BR[0], BR[1]+diff, BR[2]+diff])
            TR_inner = np.array([TR[0], TR[1]+diff, TR[2]-diff])
            TL_inner = np.array([TL[0], TL[1]-diff, TL[2]-diff])
            BL =  np.dot(rotation_matrix, BL - BL) + BL
            BR =  np.dot(rotation_matrix, BR - BL) + BL
            TR =  np.dot(rotation_matrix, TR - BL) + BL
            TL =  np.dot(rotation_matrix, TL - BL) + BL
            gate_x = np.array([BL[0], BR[0], TR[0], TL[0], BL[0]])
            gate_y = np.array([BL[1], BR[1], TR[1], TL[1], BL[1]])
            gate_z = np.array([BL[2], BR[2], TR[2], TL[2], BL[2]])
            self.gate_x_points.append(gate_x)
            self.gate_y_points.append(gate_y)
            self.gate_z_points.append(gate_z)
            BL_inner =  np.dot(rotation_matrix, BL_inner - BL) + BL
            BR_inner =  np.dot(rotation_matrix, BR_inner - BL) + BL
            TR_inner =  np.dot(rotation_matrix, TR_inner - BL) + BL
            TL_inner =  np.dot(rotation_matrix, TL_inner - BL) + BL
            gate_inner_x = np.array([BL_inner[0], BR_inner[0], TR_inner[0], TL_inner[0], BL_inner[0]])
            gate_inner_y = np.array([BL_inner[1], BR_inner[1], TR_inner[1], TL_inner[1], BL_inner[1]])
            gate_inner_z = np.array([BL_inner[2], BR_inner[2], TR_inner[2], TL_inner[2], BL_inner[2]])
            self.gate_inner_x_points.append(gate_inner_x)
            self.gate_inner_y_points.append(gate_inner_y)
            self.gate_inner_z_points.append(gate_inner_z)