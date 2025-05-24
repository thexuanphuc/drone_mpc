class Data_Extraction():

    def data(self, cat_states, cat_controls):
        depth = cat_states.shape[0]
        for i in range(depth):
            if i == 0:
                x = cat_states[i, 0, :]
            if i == 1:
                y = cat_states[i, 0, :]
            if i == 2:
                z = cat_states[i, 0, :]
            if i == 3:
                u = cat_states[i, 0, :]
            if i == 4:
                v = cat_states[i, 0, :]
            if i == 5:
                w = cat_states[i, 0, :]
            if i == 6:
                phi = cat_states[i, 0, :]
            if i == 7:
                theta = cat_states[i, 0, :]
            if i == 8:
                psi = cat_states[i, 0, :]
            if i == 9:
                p = cat_states[i, 0, :]
            if i == 10:
                q = cat_states[i, 0, :]
            if i == 11:
                r = cat_states[i, 0, :]
            
        u1 = []
        u2 = []
        u3 = []
        u4 = []
        for i in range(0, cat_controls.shape[0], 4):
            u1.append(cat_controls[i+0,0])
            u2.append(cat_controls[i+1,0])
            u3.append(cat_controls[i+2,0])
            u4.append(cat_controls[i+3,0])

        return x, y, z, u, v, w, phi, theta, psi, p, q, r, u1, u2, u3, u4
    