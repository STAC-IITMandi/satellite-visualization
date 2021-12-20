
import math

pi = math.pi
cos = lambda x : math.cos(x * pi/180)
sin = lambda x : math.sin(x * pi/180)

circles = []
vertices = []
vert_norm_tex = []
faces = []

# Generate coordinates
factor = 5          # 1,3,5,9,15
circles.append(([(0.,-1.,0.,0.,1.)], 0))
for theta in range(-87, 90, factor):
    if abs(theta) >= 82:
        step = 8 * factor
    elif abs(theta) >= 75:
        step = 4 * factor
    elif abs(theta) >= 60:
        step = 2 * factor
    else :
        step = 1 * factor
    A = range(0,360,step)
    # In a sphere, normal vector == position vector
    # x,y,z,u,v
    circ = [(cos(theta)*cos(a), sin(theta), cos(theta)*sin(a),
             1.0 - a/360., 0.5 - theta/180.) for a in A]
    circ.append((cos(theta), sin(theta), 0.0, 0.0, 0.5-theta/180.))
    
    # circles.append((circ, len(circ)))
    circles.append((circ, circles[-1][1]+len(circles[-1][0])))

circles.append(([(0.,1.,0.,0.,0.)], circles[-1][1]+len(circles[-1][0])))

# Create list of faces
for theta, (points, cc) in enumerate(circles):
    # print(theta*factor-90, cc)
    i = theta * factor - 90
    if theta == 0:
        l = len(circles[theta+1][0])
        for p in range(l):
            faces.extend([0, cc+p + 1, (cc+p+1)%l + 1])
    elif theta == len(circles)-2:
        l = len(points)+1
        for p in range(l):
            faces.extend([cc+p, cc+ (p+1)%l, cc+len(points)])
    elif theta == len(circles)-1:
        pass
    elif len(circles[theta][0]) == len(circles[theta+1][0]) :
        l = len(circles[theta][0])
        for p in range(l):
            faces.extend([cc+p, cc + (p+1)%l, cc+p+l])
            faces.extend([cc+p+l, cc+l + (p+1)%l, cc + (p+1)%l])
    elif len(circles[theta][0]) < len(circles[theta+1][0]) :
        l = len(circles[theta][0])-1
        for p in range(l):
            faces.extend([cc+p, cc+l+2*p+1, cc+l+2*p+2])
            faces.extend([cc+l+2*p+2, cc+l + (2*p+3), cc + (p+1)])
            faces.extend([cc+p, cc+l+2*p+2, cc + (p+1)])
    elif len(circles[theta][0]) > len(circles[theta+1][0]) :
        l = len(circles[theta+1][0])-1
        for p in range(l):
            faces.extend([cc+2*p, cc+2*l+p+1, cc+2*p+1])
            faces.extend([cc+2*p+1, cc + (2*p+2), cc+2*l + (p+2)])
            faces.extend([cc+2*l+p+1, cc+2*p+1, cc+2*l + (p+2)])
    else :
        assert False

    for pos in points :
        vertices.extend(pos[:3])
        vert_norm_tex.extend(pos[:3])
        vert_norm_tex.extend(pos)

spherical_mesh = {'v':vertices, 'f':faces, 'format':
    [(b'v_pos', 3, 'float')]}
spherical_mesh_tex = {'v':vert_norm_tex, 'f':faces, 'format':
    [(b'v_pos', 3, 'float'),(b'v_norm', 3, 'float'),(b'v_texc', 2, 'float')]}

c = 0.02
cube_mesh = {'v':
        [-c,-c,-c, -1.,0.,0.,   -c,-c,c, -1.,0.,0., -c,c,c, -1.,0.,0.,  -c,c,-c, -1.,0.,0., 
         c,-c,-c, 1.,0.,0.,     c,-c,c, 1.,0.,0.,   c,c,c, 1.,0.,0.,    c,c,-c, 1.,0.,0., 
         -c,-c,-c, 0.,-1.,0.,   c,-c,-c, 0.,-1.,0., c,-c,c, 0.,-1.,0.,  -c,-c,c, 0.,-1.,0., 
         -c,c,c, 0.,1.,0.,      c,c,c, 0.,1.,0.,    c,c,-c, 0.,1.,0.,   -c,c,-c, 0.,1.,0., 
         -c,-c,-c, 0.,0.,-1.,   c,-c,-c, 0.,0.,-1., c,c,-c, 0.,0.,-1.,  -c,c,-c, 0.,0.,-1., 
         c,-c,c, 0.,0.,1.,      -c,-c,c, 0.,0.,1.,  -c,c,c, 0.,0.,1.,   c,c,c, 0.,0.,1.,
        ],
        'f':[0,1,2, 0,2,3, 4,5,6, 4,6,7, 8,9,10, 8,10,11, 
             12,13,14, 12,14,15, 16,17,18, 16,18,19, 20,21,22, 20,22,23],
        'format':[(b'v_pos', 3, 'float'),(b'v_norm', 3, 'float')]}

