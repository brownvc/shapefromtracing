import pyredner
import torch
import math
import redner
import os

import torch.nn.functional as F

# Use GPU if available
pyredner.set_use_gpu(torch.cuda.is_available())

radius = 1.5*math.sqrt(2)
radius2 = 0.8*math.sqrt(2)
light_intensity = 100.0
path = "results/shadow_art/"
camera_file = "cameras.xml"
step = "step_6_smooth"
num_iters = 51

def resample(inputmesh, outputmesh):
    os.system("/home/andy/Desktop/shape-from-tracing-old/mesh_resample/build/resample " + inputmesh + " " + path + camera_file + " " + outputmesh)

def poisson(inputmesh, outputmesh):
    os.system("meshlabserver -i " + inputmesh + " -o " + outputmesh + " -s poisson.mlx")

def smooth(inputmesh, outputmesh):
    os.system("meshlabserver -i " + inputmesh + " -o " + outputmesh + " -s smooth_a_little.mlx")

def normals_and_poisson(inputmesh, outputmesh):
    os.system("meshlabserver -i " + inputmesh + " -o " + outputmesh + " -s normals-and-poisson.mlx")

def write_obj(filename, mesh):
    output = open(filename, "w")
    seen_vertices = 0
    mesh = mesh[0]
    for i, v in enumerate(mesh.vertices):
        output.write("v " + str(v[0].item()) + " " + str(v[1].item()) + " " + str(v[2].item()) + '\n')
    for i, f in enumerate(mesh.indices):
        output.write("f " + str(f[0].item() + 1 + seen_vertices) + " " + str(f[1].item() + 1 + seen_vertices) + " " + str(f[2].item() + 1 + seen_vertices) + '\n')
    seen_vertices += len(mesh.vertices)
    output.close()

def fibonacci_sphere(samples=1,randomize=True):
    rnd = 1.
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return points

lights = []
num_cameras = 1
fov = 45.0
resolution = 1
look_at = torch.tensor([0.0, 0.0, 0.0])
camLocs = fibonacci_sphere(num_cameras, False)


cams = []
target_objects = pyredner.load_obj('resources/monkey.obj', return_objects=True)
camera0 = pyredner.automatic_camera_placement(target_objects, resolution)
for ind, pos in enumerate(camLocs):
    pos = torch.tensor([0.5, 0.0, 100.0])
    pos = torch.tensor(pos)
    normal = pos.div(torch.norm(pos))                                                     
    pos = normal * radius2    
    lights.append(pyredner.generate_quad_light(position = pos + torch.tensor([0.0,0.0,-15.0]), \
                                     look_at = camera0.look_at, \
                                     size = torch.tensor([2.0, 2.0]), \
                                     intensity = torch.tensor([light_intensity, light_intensity, light_intensity])))
print("LIGHT ONE DONE")
for ind, pos in enumerate(camLocs):
    pos = torch.tensor([100, 0.0, -3.0]) 
    normal = pos.div(torch.norm(pos - torch.tensor([-3.0, 0.0, -3.0]) ))                                                   
    pos = normal * radius2   
    
    lights.append(pyredner.generate_quad_light(position = pos + torch.tensor([10.0, 0.0, -3.0]), \
                                     look_at = camera0.look_at, \
                                     size = torch.tensor([2.0, 2.0]), \
                                     intensity = torch.tensor([light_intensity, light_intensity, light_intensity])))
    
print("LIGHT TWO DONE")    

cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.5]),
                      look_at = torch.tensor([0.0, 0.0, -3.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      camera_type = redner.CameraType.perspective,
                      fov = torch.tensor([45.0]),
                      clip_near = 1e-2, # needs to > 0
                      resolution = (512, 512),
                      fisheye = False)

cam3 = pyredner.Camera(position =  torch.tensor( [2.5, 0.0, -3.0]) ,
                      look_at = torch.tensor([0.0, 0.0, -3.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      camera_type = redner.CameraType.perspective,
                      fov = torch.tensor([45.0]),
                      clip_near = 1e-2, # needs to > 0
                      resolution = (512, 512),
                      fisheye = False)

for obj in range(1, num_iters):
    target_obj1 = pyredner.load_obj('results/shadow_art/multitarget/' + step + '/iter_' + str(obj) + '.obj', return_objects=True)

    target_obj1[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]), two_sided=True)

    target_obj1[0].normals = pyredner.compute_vertex_normal(target_obj1[0].vertices, target_obj1[0].indices)

    shapes = []
    shapes.append(target_obj1[0])

    numShapes = len(shapes)
    shapes.extend(lights)

    area_lights = []
    for i in range(numShapes, len(shapes)):
        area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity, light_intensity, light_intensity])))
        area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity*10, light_intensity*10, light_intensity*10])))

    scene = pyredner.Scene(cam, objects = [shapes[0], shapes[1]],area_lights = [area_lights[0]], envmap = None)
    scene_intense = pyredner.Scene(cam, objects = [shapes[0], shapes[1]], area_lights = [area_lights[1]], envmap = None)

    target = pyredner.render_pathtracing(scene = [scene], num_samples=(512, 0), max_bounces=1)[0]
    pyredner.imwrite(target.cpu(), 'results/shadow_art/high_res/' + step + '/' + str(obj) + '_0.png')

    area_lights = []
    for i in range(numShapes, len(shapes)):
        area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity, light_intensity, light_intensity])))
        area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity*10, light_intensity*10, light_intensity*10])))

    shape0_vertices = shapes[0].vertices.clone()
    shapes[0].vertices = \
        (shape0_vertices)

    scene_3 = pyredner.Scene(cam3, objects=[shapes[0], shapes[2]], area_lights = [area_lights[2]], envmap = None)

    target2 = pyredner.render_pathtracing(scene = [scene_3], num_samples=(512, 0), max_bounces=1)[0]
    pyredner.imwrite(target2.cpu(), 'results/shadow_art/high_res/' + step + '/' + str(obj) + '_1.png')
