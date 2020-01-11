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
scale = 1.0
step = "step_11_smooth"
mesh = "sb10.obj"
loss_epsilon = 0.05
learning_rate = 5e-5
first_step = 0

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
    lights.append(pyredner.generate_quad_light(position = pos + torch.tensor([0.0,0.0,-13.0]), \
                                     look_at = camera0.look_at, \
                                     size = torch.tensor([2.0, 2.0]), \
                                     intensity = torch.tensor([light_intensity, light_intensity, light_intensity])))
print("LIGHT ONE DONE")
for ind, pos in enumerate(camLocs):
    pos = torch.tensor([100, 0.0, -3.0]) 
    normal = pos.div(torch.norm(pos - torch.tensor([-3.0, 0.0, -3.0]) ))                                                   
    pos = normal * radius2   
    
    lights.append(pyredner.generate_quad_light(position = pos + torch.tensor([8.0, 0.0, -3.0]), \
                                     look_at = camera0.look_at, \
                                     size = torch.tensor([2.0, 2.0]), \
                                     intensity = torch.tensor([light_intensity, light_intensity, light_intensity])))
    
print("LIGHT TWO DONE")    

cube = pyredner.load_obj('resources/shadow_cube.obj', return_objects=True)
start = pyredner.load_obj('resources/' + mesh, return_objects=True)
target_obj1 = pyredner.load_obj('resources/monkey.obj', return_objects=True)

cube[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]))
start[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]))
target_obj1[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]))

cube[0].normals = pyredner.compute_vertex_normal(cube[0].vertices, cube[0].indices)
start[0].normals = pyredner.compute_vertex_normal(start[0].vertices, start[0].indices)
target_obj1[0].normals = pyredner.compute_vertex_normal(target_obj1[0].vertices, target_obj1[0].indices)

# Setup camera
cam = pyredner.Camera(position = torch.tensor([0.0, 0.0, -2.5]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      camera_type = redner.CameraType.orthographic,
                      clip_near = 1e-1, # needs to > 0
                      resolution = (64, 64),
                      fisheye = False)

cam2 = pyredner.Camera(position = torch.tensor([0.0, 0.0, -5.0]),
                      look_at = torch.tensor([0.0, 0.0, 0.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      camera_type = redner.CameraType.perspective,
                      fov = torch.tensor([45.0]),
                      clip_near = 1e-2, # needs to > 0
                      resolution = (512, 512),
                      fisheye = False)

cam3 = pyredner.Camera(position =  torch.tensor( [-0.5, 0.0, -3.0]) ,
                      look_at = torch.tensor([-10.0, 0.0, -3.0]),
                      up = torch.tensor([0.0, 1.0, 0.0]),
                      camera_type = redner.CameraType.orthographic,
                      clip_near = 1e-1, # needs to > 0
                      resolution = (64, 64),
                      fisheye = False)

# SET UP FIRST TARGET AND TWO LIGHTS
shapes = []
shapes.append(target_obj1[0])
shapes.append(cube[0])

numShapes = len(shapes)
print("NUM SHAPES", numShapes)
shapes.extend(lights)

area_lights = []
for i in range(numShapes, len(shapes)):
    area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity, light_intensity, light_intensity])))
    area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity*10, light_intensity*10, light_intensity*10])))

# FOR SCENE DEBUGGING
#envmap = pyredner.imread('sunsky.exr')
#if pyredner.get_use_gpu():
#    envmap = envmap.cuda()
#envmap = pyredner.EnvironmentMap(envmap)
envmap = None


# SET UP SCENES
maker_translation_params0 = torch.tensor([0.0, 0.0, -3.1],
    device = pyredner.get_device())
translation = torch.tensor([0.0, 0.0, 0.01],
    device = pyredner.get_device())

shape0_vertices = shapes[0].vertices.clone()
shape1_vertices = shapes[1].vertices.clone() 

shapes[0].vertices = \
    (shape0_vertices)  + maker_translation_params0

shapes[1].vertices = \
    (shape1_vertices) * 5 + translation

scene = pyredner.Scene(cam, objects = [shapes[0], shapes[1], shapes[2]],area_lights = [area_lights[0]], envmap = None)
scene_intense = pyredner.Scene(cam, objects = [shapes[0], shapes[1], shapes[2]], area_lights = [area_lights[1]], envmap = None)

target = pyredner.render_pathtracing(scene = [scene], num_samples=(512, 0), max_bounces=1)[0]
pyredner.imwrite(target.cpu(), 'results/shadow_art/target.png')

if pyredner.get_use_gpu():
    target = target.cuda()
# SECOND LIGHT INTENSITY

target_intense = pyredner.render_pathtracing(scene = [scene_intense], num_samples=(512, 0), max_bounces=1)[0]
pyredner.imwrite(target_intense.cpu(), 'results/shadow_art/target_intense.png')
if pyredner.get_use_gpu():
    target_intense = target_intense.cuda()

target_obj2 = pyredner.load_obj('resources/bunny_big.obj', return_objects=True)

target_obj2[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]))
target_obj2[0].normals = pyredner.compute_vertex_normal(target_obj2[0].vertices, target_obj2[0].indices)

shapes = []
shapes.append(target_obj2[0])
shapes.append(cube[0])
numShapes = len(shapes)
shapes.extend(lights)

for i in range(numShapes, len(shapes)):
    area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity*1.3, light_intensity*1.3, light_intensity*1.3])))
    area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity*10, light_intensity*10, light_intensity*10])))

shape0_vertices = shapes[0].vertices.clone()
shapes[0].vertices = \
    (shape0_vertices)  + maker_translation_params0

translation2 = torch.tensor([-3.2, 0.0, -3.0],
    device = pyredner.get_device())
shapes[1].vertices = \
    (shape1_vertices) * 5 + translation2

scene_3 = pyredner.Scene(cam3, objects=[shapes[0], shapes[1], shapes[3]], area_lights = [area_lights[2]], envmap = None)
scene_intense_2 = pyredner.Scene(cam3, objects=[shapes[0], shapes[1], shapes[3]], area_lights = [area_lights[3]], envmap = None)

target2 = pyredner.render_pathtracing(scene = [scene_3], num_samples=(512, 0), max_bounces=1)[0]
pyredner.imwrite(target2.cpu(), 'results/shadow_art/target_2.png')
if pyredner.get_use_gpu():
    target2 = target2.cuda()

target2_intense = pyredner.render_pathtracing(scene = [scene_intense_2], num_samples=(512, 0), max_bounces=1)[0]
pyredner.imwrite(target2_intense.cpu(), 'results/shadow_art/target_2_intense.png')

if pyredner.get_use_gpu():
    target2_intense = target2_intense.cuda()

# SET UP INITAL GUESS

cube = pyredner.load_obj('resources/shadow_cube.obj', return_objects=True)

cube[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]))

cube[0].normals = pyredner.compute_vertex_normal(cube[0].vertices, cube[0].indices)

shapes = []
shapes.append(start[0])
shapes.append(cube[0])
maker_translation_params0 = torch.tensor([0.0, 0.0, -3.0],
    device = pyredner.get_device())
numShapes = len(shapes)
print("INIT SCENE", numShapes, light_intensity)
shapes.extend(lights)

area_lights = []
for i in range(numShapes, len(shapes)):
    area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity, light_intensity, light_intensity])))
    area_lights.append(pyredner.AreaLight(shape_id = numShapes, intensity = torch.tensor([light_intensity*10, light_intensity*10, light_intensity*10])))

shape0_vertices = shapes[0].vertices.clone() * scale
shape1_vertices = shapes[1].vertices.clone()

shapes[0].vertices = \
    (shape0_vertices) + maker_translation_params0

shapes[1].vertices = \
    (shape1_vertices) * 5 + translation

scene_v1 = pyredner.Scene(cam, objects = [shapes[0], shapes[1], shapes[2]],area_lights = [area_lights[0]], envmap = None)

init = pyredner.render_pathtracing(scene = [scene_v1], num_samples=(512, 0), max_bounces=1)[0]
pyredner.imwrite(init.cpu(), 'results/shadow_art/init.png')

shapes[1].vertices = \
    (shape1_vertices) * 5 + translation2
scene_v2 = pyredner.Scene(cam3, objects = [shapes[0], shapes[1], shapes[3]], area_lights = [area_lights[2]], envmap = None)

init2 = pyredner.render_pathtracing(scene = [scene_v2], num_samples=(512, 0), max_bounces=1)[0]
pyredner.imwrite(init2.cpu(), 'results/shadow_art/init_2.png')

pyredner.imwrite(torch.abs(target-init).cpu(), 'results/shadow_art/diff1.png')
pyredner.imwrite(torch.abs(target2-init2).cpu(), 'results/shadow_art/diff2.png')

# OPTIMIZATION TIME
best_loss = 1000000
strikes = 0
delta_strategy = 0

vertex_deltas = torch.zeros(shapes[0].vertices.shape, device=pyredner.get_device(), requires_grad=True)
optimizer = torch.optim.Adam([vertex_deltas], lr=learning_rate)
pic_num = 0
for target_switch in range(2):
    which_target = target_switch % 2
    print("which target")
    for t in range(1500):
        pic_num += 1
        print('iteration:', pic_num)
        optimizer.zero_grad()

        with torch.no_grad():
            shapes[0].normals = pyredner.compute_vertex_normal(shapes[0].vertices, shapes[0].indices)

        for v in range(shapes[0].vertices.shape[0]):
            if first_step == 1 or delta_strategy == 0:
                with torch.no_grad():
                    if(first_step):
                        phi = torch.atan2(shapes[0].vertices[v][1], shapes[0].vertices[v][0])
                        check_for_safety = ( shapes[0].vertices[v][2] + 3.0) / torch.norm(shapes[0].vertices[v] - maker_translation_params0) 
                        if check_for_safety < -1.0: 
                            check_for_safety = torch.tensor(-1.0, device=pyredner.get_device())
                        if check_for_safety > 1.0:
                            check_for_safety = torch.tensor(1.0, device=pyredner.get_device())
                        theta = torch.acos(check_for_safety) 
                    else:
                        phi = torch.atan2(shapes[0].normals[v][1], shapes[0].normals[v][0]) 
                        check_for_safety = shapes[0].normals[v][2]
                        if check_for_safety < -1.0: 
                            check_for_safety = torch.tensor(-1.0, device=pyredner.get_device())
                        if check_for_safety > 1.0:
                            check_for_safety = torch.tensor(1.0, device=pyredner.get_device())
                        theta = torch.acos(check_for_safety) 
                
                polar = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta)*torch.sin(phi),torch.cos(theta)])
                
                shapes[0].vertices[v] = shapes[0].vertices[v] + vertex_deltas[v][2] * polar
                
            else:
                shapes[0].vertices[v] = shapes[0].vertices[v] + vertex_deltas[v]

        loss = 0
        if(True or  which_target == 1 or first_step == 1):
            with torch.no_grad():    
                shapes[1].vertices = \
                    (shape1_vertices) * 5 + translation
            scene_optim = pyredner.Scene(cam, objects = [shapes[0], shapes[1], shapes[2]],area_lights = [area_lights[0]], envmap = None)
            img = pyredner.render_pathtracing(scene = [scene_optim], num_samples=(64, 64), max_bounces=1)[0]
            pyredner.imwrite(img.cpu(), 'results/shadow_art/multitarget2/' + step + '/iter_{}_t0.png'.format(pic_num))
            
            loss += (img - target).pow(2).sum() * 0.5 
        
        if(True or which_target == 0 or first_step == 1):
            
            with torch.no_grad(): 
                shapes[1].vertices = \
                    (shape1_vertices) * 5 + translation2
            scene_optim_v2 = pyredner.Scene(cam3, objects=[shapes[0], shapes[1], shapes[3]], area_lights = [area_lights[2]], envmap = None)
            img2 = pyredner.render_pathtracing(scene = [scene_optim_v2], num_samples=(64, 64), max_bounces=1)[0]

            pyredner.imwrite(img2.cpu(), 'results/shadow_art/multitarget2/' + step + '/iter_{}_t1.png'.format(pic_num))
            loss += (img2-target2).pow(2).sum() * 0.5  
            
        print('loss:', loss.item(), " with strategy " , delta_strategy)
        write_obj("results/shadow_art/running-output.obj", shapes)
        write_obj("results/shadow_art/multitarget2/" + step + "/iter_{}.obj".format(pic_num), shapes)
        # Backpropagate the gradients.
        loss.backward(retain_graph=True)
    
        
        # Take a gradient descent step.
        optimizer.step()
        optimizer.zero_grad()
        if(loss - loss_epsilon < best_loss):
            if(loss < best_loss):
                best_loss = loss
            strikes = 0
        else:
            strikes += 1
        if first_step == 0 and strikes == 3:
            strikes = 0
            if delta_strategy == 1:
                smooth("results/shadow_art/running-output.obj", "results/shadow_art/running-output.obj")
                material_map1, mesh_list1, light_map1 = pyredner.load_obj('results/shadow_art/running-output.obj')
                output = pyredner.load_obj('results/shadow_art/running-output.obj', return_objects=True)
                output[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]))

                output[0].normals = pyredner.compute_vertex_normal(cube[0].vertices, cube[0].indices)
                shapes[0] = output[0]
            
            delta_strategy = (delta_strategy + 1) % 2
            vertex_deltas = torch.zeros(shapes[0].vertices.shape, device=pyredner.get_device(), requires_grad=True)
            optimizer = torch.optim.Adam([vertex_deltas], lr=learning_rate) 
    smooth("results/shadow_art/running-output.obj", "results/shadow_art/running-output.obj")
    material_map1, mesh_list1, light_map1 = pyredner.load_obj('results/shadow_art/running-output.obj')
    for mtl_name, mesh in mesh_list1:
        shapes[0] = (pyredner.Shape(\
            vertices = mesh.vertices,
            indices = mesh.indices,
            material_id = 0,
            uvs = mesh.uvs,
            normals = mesh.normals,
            uv_indices = mesh.uv_indices))
            
    vertex_deltas = torch.zeros(shapes[0].vertices.shape, device=pyredner.get_device(), requires_grad=True)
    optimizer = torch.optim.Adam([vertex_deltas], lr=learning_rate) 
    best_loss = 100000
    strikes = 0
    delta_strategy = 0

# Render the final result.
scene_args = pyredner.RenderFunction.serialize_scene(\
    scene = scene,
    num_samples = 512,
    max_bounces = 1)
img = render(202, *scene_args)
# Save the images and differences.
pyredner.imwrite(img.cpu(), 'results/shadow_art/final.exr')
pyredner.imwrite(img.cpu(), 'results/shadow_art/final.png')
pyredner.imwrite(torch.abs(target - img).cpu(), 'results/shadow_art/final_diff.png')

# Convert the intermediate renderings to a video.
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/shadow_art/iter_%d.png", "-vb", "20M",
    "results/shadow_art/out.mp4"])
from subprocess import call
call(["ffmpeg", "-framerate", "24", "-i",
    "results/shadow_art/iter_pan%d.png", "-vb", "20M",
    "results/shadow_art/out_pan.mp4"])
