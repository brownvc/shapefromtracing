import torch
import pyredner
import math
import os
import sys

pyredner.render_pytorch.print_timing = False

def simplify(inputmesh, outputmesh, target):
  simp_str = """<!DOCTYPE FilterScript>
                  <FilterScript>
                  <filter name="Quadric Edge Collapse Decimation">
                  <Param type="RichInt" value=""" + '\"' + str(target) + '\"' + """ name="TargetFaceNum"/>
                  <Param type="RichFloat" value="0" name="TargetPerc"/>
                  <Param type="RichFloat" value="0.3" name="QualityThr"/>
                  <Param type="RichBool" value="false" name="PreserveBoundary"/>
                  <Param type="RichFloat" value="1" name="BoundaryWeight"/>
                  <Param type="RichBool" value="false" name="PreserveNormal"/>
                  <Param type="RichBool" value="true" name="PreserveTopology"/>
                  <Param type="RichBool" value="true" name="OptimalPlacement"/>
                  <Param type="RichBool" value="false" name="PlanarQuadric"/>
                  <Param type="RichBool" value="false" name="QualityWeight"/>
                  <Param type="RichBool" value="true" name="AutoClean"/>
                  <Param type="RichBool" value="false" name="Selected"/>
                  </filter>
                  </FilterScript>"""

  simp_script = open("simplify.mlx", "w")
  simp_script.write(simp_str)
  simp_script.close()
  os.system("meshlabserver -i " + inputmesh + " -o " + outputmesh + " -s meshlab/simplify.mlx")

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

        points.append(torch.tensor([x,y,z]))

    return points

def generate_scenes(camLocs,objects,envmap=None):
  scenes = []
  up = torch.tensor([0.0, 1.0, 0.0])
  offset_factor = 2.0
  light_intensity = 500.0            

  for ind, loc in enumerate(camLocs):
    camera = pyredner.Camera(position = camera0.look_at + radius * loc,
                          look_at = camera0.look_at,
                          up = camera0.up,
                          fov = camera0.fov,
                          resolution = camera0.resolution)
    
    normal = camera.position.div(torch.norm(camera.position))
    tangent = torch.cross(normal, up)
    tangent = tangent.div(torch.norm(tangent))
    bitangent = torch.cross(normal, tangent)
    bitangent = bitangent.div(torch.norm(bitangent))

    lightPos = camera.position + offset_factor * tangent
    light = pyredner.generate_quad_light(position = lightPos,
                                     look_at = camera0.look_at,
                                     size = torch.tensor([0.1, 0.1]),
                                     intensity = torch.tensor([light_intensity, light_intensity, light_intensity]))
    
    # Camera data for voxel carving
    #print(str(ind) + " " + str(camera.position.data[0].item()) + " " + str(camera.position.data[1].item()) + " " + str(camera.position.data[2].item()) + " " + str(camera.look_at.data[0].item()) + " " + str(camera.look_at.data[1].item()) + " " + str(camera.look_at.data[2].item()))

    scenes.append(pyredner.Scene(camera = camera, objects = [objects[0], light], envmap=envmap))
  return scenes

def getGaussianFilter(kernel_size = 3, sigma = 3, channels = 3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()    
    
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp( -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).cuda()
    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, groups=channels, kernel_size=kernel_size, bias=False, stride=2, padding=[int((kernel_size - 1)/2), int((kernel_size - 1)/2)])    # with torch.no_grad():

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = True
    return gaussian_filter
gaussian_func = getGaussianFilter()

# Parse cmdline
path = sys.argv[1]
path = path + "/"
target_obj_file = sys.argv[2]
init_obj_file = sys.argv[3]
face_target = int(sys.argv[4])

# Load Target model
target_objects = pyredner.load_obj(target_obj_file, return_objects=True)

print(target_objects[0].vertices.shape)

resolution = (256, 256)
num_cameras = 32
radius = 1.4

camera0 = pyredner.automatic_camera_placement(target_objects, resolution)
camLocs = fibonacci_sphere(num_cameras, False)

target_scenes = generate_scenes(camLocs, target_objects, envmap=None)

# Binary mask
#target_objects[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]))
#mask_scenes = generate_scenes(camLocs, target_objects, envmap)

# Binary mask
#masks = pyredner.render_albedo(scene = mask_scenes, num_samples = (128, 0))

# Render Targets
targets = pyredner.render_pathtracing(scene = target_scenes, num_samples = (128, 0), max_bounces=1)

for ind, img in enumerate(targets):
  img = img.data.cpu()
  pyredner.imwrite(img[:,:,0], path + "targets/target_" + str(ind) + ".png")



num_gaussian_levels = 0
def loss_function(renders, targets):
    renders = renders.permute(0, 3, 1, 2)
    targets = targets.permute(0, 3, 1, 2)

    loss = 0
    loss = torch.sum((renders - targets) ** 2)

    for i in range(num_gaussian_levels):
        targets = gaussian_func(targets)
        renders = gaussian_func(renders)

        loss += math.pow(2, i + 1) * torch.sum((renders - targets) ** 2)

    return loss

def model(initial_verts, initial_normals, offsets, optim_objects, use_vertex_offsets):

  if use_vertex_offsets: # Vertex optim
    optim_objects[0].vertices = initial_verts + offsets
  else: # Normal optim
    off = torch.stack([offsets, offsets, offsets], dim=1)
    optim_objects[0].vertices = initial_verts + off * initial_normals

  optim_scenes = generate_scenes(camLocs, optim_objects)
  renders = pyredner.render_pathtracing(scene = optim_scenes, num_samples=(64, 64), max_bounces=1)
  return renders


use_vertex_offsets = True
refinemesh = init_obj_file
face_target = 4000

i = 0
for subdiv in range(10):  
  print("Subdivision: ", subdiv)
  optim_objects = pyredner.load_obj(refinemesh, return_objects=True)
  print(optim_objects[0].vertices.shape)

  initial_verts = optim_objects[0].vertices.clone()
  initial_normals = pyredner.compute_vertex_normal(optim_objects[0].vertices, optim_objects[0].indices)

  if (use_vertex_offsets): # Vertex optim
    offsets = torch.zeros(initial_verts.shape, device=pyredner.get_device(), requires_grad=True)
  else: # Normal optim
    offsets = torch.zeros(initial_verts.shape[0], device=pyredner.get_device(), requires_grad=True)

  #lr = 0.0001 / 2^subdiv_level
  optimizer = torch.optim.Adam([offsets], lr=0.0001)
  prev_loss = 10000000000
  while True:
    optimizer.zero_grad()
    renders = model(initial_verts, initial_normals, offsets, optim_objects, use_vertex_offsets)
    loss = loss_function(renders, targets)
    print('iter: ', i, ' loss:', loss.item())

    if loss > prev_loss:
      break

    pyredner.save_obj(optim_objects[0], path + "models/output_" + str(i) + ".obj")

    loss.backward()
    optimizer.step()
    prev_loss = loss
    i += 1

  if (len(optim_objects[0].indices) > face_target):
    simplify(refinemesh, refinemesh, face_target)
    face_target += 1000

  refinemesh = path + "models/subdivision_" + str(subdiv) + ".obj"
  os.system("meshlabserver -i " + path + "models/output_" + str(i - 1) + ".obj" + " -o " + refinemesh + " -s meshlab/subdivide.mlx")
  