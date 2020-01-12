import torch
import pyredner
import math
import os
import sys
from itertools import repeat

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

  simp_script = open("meshlab/simplify.mlx", "w")
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
  light_intensity = 5000.0            

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
face_max = int(sys.argv[5])

# Load Target model
target_objects = pyredner.load_obj(target_obj_file, return_objects=True)
print(target_objects[0].vertices.shape)

diffuse = pyredner.imread('resources/wood_diffuse_low_res.jpg')
specular_uv = pyredner.imread('resources/wood_specular_low_res.jpg') / 100.0
roughness = (1.0 - specular_uv) / 10.0
normal_map = None
target_objects[0].material = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=specular_uv, roughness=roughness, normal_map=normal_map)


resolution = (128, 128)
num_cameras = 32
radius = 1.4

camera0 = pyredner.automatic_camera_placement(target_objects, resolution)
camLocs = fibonacci_sphere(num_cameras, False)
target_scenes = generate_scenes(camLocs, target_objects, None)

# Binary mask
#target_objects[0].material = pyredner.Material(diffuse_reflectance=torch.tensor([1.0, 1.0, 1.0]))
#mask_scenes = generate_scenes(camLocs, target_objects, envmap)

# Binary mask
#masks = pyredner.render_albedo(scene = mask_scenes, num_samples = (128, 0))

seeds = list(repeat(7, num_cameras))

# Render Targets
targets = pyredner.render_pathtracing(scene = target_scenes, num_samples = (128, 0), max_bounces=1, seed=seeds)

for ind, img in enumerate(targets):
  img = img.data.cpu()
  pyredner.imwrite(img, path + "targets/target_" + str(ind) + ".png")

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


def tex_model(optim_scenes, num_samples=(64, 64), max_bounces=1):
  renders = pyredner.render_pathtracing(scene = optim_scenes, num_samples=num_samples, max_bounces=max_bounces)
  return renders

def geom_model(initial_verts, initial_normals, offsets, optim_objects, use_vertex_offsets, num_samples=(64, 64), max_bounces=1):
  if use_vertex_offsets: # Vertex optim
    optim_objects[0].vertices = initial_verts + offsets
  else: # Normal optim
    off = torch.stack([offsets, offsets, offsets], dim=1)
    optim_objects[0].vertices = initial_verts + off * initial_normals

  #torch.set_printoptions(profile="full")
  #print(optim_objects[0].vertices)

  optim_scenes = generate_scenes(camLocs, optim_objects)
  renders = pyredner.render_pathtracing(scene = optim_scenes, num_samples=num_samples, max_bounces=max_bounces, seed=seeds)
  return renders


use_vertex_offsets = True
refinemesh = init_obj_file
res = 3

optim_objects = pyredner.load_obj(refinemesh, return_objects=True)
print(optim_objects[0].vertices.shape)


i = 0
for subdiv in range(100):
  print("Subdivision: ", subdiv)

  # Material Optimization
  print("Material Optimization")
  optim_objects = pyredner.load_obj(refinemesh, return_objects=True)

  texels = torch.zeros([optim_objects[0].indices.shape[0] * int(((res + 1) * (res + 2)) / 2) * 3], device = pyredner.get_device()) + 0.3
  diffuse = pyredner.Texture(texels, mesh_colors_resolution=res)
  specular = pyredner.Texture(texels.clone(), mesh_colors_resolution=res)
  roughness = (1.0 - specular_uv) / 10.0

  #texels = torch.zeros([optim_objects[0].indices.shape[0] * int(((res + 1) * (res + 2)) / 2) * 3], device = pyredner.get_device()) + 0.8
  #roughness = pyredner.Texture(texels, mesh_colors_resolution=res)

  optim_objects[0].material = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=specular, roughness=roughness, normal_map=None)
  optim_objects[0].material.diffuse_reflectance.texels.requires_grad = True
  optim_objects[0].material.specular_reflectance.texels.requires_grad = True
  #optim_objects[0].material.roughness.texels.requires_grad = True

  optimizer = torch.optim.Adam([optim_objects[0].material.diffuse_reflectance.texels,
                                optim_objects[0].material.specular_reflectance.texels], lr=0.01)

  prev_loss = 10000000000
  for mat_ind in range(200):
    optimizer.zero_grad()
    optim_scenes = generate_scenes(camLocs, optim_objects)
    renders = tex_model(optim_scenes)
    loss = loss_function(renders, targets)
    print('iter: ', i, ' loss:', loss.item())

    for ind, img in enumerate(renders):
      if ind == 5:
        img = img.data.cpu()
        pyredner.imwrite(img, path + "renders/render_" + str(i) + ".png")

    #if loss > prev_loss:
      #break

    loss.backward()
    optimizer.step()
    optim_objects[0].material.diffuse_reflectance.texels.data.clamp_(0.0, 1.0)
    optim_objects[0].material.specular_reflectance.texels.data.clamp_(0.0, 1.0)
    #optim_objects[0].material.roughness.texels.data.clamp_(0.0, 1.0)
    prev_loss = loss
    i += 1

  optim_objects[0].material.diffuse_reflectance.texels.detach()
  optim_objects[0].material.specular_reflectance.texels.detach()
  #optim_objects[0].material.roughness.texels.detach()

  torch.save(optim_objects[0].material.diffuse_reflectance.texels, path + "mesh-colors/diffuse_" + str(subdiv) + ".pt")
  torch.save(optim_objects[0].material.specular_reflectance.texels, path + "mesh-colors/specular_" + str(subdiv) + ".pt")
  #torch.save(optim_objects[0].material.roughness.texels, path + "mesh-colors/roughness_" + str(subdiv) + ".pt")

  #optim_objects[0].material.diffuse_reflectance.texels = torch.load(path + "mesh-colors/diffuse.pt")
  #optim_objects[0].material.specular_reflectance.texels = torch.load(path + "mesh-colors/specular.pt")
  #optim_objects[0].material.roughness.texels = torch.load(path + "mesh-colors/roughness.pt")


  # Geometry Optimization
  print(optim_objects[0].vertices.shape)
  print("Geometry optimization")

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
    renders = geom_model(initial_verts, initial_normals, offsets, optim_objects, use_vertex_offsets)
    loss = loss_function(renders, targets)
    print('iter: ', i, ' loss:', loss.item())

    for ind, img in enumerate(renders):
      if (ind == 5):
        img = img.data.cpu()
        pyredner.imwrite(img, path + "renders/geom_" + str(i) + ".png")

    if loss > prev_loss:
      break

    pyredner.save_obj(optim_objects[0], path + "models/output_" + str(i) + ".obj")

    loss.backward()
    optimizer.step()
    prev_loss = loss
    i += 1

  res_mesh = path + "models/output_" + str(i - 1) + ".obj"

  if (len(optim_objects[0].indices) > face_target):
    print("Simplifying ", res_mesh)
    simplify(res_mesh, res_mesh, face_target)

    if face_target < face_max:
      face_target += 2048
  print("Subdividing", res_mesh, "into", refinemesh)
  refinemesh = path + "models/subdivision_" + str(subdiv) + ".obj"
  os.system("meshlabserver -i " + res_mesh + " -o " + refinemesh + " -s meshlab/subdivide.mlx")
  
pyredner.save_obj(optim_objects[0], path + "models/output_.obj")
