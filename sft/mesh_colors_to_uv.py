import torch
import pyredner
import math
import sys

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

def generate_scenes(camLocs,objects,envmap=None, lightLocs=None):
  scenes = []
  lights = []
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
    
    offsets = [offset_factor * tangent, offset_factor * -tangent, offset_factor * bitangent, offset_factor * -bitangent, 0]
    lightLocs = [(camera.position + offset) for offset in offsets]
    #else:
    #  lightPos = lightLocs[ind]
    lights = [pyredner.generate_quad_light(position = lightPos,
                                     look_at = camera0.look_at,
                                     size = torch.tensor([0.1, 0.1]),
                                     intensity = torch.tensor([light_intensity, light_intensity, light_intensity])) for lightPos in lightLocs]
    # Camera data for voxel carving
    #print(str(ind) + " " + str(camera.position.data[0].item()) + " " + str(camera.position.data[1].item()) + " " + str(camera.position.data[2].item()) + " " + str(camera.look_at.data[0].item()) + " " + str(camera.look_at.data[1].item()) + " " + str(camera.look_at.data[2].item()))
    for light in lights:
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

# Load everything in from command line.
output_path = sys.argv[1]
target_path = sys.argv[2]
diffuse_texels_path = sys.argv[3]
specular_texels_path = sys.argv[4]

gaussian_func = getGaussianFilter()

target_objects = pyredner.load_obj(target_path, return_objects=True)
print(target_objects[0].vertices.shape)

# Load in the mesh colors info.
mesh_colors_resolution = 1
diffuse_texels = torch.tensor(torch.load(diffuse_texels_path), device=pyredner.get_device())
specular_texels = torch.tensor(torch.load(specular_texels_path), device=pyredner.get_device())
target_diffuse = pyredner.Texture(diffuse_texels, mesh_colors_resolution=mesh_colors_resolution)
target_specular = pyredner.Texture(specular_texels, mesh_colors_resolution=mesh_colors_resolution)
target_roughness = torch.tensor([0.6]) # For now, roughness is fixed.
target_objects[0].material = pyredner.Material(diffuse_reflectance=target_diffuse, specular_reflectance=target_specular, roughness=target_roughness, normal_map=None)

# Set up the cameras and lights.
resolution = (128, 128)
num_cameras = 8
radius = 2.0
lightLocs = None 
camera0 = pyredner.automatic_camera_placement(target_objects, resolution)
camLocs = fibonacci_sphere(num_cameras, False)
target_scenes = generate_scenes(camLocs, target_objects, None, lightLocs)

# Render targets.
pyredner.render_pytorch.print_timing = False
targets = pyredner.render_pathtracing(scene = target_scenes, num_samples = (128, 0), max_bounces=1)

# Write out targets.
for ind, img in enumerate(targets):
  img = img.data.cpu()
  pyredner.imwrite(img, output_path + "/targets/target_" + str(ind) + ".png")

# Loss function definition.
num_gaussian_levels = 0
def loss_function(renders, targets):
  renders = renders.permute(0, 3, 1, 2)
  targets = targets.permute(0, 3, 1, 2)

  loss = 0
  loss = torch.sum((renders - targets) ** 2.0)

  for i in range(num_gaussian_levels):
    targets = gaussian_func(targets)
    renders = gaussian_func(renders)

    loss += math.pow(2, i + 1) * torch.sum((renders - targets) ** 2)

  return loss

# Model definition.
def model(optim_scenes, num_samples=(64, 64), max_bounces=1):
  renders = pyredner.render_pathtracing(scene = optim_scenes, num_samples=num_samples, max_bounces=max_bounces)
  return renders

# Set up the uv textures.
diffuse_uv = torch.zeros(2048, 2048, 3)
specular_uv = torch.zeros(2048, 2048, 3)
diffuse_uv_texture = pyredner.Texture(torch.zeros(2048, 2048, 3), mesh_colors_resolution=0)
specular_uv_texture = pyredner.Texture(torch.zeros(2048, 2048, 3), mesh_colors_resolution=0)
roughness_uv = torch.tensor([0.6]) # for now, roughness is fixed

target_objects[0].material = pyredner.Material(diffuse_reflectance=diffuse_uv_texture, specular_reflectance=specular_uv_texture, roughness=roughness_uv, normal_map=None)
target_objects[0].material.diffuse_reflectance.mipmap.requires_grad = True
target_objects[0].material.specular_reflectance.mipmap.requires_grad = True

optimizer = torch.optim.Adam([target_objects[0].material.diffuse_reflectance.mipmap,
                              target_objects[0].material.specular_reflectance.mipmap], lr=1e-2)


prev_loss = 10000000000
i = 0
while True:
  optimizer.zero_grad()
  optim_scenes = generate_scenes(camLocs, target_objects, None, lightLocs)
  renders = model(optim_scenes)
  loss = loss_function(renders, targets)
  print('iter: ', i, ' loss:', loss.item())

  for ind, img in enumerate(renders):
    img = img.data.cpu()
    pyredner.imwrite(img, output_path + "/renders/render_" + str(i) + "_" + str(ind) + ".png")

  # Write out the textures.
  pyredner.imwrite(diffuse_uv.cpu() , output_path + "/textures/diffuse_" + str(i))
  pyredner.imwrite(specular_uv.cpu() , output_path + "/textures/specular_" + str(i))

  if loss > prev_loss:
    break

  loss.backward()
  optimizer.step()
  target_objects[0].material.diffuse_reflectance.mipmap.data.clamp_(0.0, 1.0)
  target_objects[0].material.specular_reflectance.mipmap.data.clamp_(0.0, 1.0)
  prev_loss = loss
  i += 1

