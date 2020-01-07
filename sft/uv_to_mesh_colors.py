import torch
import pyredner
import math

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

    light = pyredner.generate_quad_light(position = camera.position + offset_factor * tangent,
                                     look_at = camera0.look_at,
                                     size = torch.tensor([0.1, 0.1]),
                                     intensity = torch.tensor([light_intensity, light_intensity, light_intensity]))
    
    # Camera data for voxel carving
    #print(str(ind) + " " + str(camera.position.data[0].item()) + " " + str(camera.position.data[1].item()) + " " + str(camera.position.data[2].item()) + " " + str(camera.look_at.data[0].item()) + " " + str(camera.look_at.data[1].item()) + " " + str(camera.look_at.data[2].item()))

    scene_objs = objects
    if envmap is None:
      scene_objs = [objects[0], light]

    scenes.append(pyredner.Scene(camera = camera, objects = scene_objs, envmap=envmap))
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

target_objects = pyredner.load_obj('resources/bunny-uv.obj', return_objects=True)
print(target_objects[0].vertices.shape)

diffuse = pyredner.imread('resources/wood_diffuse.jpg')
specular = pyredner.imread('resources/wood_specular.jpg') / 100.0 #None #pyredner.imread('resources/checkerboard.png')
#normal_map = pyredner.imread('resources/GroundForest003_NRM_3K.jpg', gamma=1.0)
roughness = (1.0 - specular) / 10.0
normal_map = None
target_objects[0].material = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=specular, roughness=roughness, normal_map=normal_map)

resolution = (256, 256)
num_cameras = 1
radius = 1.0

camera0 = pyredner.automatic_camera_placement(target_objects, resolution)
camLocs = fibonacci_sphere(num_cameras, False)
camLocs = [torch.tensor([0.0, 0.0, -1.0])]

#envmap_img = pyredner.imread('grace-new.exr')
#envmap = pyredner.EnvironmentMap(envmap_img * 10.0) # *10 to make the image brighter
target_scenes = generate_scenes(camLocs, target_objects)



pyredner.render_pytorch.print_timing = False
targets = pyredner.render_pathtracing(scene = target_scenes, num_samples = (128, 0), max_bounces=1)

for ind, img in enumerate(targets):
  img = img.data.cpu()
  pyredner.imwrite(img, "output/uv-meshcolors/targets/target_" + str(ind) + ".png")

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

def model(optim_scenes, num_samples=(64, 64), max_bounces=1):
  renders = pyredner.render_pathtracing(scene = optim_scenes, num_samples=num_samples, max_bounces=max_bounces)
  return renders

res = 3
# Material Optimization
texels = torch.zeros([target_objects[0].indices.shape[0] * int(((res + 1) * (res + 2)) / 2) * 3], device = pyredner.get_device()) + 0.3
diffuse = pyredner.Texture(texels, mesh_colors_resolution=res)
specular = pyredner.Texture(texels.clone(), mesh_colors_resolution=res)
texels = torch.zeros([target_objects[0].indices.shape[0] * int(((res + 1) * (res + 2)) / 2) * 3], device = pyredner.get_device()) + 0.8
roughness = pyredner.Texture(texels, mesh_colors_resolution=res)


target_objects[0].material = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=specular, roughness=roughness, normal_map=None)
target_objects[0].material.diffuse_reflectance.texels.requires_grad = True
target_objects[0].material.specular_reflectance.texels.requires_grad = True
target_objects[0].material.roughness.texels.requires_grad = True

optimizer = torch.optim.Adam([target_objects[0].material.diffuse_reflectance.texels,
                              target_objects[0].material.specular_reflectance.texels,
                              target_objects[0].material.roughness.texels], lr=1e-2)


prev_loss = 10000000000
i = 0
while True:
  optimizer.zero_grad()
  optim_scenes = generate_scenes(camLocs, target_objects)
  renders = model(optim_scenes)
  loss = loss_function(renders, targets)
  print('iter: ', i, ' loss:', loss.item())

  for ind, img in enumerate(renders):
    img = img.data.cpu()
    pyredner.imwrite(img, "output/uv-meshcolors/renders/render_" + str(i) + "_" + str(ind) + ".png")

  if loss > prev_loss:
    break

  loss.backward()
  optimizer.step()
  target_objects[0].material.diffuse_reflectance.texels.data.clamp_(0.0, 1.0)
  target_objects[0].material.specular_reflectance.texels.data.clamp_(0.0, 1.0)
  target_objects[0].material.roughness.texels.data.clamp_(0.0, 1.0)
  prev_loss = loss
  i += 1

