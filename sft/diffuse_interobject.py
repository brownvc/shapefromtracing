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
 
def generate_scenes(camLocs,objects,envmap=None, lightLocs=None):
  scenes = []
  up = torch.tensor([0.0, 1.0, 0.0])
  offset_factor = 0.0
  light_intensity = 250.0            

  for ind, loc in enumerate(camLocs):
    camera = pyredner.Camera(position = loc,
                          look_at = torch.tensor([0.0, 0.0, 0.0]),
                          up = camera0.up,
                          fov = torch.tensor([90.0]), #fov = camera0.fov,
                          resolution = camera0.resolution)
    
    normal = camera.position.div(torch.norm(camera.position))
    tangent = torch.cross(normal, up)
    tangent = tangent.div(torch.norm(tangent))
    bitangent = torch.cross(normal, tangent)
    bitangent = bitangent.div(torch.norm(bitangent))
    
    offsets = [offset_factor * tangent]
    lightLocs = [(camera.position + offset) for offset in offsets]


    lights = [pyredner.generate_quad_light(position = lightPos,
                                     look_at = camera0.look_at,
                                     size = torch.tensor([0.1, 0.1]),
                                     intensity = torch.tensor([light_intensity, light_intensity, light_intensity])) for lightPos in lightLocs]

    scene_objects = objects
    objects.append(lights[0])

    scenes.append(pyredner.Scene(camera = camera, objects = scene_objects, envmap=None))
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
print(target_objects)

normal_map = None
diffuse = torch.tensor([0.7, 0.0, 0.0])
specular = torch.tensor([0.0, 0.0, 0.0])
roughness = torch.tensor([0.6])

diffuse = torch.tensor([0.0, 0.0, 1.0])
target_objects[0].material = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=specular, roughness=roughness, normal_map=normal_map, two_sided=True)

diffuse = torch.tensor([1.0, 0.0, 0.0])
target_objects[1].material = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=specular, roughness=roughness, normal_map=normal_map, two_sided=True)

diffuse = torch.tensor([1.0, 1.0, 1.0])
target_objects[2].material = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=specular, roughness=roughness, normal_map=normal_map, two_sided=True)

resolution = (256, 256)
num_cameras = 2
radius = float(sys.argv[5])
lightLocs = None
camera0 = pyredner.automatic_camera_placement(target_objects, resolution)
#camLocs = fibonacci_sphere(num_cameras, False)
camLocs = [torch.tensor([-0.1, 0.1, 0.1])]
target_scenes = generate_scenes(camLocs, target_objects, None, lightLocs)

max_bounces_targets = 4
max_bounces_optim = 1
use_deferred_rendering = False

# Render Targets
targets = pyredner.render_pathtracing(scene = target_scenes, num_samples = (512, 0), max_bounces=max_bounces_targets)

for ind, img in enumerate(targets):
  img = img.data.cpu()
  pyredner.imwrite(img, path + "targets/target_" + str(ind) + ".png")

target_texture = pyredner.render_albedo(scene = target_scenes, num_samples = (512, 0))

for ind, img in enumerate(target_texture):
  img = img.data.cpu()
  pyredner.imwrite(img, path + "targets/texture_" + str(ind) + ".png")

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
    return pyredner.render_pathtracing(scene = optim_scenes, num_samples=num_samples, max_bounces=max_bounces)


refinemesh = init_obj_file
res = 3

i = 0
print("Material Optimization")

optim_objects = target_objects

# Material Optimization
params = []
for ind, obj in enumerate(optim_objects):
    #obj.material.diffuse_reflectance.texels = torch.tensor([0.5, 0.5, 0.5])
    if (ind == 2):
        obj.material.diffuse_reflectance.texels = torch.tensor([1.0, 0.8, 1.0])
        #obj.material = pyredner.Material(diffuse_reflectance=diffuse, specular_reflectance=specular, roughness=roughness, normal_map=None)
        obj.material.diffuse_reflectance.texels.requires_grad = True
        params.append(obj.material.diffuse_reflectance.texels)

optimizer = torch.optim.Adam(params, lr=1e-2)

prev_loss = 10000000000
i = 0
while True:
    optimizer.zero_grad()
    renders = tex_model(target_scenes, num_samples=(64, 64), max_bounces=max_bounces_optim)
    loss = loss_function(renders, targets)
    print('iter: ', i, ' loss:', loss.item())

    textures = pyredner.render_albedo(target_scenes, num_samples=(64, 0))

    for ind, img in enumerate(renders):
        img = img.data.cpu()
        pyredner.imwrite(img, path + "renders/render_" + str(i) + ".png")

    for ind, img in enumerate(textures):
        img = img.data.cpu()
        pyredner.imwrite(img, path + "renders/texture_" + str(i) + ".png")

    # Save the texels to a file.
    #torch.save(optim_objects[0].material.diffuse_reflectance.texels, path + "mesh-colors/diffuse_" + str(subdiv) + "_" + str(i) + ".pt")
    #torch.save(optim_objects[0].material.specular_reflectance.texels, path + "mesh-colors/specular_" + str(subdiv) + "_" + str(i) + ".pt")

    loss.backward()
    optimizer.step()
    
    texture_diff = torch.sum((target_texture[0] - textures[0]) ** 2)
    
    print("Texture difference:", texture_diff)

    for obj in optim_objects:
        obj.material.diffuse_reflectance.texels.data.clamp_(0.0, 1.0)

    prev_loss = loss
    i += 1
 