import torch


def rotation_from_axis(axis):
    """
    Expects Bx1x3 axis Tensor
    The axis is x, y, z with the angle being norm(axis).
    Returns a Bx3x3 rotation matix tensor
    """
    device = axis.device

    # Trick for batch_size broadcasting
    theta = torch.norm(axis, p=2, dim=2).view(-1, 1, 1)
    axis = torch.nn.functional.normalize(axis)

    ctheta = torch.cos(theta)
    stheta = torch.sin(theta)

    batch_size = axis.size()[0]
    identities = torch.eye(3).expand(batch_size, -1, -1).to(device)

    axis_cross = torch.matmul(axis.transpose(1, 2), axis)

    x = axis[:, :, 0]
    y = axis[:, :, 1]
    z = axis[:, :, 2]
    zero = torch.zeros(axis.size()[:-1]).to(x.device)
    axis_orth = torch.cat((zero, -z, y,
                           z, zero, -x,
                           -y, x, zero,), dim=1).view(-1, 3, 3)

    rotations = (ctheta * identities +
                 stheta * axis_orth +
                 (1 - ctheta) * axis_cross)
    return rotations


class Camera:
    def __init__(self, position=None, rotation=None, perspective=True, viewing_angle=30,
            near=0.01, far=100, image_size=256):

        self.position = position
        self.rotation = rotation

        self.perspective = perspective
        self.viewing_angle = viewing_angle
        self.near = near
        self.far = far
        self.image_size = image_size
