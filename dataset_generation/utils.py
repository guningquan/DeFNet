import cv2
import softgym.envs.tshirt_descriptor as td
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate

def remove_dups(camera_params, knots, coords, depth, rgb=None, zthresh=0.001):
    knots = deepcopy(knots)
    if depth.shape[0] < camera_params['default_camera']['height']:
        print('Warning: resizing depth')
        depth = cv2.resize(depth, (camera_params['default_camera']['height'], camera_params['default_camera']['width']))

    unoccluded_knots = []
    occluded_knots = []
    for i, uv in enumerate(knots):
        u_f, v_f = uv[0], uv[1]
        if np.isnan(u_f) or np.isnan(v_f):
            continue
        u, v = int(np.rint(u_f)), int(np.rint(v_f))

        if u < 0 or v < 0 or u >= depth.shape[0] or v >= depth.shape[1]:
            # pixel is outside of image bounds
            knots[i] = [float('NaN'), float('NaN')]
            continue
        
        d = depth[u, v]

        # Get depth into world coordinates
        proj_coords = td.uv_to_world_pos(camera_params, depth, u_f, v_f, particle_radius=0, on_table=False)[0:3]
        z_diff = proj_coords[1] - coords[i][1]

        # Check is well projected xyz point
        if z_diff > zthresh:
            # invalidate u, v and continue
            occluded_knots.append(deepcopy(knots[i]))
            knots[i] = [float('NaN'), float('NaN')]
            continue

        unoccluded_knots.append(deepcopy(knots[i]))
    
    # print("unoccluded knots: ", len(unoccluded_knots))

    if False: # debug visualization
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # for i, (u, v) in enumerate(knots):
        #     c = 'r' if np.isnan(u) or np.isnan(v) else 'b'
        #     ax.scatter(coords[i, 0], coords[i, 2], coords[i, 1], s=1, c=c)
        # plt.show()

        fig, ax = plt.subplots(1, 3, dpi=200)
        ax[0].set_title('depth')
        ax[0].imshow(depth)
        ax[1].set_title('occluded points\nin red')
        ax[1].imshow(depth)
        if occluded_knots != []:
            occluded_knots = np.array(occluded_knots)
            ax[1].scatter(occluded_knots[:, 1], occluded_knots[:, 0], marker='.', s=1, c='r', alpha=0.4)
        ax[2].imshow(depth)
        ax[2].set_title('unoccluded points\nin blue')
        unoccluded_knots = np.array(unoccluded_knots)
        ax[2].scatter(unoccluded_knots[:, 1], unoccluded_knots[:, 0], marker='.', s=1, alpha=0.4)
        plt.show()
        
    return knots, np.array(unoccluded_knots)

def spatial_softmax(heatmap):
    return F.softmax(heatmap.flatten(-2), -1).view(heatmap.shape)

def softargmax_coords(heatmap, normalize = True):

    # no normalization
    if(normalize):
        prob = spatial_softmax(heatmap)
    else:
        prob = heatmap

    H, W = heatmap.shape[-2], heatmap.shape[-1]
    #c = (torch.linspace(-1.0, 1.0, W).to(heatmap.device) * prob.sum(-2)).sum(-1, keepdim=True)
    #r = (torch.linspace(-1.0, 1.0, H).to(heatmap.device) * prob.sum(-1)).sum(-1, keepdim=True)
    c = (torch.linspace(0., 1.0, W).to(heatmap.device) * prob.sum(-2)).sum(-1, keepdim=True)
    r = (torch.linspace(0., 1.0, H).to(heatmap.device) * prob.sum(-1)).sum(-1, keepdim=True)

    return torch.cat([r,c], dim=-1)


def get_harris(mask, thresh=0.2):
    """Harris corner detector
    Params
    ------
        - mask: np.float32 image of 0.0 and 1.0
        - thresh: threshold for filtering small harris values    Returns
    -------
        - harris: np.float32 array of
    """
    # Params for cornerHarris: 
    # mask - Input image, it should be grayscale and float32 type.
    # blockSize - It is the size of neighbourhood considered for corner detection
    # ksize - Aperture parameter of Sobel derivative used.
    # k - Harris detector free parameter in the equation.
    # https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345
    harris = cv2.cornerHarris(mask, blockSize=5, ksize=5, k=0.01)
    harris[harris<thresh*harris.max()] = 0.0 # filter small values
    harris[harris!=0] = 1.0
    harris_dilated = cv2.dilate(harris, kernel=np.ones((7,7),np.uint8))
    harris_dilated[mask == 0] = 0
    return harris_dilated

def plot_flow(ax, flow_im, skip=1):
    """Plot flow as a set of arrows on an existing axis.
    """
    h,w,c = flow_im.shape
    bg = np.zeros((h, w, 3))
    #bg[:,:] = np.array([1.0,0.0,0.0])
    ax.imshow(bg)
    ys, xs, _ = np.where(flow_im != 0)

    # # white arrows
    # ax.quiver(xs[::skip], ys[::skip],
    #             flow_im[ys[::skip], xs[::skip], 1], flow_im[ys[::skip], xs[::skip], 0], 
    #             alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)

    # old skip code
    # flu = flow_im[ys[::skip], xs[::skip], 1]
    # flv = flow_im[ys[::skip], xs[::skip], 0]
    # mags = np.linalg.norm(flow_im[ys[::skip], xs[::skip], :], axis=1)
    # norm = mpl.colors.Normalize()
    # norm.autoscale(mags)
    # cm = mpl.cm.cividis
    # ax.quiver(xs[::skip], ys[::skip], flu, flv, alpha=0.8, color=cm(norm(mags)), 
    #             angles='xy', scale_units='xy', scale=1, width=0.01)

    # sample instead of skip, skip param is percentage (0 - 1)
    n = len(xs)
    skip = np.clip(skip,0.0,1.0)
    inds = np.random.choice(np.arange(n), size=int(n*skip), replace=False)
    flu = flow_im[ys[inds], xs[inds], 1]
    flv = flow_im[ys[inds], xs[inds], 0]
    mags = np.linalg.norm(flow_im[ys[inds], xs[inds], :], axis=1)
    norm = mpl.colors.Normalize()
    norm.autoscale(mags)
    cm = mpl.cm.cividis

    ax.quiver(xs[inds], ys[inds], flu, flv, alpha=0.8, color=cm(norm(mags)), 
                angles='xy', scale_units='xy', scale=1, width=0.025, headwidth=5.)

def flow_afwarp(flow_im, angle, dx, dy):
    """Affine transformation of flow image and per-pixel flow vectors.    

    Parameters
    -----------
    flow_im : np.ndarray
        Flow image
        
    angle : float
        Angle in degrees of rotation    
        
    dx : int
        delta x translation    
        
    dy : int
        delta y translation    

    Returns
    -------
    flow_pxrot: np.ndarray
        Flow image transformed both on image level and per-pixel level.
    """
    # Translate and rotate image
    h, w, _ = flow_im.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    M[:2, 2] += [dx, dy]
    flow_im_tf = cv2.warpAffine(flow_im, M, (w, h), flags=cv2.INTER_NEAREST)

    # Rotate per-pixel flow values
    R = M[:2, :2].T
    px = np.reshape(flow_im_tf, (h*w, 2)).T # (2 x h*w) pixel values
    flow_px_tf = np.reshape((R @ px).T, (h, w, 2))
    return flow_px_tf

def avg_flow_angle(flow_im):
    """
    Get the average flow angle for all flow vectors in the image. 
    """
    h, w, c = flow_im.shape
    flow_flat = np.reshape(flow_im, (h*w, c))
    flow_vecs = flow_flat[np.all(flow_flat != 0, axis=1)]
    # https://stackoverflow.com/questions/491738/how-do-you-calculate-the-average-of-a-set-of-circular-data
    avg_angle = np.arctan2(np.sum(flow_vecs[:, 0]), np.sum(flow_vecs[:, 1]))
    return avg_angle


def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def get_flow_place_pt(u,v, flow):
    flow_u_idxs = np.argwhere(flow[0,:,:])
    flow_v_idxs = np.argwhere(flow[1,:,:])
    nearest_u_idx = flow_u_idxs[((flow_u_idxs - [u,v])**2).sum(1).argmin()]
    nearest_v_idx = flow_v_idxs[((flow_v_idxs - [u,v])**2).sum(1).argmin()]

    flow_u = flow[0,nearest_u_idx[0],nearest_u_idx[1]]
    flow_v = flow[1,nearest_v_idx[0],nearest_v_idx[1]]

    new_u = u + flow_u
    new_v = v + flow_v

    return new_u,new_v


def get_gaussian(u, v, sigma=5, size=200):
    x0, y0 = u, v
    num = torch.arange(size).float()
    x, y = num, num
    gx = torch.exp(-(x-x0)**2/(2*sigma**2))
    gy = torch.exp(-(y-y0)**2/(2*sigma**2))
    g = torch.outer(gx, gy)
    g = (g - g.min())/(g.max() - g.min())
    g = g.unsqueeze(0)

    return g

def action_viz(img, action, unmasked_pred):
    ''' img: cv2 image
        action: pick1, place1, pick2, place2
        unmasked_pred: pick1_pred, pick2_pred'''
    pick1, place1, pick2, place2 = action
    pick1_pred, pick2_pred = unmasked_pred

    # draw the original predictions
    u,v = pick1_pred
    cv2.drawMarker(img, (int(v), int(u)), (0,0,200), markerType=cv2.MARKER_STAR, 
                    markerSize=10, thickness=2, line_type=cv2.LINE_AA)
    u,v = pick2_pred
    cv2.drawMarker(img, (int(v), int(u)), (0,0,200), markerType=cv2.MARKER_STAR, 
                    markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    # draw the masked action
    u1,v1 = pick1
    u2,v2 = place1
    cv2.circle(img, (int(v1),int(u1)), 6, (0,200,0), 2)
    cv2.arrowedLine(img, (int(v1),int(u1)), (int(v2),int(u2)), (0, 200, 0), 2)
    u1,v1 = pick2
    u2,v2 = place2
    cv2.circle(img, (int(v1),int(u1)), 6, (0,200,0), 2)
    cv2.arrowedLine(img, (int(v1),int(u1)), (int(v2),int(u2)), (0, 200, 0), 2)

    return img


# RAFT
class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
