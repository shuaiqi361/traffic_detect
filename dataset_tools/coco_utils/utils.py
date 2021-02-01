import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cv2
from pycocotools import mask as cocomask
from skimage import measure


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def get_connected_polys_with_measure(polygons, img_hw, n_vertices, closing_max_kernel=50, tolerance=0):
    h_img, w_img = img_hw
    rles = cocomask.frPyObjects(polygons, h_img, w_img)
    rle = cocomask.merge(rles)  # ['counts'].decode('ascii')
    m = cocomask.decode(rle)  # draw mask on binary image
    padded_binary_mask = np.pad(m, pad_width=1, mode='constant', constant_values=0)
    if len(polygons) > 1:
        pads = 5
        while True:
            kernel = np.ones((pads, pads), np.uint8)
            bg_closed = cv2.morphologyEx(padded_binary_mask, cv2.MORPH_CLOSE, kernel)
            obj_contours = measure.find_contours(bg_closed, 0.5)
            if len(obj_contours) > 1:
                pads += 5
            else:
                contour = np.clip(np.subtract(obj_contours[0], 1), a_min=0, a_max=None)  # recover the original contours without padding
                contour = close_contour(contour)
                contour = measure.approximate_polygon(contour, tolerance)
                # if len(contour) < 3:  # reduce to a line segment
                #     continue
                contour = np.flip(contour, axis=1)
                return np.ndarray.flatten(contour).tolist()

            if pads > closing_max_kernel:
                obj_contours = sorted(obj_contours, key=cv2.contourArea)
                contour = np.clip(np.subtract(obj_contours[-1], 1), a_min=0, a_max=None)
                contour = close_contour(contour)
                contour = measure.approximate_polygon(contour, tolerance)
                contour = np.flip(contour, axis=1)
                return np.ndarray.flatten(contour).tolist()  # The largest piece

    else:
        obj_contours = measure.find_contours(padded_binary_mask, 0.5)
        contour = np.clip(np.subtract(obj_contours[0], 1), a_min=0, a_max=None)  # recover the original contours without padding
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        # if len(contour) < 3:  # reduce to a line segment
        #     continue
        contour = np.flip(contour, axis=1)
        return np.ndarray.flatten(contour).tolist()


def get_connected_polygon_with_mask(polygons, img_hw, n_vertices, closing_max_kernel=50):
    h_img, w_img = img_hw
    if len(polygons) > 1:
        bg = np.zeros((h_img, w_img, 1), dtype=np.uint8)
        for poly in polygons:
            len_poly = len(poly)
            vertices = np.zeros((1, len_poly // 2, 2), dtype=np.int32)
            for i in range(len_poly // 2):
                vertices[0, i, 0] = int(np.round(poly[2 * i]))
                vertices[0, i, 1] = int(np.round(poly[2 * i + 1]))
            cv2.drawContours(bg, vertices, color=255, contourIdx=-1, thickness=-1)

        pads = 5
        while True:
            kernel = np.ones((pads, pads), np.uint8)
            bg_closed = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)
            obj_contours, _ = cv2.findContours(bg_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(obj_contours) > 1:
                pads += 5
            else:
                return np.ndarray.flatten(obj_contours[0]).tolist()

            if pads > closing_max_kernel:
                obj_contours = sorted(obj_contours, key=cv2.contourArea)
                return np.ndarray.flatten(obj_contours[-1]).tolist()  # The largest piece

    else:
        if len(polygons[0]) <= n_vertices:
            return polygons[0]
        bg = np.zeros((h_img, w_img, 1), dtype=np.uint8)
        for poly in polygons:
            len_poly = len(poly)
            vertices = np.zeros((1, len_poly // 2, 2), dtype=np.int32)
            for i in range(len_poly // 2):
                vertices[0, i, 0] = int(np.round(poly[2 * i]))
                vertices[0, i, 1] = int(np.round(poly[2 * i + 1]))
            cv2.drawContours(bg, vertices, color=255, contourIdx=-1, thickness=-1)

        obj_contours, _ = cv2.findContours(bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        return np.ndarray.flatten(obj_contours[0]).tolist()


def get_connected_polygon_coco_mask(polygons, img_hw, closing_max_kernel=50):
    h_img, w_img = img_hw
    rles = cocomask.frPyObjects(polygons, h_img, w_img)
    rle = cocomask.merge(rles)  # ['counts'].decode('ascii')
    m = cocomask.decode(rle)  # draw mask on binary image
    if len(polygons) > 1:
        pads = 5
        while True:
            kernel = np.ones((pads, pads), np.uint8)
            bg_closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            obj_contours, _ = cv2.findContours(bg_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(obj_contours) > 1:
                pads += 5
            else:
                return np.ndarray.flatten(obj_contours[0]).tolist()

            if pads > closing_max_kernel:
                obj_contours = sorted(obj_contours, key=cv2.contourArea)
                return np.ndarray.flatten(obj_contours[-1]).tolist()  # The largest piece

    else:
        obj_contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        return np.ndarray.flatten(obj_contours[0]).tolist()


def get_connected_polygon_using_mask(polygons, img_hw, n_vertices, closing_max_kernel=50):
    h_img, w_img = img_hw
    rles = cocomask.frPyObjects(polygons, h_img, w_img)
    rle = cocomask.merge(rles)  # ['counts'].decode('ascii')
    m = cocomask.decode(rle)  # draw mask on binary image
    is_simple = 0  # 0 multiple parts, 1 simple, 2 complicated
    if len(polygons) > 1:
        pads = 5
        while True:
            kernel = np.ones((pads, pads), np.uint8)
            bg_closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            obj_contours, _ = cv2.findContours(bg_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # obj_contours, _ = cv2.findContours(bg_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
            if len(obj_contours) > 1:
                pads += 5
            else:
                return np.ndarray.flatten(obj_contours[0]).tolist(), is_simple

            if pads > closing_max_kernel:
                obj_contours = sorted(obj_contours, key=cv2.contourArea)
                return np.ndarray.flatten(obj_contours[-1]).tolist(), is_simple  # The largest piece

    else:
        if len(polygons[0]) <= n_vertices:
            is_simple = 1
            return polygons[0], is_simple

        obj_contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # obj_contours, _ = cv2.findContours(bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        is_simple = 2

        return np.ndarray.flatten(obj_contours[0]).tolist(), is_simple


def get_connected_polygon(polygons, img_hw, closing_max_kernel=50):
    h_img, w_img = img_hw
    if len(polygons) > 1:
        bg = np.zeros((h_img, w_img, 1), dtype=np.uint8)
        for poly in polygons:
            len_poly = len(poly)
            vertices = np.zeros((1, len_poly // 2, 2), dtype=np.int32)
            for i in range(len_poly // 2):
                vertices[0, i, 0] = int(poly[2 * i])
                vertices[0, i, 1] = int(poly[2 * i + 1])
            cv2.drawContours(bg, vertices, color=255, contourIdx=-1, thickness=-1)

        pads = 5
        while True:
            kernel = np.ones((pads, pads), np.uint8)
            bg_closed = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel)
            obj_contours, _ = cv2.findContours(bg_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(obj_contours) > 1:
                pads += 5
            else:
                return np.ndarray.flatten(obj_contours[0]).tolist()

            if pads > closing_max_kernel:
                obj_contours = sorted(obj_contours, key=cv2.contourArea)
                return np.ndarray.flatten(obj_contours[-1]).tolist()  # The largest piece

    else:
        # continue
        return polygons[0]


def uniform_sample_segment(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum - 1, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    pgtnp_px2 = pgtnp_px2[:-1]
    pnum = pnum - 1
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
    for i in range(pnum):
        if edgenum[i] == 0:
            edgenum[i] = 1

    # after round, it may has 1 or 2 mismatch
    edgenumsum = np.sum(edgenum)
    if edgenumsum != newpnum:

        if edgenumsum > newpnum:

            id = -1
            passnum = edgenumsum - newpnum
            while passnum > 0:
                edgeid = edgeidxsort_p[id]
                if edgenum[edgeid] > passnum:
                    edgenum[edgeid] -= passnum
                    passnum -= passnum
                else:
                    passnum -= edgenum[edgeid] - 1
                    edgenum[edgeid] -= edgenum[edgeid] - 1
                    id -= 1
        else:
            id = -1
            edgeid = edgeidxsort_p[id]
            edgenum[edgeid] += newpnum - edgenumsum

    assert np.sum(edgenum) == newpnum

    psample = []
    for i in range(pnum):
        pb_1x2 = pgtnp_px2[i:i + 1]
        pe_1x2 = pgtnext_px2[i:i + 1]

        pnewnum = edgenum[i]
        wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

        pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
        psample.append(pmids)

    psamplenp = np.concatenate(psample, axis=0)
    return psamplenp


def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        if np.sum(edgelen_p) < 1e-6:
            print(pgtnp_px2, pgtnp_px2.shape)
            exit()
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]


def align_original_polygon(resampled, original):
    """
    For each vertex in the resampled polygon, find the closest vertex in the original polygon, and align them.
    :param resampled:
    :param original:
    :return:
    """
    poly = np.zeros(shape=resampled.shape)
    num = len(resampled)
    for i in range(num):
        vertex = resampled[i]
        poly[i] = closest_node(vertex, original)

    return poly


def turning_angle_resample(polygon, n_vertices):
    """
    :param polygon: ndarray with shape (n_vertices, 2)
    :param n_vertices: resulting number of vertices of the polygon
    :return:
    """
    assert n_vertices >= 3
    polygon = polygon.reshape((-1, 2))
    original_num_vertices = len(polygon)
    shape_poly = polygon.copy()

    if original_num_vertices == n_vertices:
        return polygon
    elif original_num_vertices < n_vertices:
        while len(shape_poly) < n_vertices:
            max_idx = -1
            max_dist = 0.
            insert_coord = np.array([-1, -1])
            for i in range(len(shape_poly)):
                x1, y1 = shape_poly[i, :]
                x2, y2 = shape_poly[(i + 1) % len(shape_poly), :]  # connect to the first vertex
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if dist > max_dist:
                    max_idx = (i + 1) % len(shape_poly)
                    max_dist = dist
                    insert_coord[0] = (x1 + x2) / 2.
                    insert_coord[1] = (y1 + y2) / 2.

            shape_poly = np.insert(shape_poly, max_idx, insert_coord, axis=0)

        return shape_poly
    else:
        turning_angles = [0] * original_num_vertices
        for i in range(original_num_vertices):
            a_p = shape_poly[i, :]
            b_p = shape_poly[(i + 1) % original_num_vertices, :]
            c_p = shape_poly[(i + 2) % original_num_vertices, :]
            turning_angles[(i + 1) % original_num_vertices] = calculate_turning_angle(a_p, b_p, c_p)

        print('Turning angles:', turning_angles)
        print(np.argsort(turning_angles).tolist())
        idx = np.argsort(turning_angles).tolist()[0:n_vertices]
        # get the indices of vertices from turning angle list from small to large, small means sharper turns
        new_poly = np.zeros((0, 2))
        for i in range(original_num_vertices):
            if i in idx:
                new_poly = np.concatenate((new_poly, shape_poly[i].reshape((1, -1))), axis=0)

        return new_poly


def calculate_turning_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    # print('Angle:', angle)

    return min(angle / np.pi * 180, 360 - angle / np.pi * 180)


def check_clockwise_polygon(polygon):
    """
    sum over edges: sum (x2 − x1)(y2 + y1)
    :param polygon:
    :return:
    """
    n_vertices = polygon.shape[0]
    sum_edges = 0
    for i in range(n_vertices):
        x1 = polygon[i % n_vertices, 0]
        y1 = polygon[i % n_vertices, 1]
        x2 = polygon[(i + 1) % n_vertices, 0]
        y2 = polygon[(i + 1) % n_vertices, 1]

        sum_edges += (x2 - x1) * (y2 + y1)

    if sum_edges > 0:
        return True
    else:
        return False


def cross_product(p1, p2):
    """
    :param p1: point_1 (x1, y1)
    :param p2: point_2 (x2, y2)
    :return: float number
    """
    return p1[0] * p2[1] - p2[0] * p1[1]


def direction(p1, p2, p3):
    """
    :param p1:
    :param p2:
    :param p3:
    :return: returns the cross product of vector p1p3 and p1p2
    If p1p3 is clockwise from p1p2 it returns positive value;
    if p1p3 is counter-clockwise from p1p2 it returns negative value
    if p1 p2 and p3 are collinear it returns 0
    """
    p1p3 = (p3[0] - p1[0], p3[1] - p1[1])
    p1p2 = (p2[0] - p1[0], p2[1] - p1[1])

    return cross_product(p1p3, p1p2)


def on_segment(p1, p2, p):
    """
    :param p1:
    :param p2:
    :param p:
    :return: return if p on line segment p1p2
    """
    return min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])


def intersect(p1, p2, p3, p4):
    """
    :param p1:
    :param p2:
    :param p3:
    :param p4:
    :return: check if ling segment p1p2 intersects p3p4
    """
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False


if __name__ == '__main__':
    a = np.array([0, -1])
    b = np.array([0, 0])
    c = np.array([-1, -1])

    alp = calculate_turning_angle(a, b, c)
    print('Turning angle: ', alp)
